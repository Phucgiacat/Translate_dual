import torch
import torch.nn as nn
from torch.nn import functional as F
from fairseq.modules import multihead_attention
from fairseq.models import FairseqEncoder
from fairseq.models.aggregators import MaxPoolingAggregator, MeanAggregator, GCNAggregator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over graph nodes with optional root node bias.

    Thay thế MaxPool: thay vì lấy max, ta học weighted sum qua tất cả nodes.
    Root node (index 0) trong AMR luôn là predicate chính của câu, nên được
    thêm một learnable bias vào attention logit để ưu tiên nó.

    Args:
        dim (int): chiều của node representation.
        root_bias (bool): nếu True, thêm learnable scalar bias cho root node.
    """
    def __init__(self, dim, root_bias=True):
        super().__init__()
        self.score_proj = nn.Linear(dim, 1, bias=True)
        self.use_root_bias = root_bias
        if root_bias:
            # Learnable scalar chỉ cộng vào logit của root node (index 0)
            self.root_logit_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.score_proj.weight)

    def forward(self, node_reps, node_mask=None):
        """
        Args:
            node_reps  (Tensor): [B, N, dim]  -- node representations
            node_mask  (Tensor): [N, B] bool  -- True nghĩa là padding node
                                  (cùng convention với encoder_padding_mask)
        Returns:
            pooled (Tensor): [B, dim]
        """
        # Tính attention score: [B, N, 1] -> [B, N]
        scores = self.score_proj(node_reps).squeeze(-1)

        if self.use_root_bias:
            # Root node = index 0; cộng bias chỉ vào cột đầu tiên
            scores = scores.clone()  # tránh in-place trên autograd
            scores[:, 0] = scores[:, 0] + self.root_logit_bias.squeeze()

        if node_mask is not None:
            # node_mask là [N, B]; transpose thành [B, N] trước khi mask
            mask_BN = node_mask.t().bool()
            scores = scores.masked_fill(mask_BN, float('-inf'))

        weights = torch.softmax(scores, dim=-1)  # [B, N]
        # Xử lý trường hợp tất cả nodes đều bị padding (softmax(-inf) = nan)
        weights = torch.nan_to_num(weights, nan=0.0)

        pooled = (weights.unsqueeze(-1) * node_reps).sum(dim=1)  # [B, dim]
        return pooled


class Highway(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Highway, self).__init__()
        if torch.cuda.is_available():
            self.H = nn.Linear(input_dim, output_dim).cuda()
            self.T = nn.Linear(input_dim, output_dim).cuda()
        else:
            self.H = nn.Linear(input_dim, output_dim)
            self.T = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.sigmoid(self.H(x))
        t = self.relu(self.T(x))
        r = h * t + (1 - h) * x
        return r


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class GraphEncoder(FairseqEncoder):
    def __init__(self, dictionary, embedding_dim, output_dim, n_layers=2,
                 dropout=0.2, pad_idx=1, aggr='maxpooling', concat=True, n_highway=1, direction='bi'
                 ):
        super(GraphEncoder, self).__init__(dictionary)
        self.hidden_dim = 2 * embedding_dim
        self.num_embeddings = len(dictionary)
        self.concat = True if concat == "True" else False
        self.n_layers = n_layers
        self.pad_idx = pad_idx
        self.direction = direction
        self.embeddings = nn.Embedding(self.num_embeddings, embedding_dim, padding_idx=pad_idx)
        # self.compress_layer = nn.Linear(self.embedding_dim + self.embedding_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.n_highways = n_highway
        if self.n_highways > 0:
            self.edge_highways = nn.ModuleList([Highway(embedding_dim, embedding_dim) for _ in range(n_highway)])
            self.indice_highways = nn.ModuleList([Highway(embedding_dim, embedding_dim) for _ in range(n_highway)])
        else:
            self.edge_highways = None
            self.indice_highways = None

        if aggr == "mean":
            self.aggregator = MeanAggregator
        elif aggr == "maxpooling":
            self.aggregator = MaxPoolingAggregator
        elif aggr == "gcn":
            self.aggregator = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", aggr)

        if self.direction in ['fw', "bi"]:
            self.fw_aggregators = nn.ModuleList()
            self._build_aggregators(self.fw_aggregators)
        if self.direction in ['bw', 'bi']:
            self.bw_aggregators = nn.ModuleList()
            self._build_aggregators(self.bw_aggregators)

        if self.concat is True:
            if self.direction == 'bi':
                self.fc_out = nn.Linear(self.hidden_dim * 4, output_dim, bias=False)
            else:
                self.fc_out = nn.Linear(self.hidden_dim * 2, output_dim, bias=False)
        else:
            if self.direction == 'bi':
                self.fc_out = nn.Linear(self.hidden_dim * 2, output_dim, bias=False)
            else:
                self.fc_out = nn.Linear(self.hidden_dim, output_dim, bias=False)

        # Attention pooling thay thế MaxPool; pool trên không gian sau fc_out
        self.attn_pool = AttentionPooling(output_dim, root_bias=True)

    def __create_mask(self, graph_tokens):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        batch_size, num_nodes = nodes.shape
        node_mask = node_feats.eq(self.pad_idx)[:-1].reshape(batch_size, -1).t()
        if torch.cuda.is_available():
            padding_indx = int(torch.max(torch.max(in_indices, dim=0)[0]).cpu().numpy().tolist())
        else:
            padding_indx = int(torch.max(torch.max(in_indices, dim=0)[0]).numpy().tolist())
        in_indice_mask = in_indices != padding_indx
        out_indice_mask = out_indices != padding_indx
        return {
            "node_mask": node_mask,
            "in_neigh_mask": in_indice_mask,
            "out_neigh_mask": out_indice_mask,
        }

    def _build_aggregators(self, aggregators, max_hops=10):
        max_hops = min(max_hops, self.n_layers)
        self.max_hops = max_hops
        for i in range(max_hops):
            if i == 0:
                dim_mul = 1
            else:
                dim_mul = 2 if self.concat else 1
            aggregator = self.aggregator(input_dim=dim_mul * self.hidden_dim, output_dim=self.hidden_dim,
                                         input_neigh_dim=dim_mul * self.hidden_dim,
                                         dropout=self.dropout_rate,
                                         bias=False,
                                         activation=torch.relu,
                                         concat=self.concat)
            aggregators.append(aggregator)

    def __prepare_info(self, graph_tokens, graph_lengths):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        mask = self.__create_mask(graph_tokens)
        node_mask, in_neigh_mask, out_neigh_mask = mask.values()
        batch_size, num_nodes = nodes.shape

        nodes = nodes.reshape(-1)  # [batch_size * num_nodes]
        edges = edges.reshape(-1)  # [ batch_size *num_node]

        # NB: total_nodes = batch_size * num_nodes
        # [total_nodes, embedding_dim]
        embedded_node_reps = self.dropout(self.embeddings(node_feats))
        embedded_edge_reps = self.dropout(self.embeddings(edge_feats))
        if self.n_highways > 0:
            for layer in range(self.n_highways):
                embedded_node_reps = self.indice_highways[layer](embedded_node_reps)
                embedded_edge_reps = self.edge_highways[layer](embedded_edge_reps)
        # [total_nodes, neigh_size]
        if self.direction in ['fw', 'bi']:
            fw_neigh_sampled_indices = nn.Embedding.from_pretrained(out_indices, )(nodes).type(torch.long)
            fw_neigh_sampled_edges = nn.Embedding.from_pretrained(out_edges, )(edges).type(torch.long)

            fw_edge_hidden = nn.Embedding.from_pretrained(embedded_edge_reps, )(fw_neigh_sampled_edges)
            fw_indice_hidden = nn.Embedding.from_pretrained(embedded_node_reps, )(fw_neigh_sampled_indices)
            fw_indice_hidden = fw_indice_hidden * out_neigh_mask[:-1].unsqueeze(-1)
            fw_edge_hidden = fw_edge_hidden * out_neigh_mask[:-1].unsqueeze(-1)

            fw_hidden = torch.cat([fw_indice_hidden, fw_edge_hidden], dim=-1)
            fw_hidden = torch.sum(fw_hidden, dim=1)
            fw_hidden = torch.relu(fw_hidden)
            if self.direction == "fw":
                return embedded_node_reps, embedded_edge_reps, None, None, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, None
        if self.direction in ['bw', 'bi']:
            bw_neigh_sampled_edges = nn.Embedding.from_pretrained(in_edges, )(edges).type(torch.long)
            bw_neigh_sampled_indices = nn.Embedding.from_pretrained(in_indices, )(nodes).type(torch.long)

            bw_edge_hidden = nn.Embedding.from_pretrained(embedded_edge_reps, )(bw_neigh_sampled_edges)
            bw_indice_hidden = nn.Embedding.from_pretrained(embedded_node_reps, )(bw_neigh_sampled_indices)
            bw_indice_hidden = bw_indice_hidden * in_neigh_mask[:-1].unsqueeze(-1)
            bw_edge_hidden = bw_edge_hidden * in_neigh_mask[:-1].unsqueeze(-1)

            bw_hidden = torch.cat([bw_indice_hidden, bw_edge_hidden], dim=-1)
            bw_hidden = torch.sum(bw_hidden, dim=1)
            bw_hidden = torch.relu(bw_hidden)
            if self.direction == "bw":
                return embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, None, None, None, bw_hidden
        return embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, bw_hidden

    def forward(self, graph_tokens, graph_lengths):
        return self.__extract_features(graph_tokens, graph_lengths)

    def __extract_features(self, graph_tokens, graph_lengths):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        mask = self.__create_mask(graph_tokens)
        node_mask, in_neigh_mask, out_neigh_mask = mask.values()
        batch_size, num_nodes = nodes.shape
        embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, bw_hidden = self.__prepare_info(
            graph_tokens, graph_lengths)

        # learning node embedding
        for i in range(self.n_layers):
            if i == 0:
                dim_mul = 1
            else:
                dim_mul = 2 if self.concat else 1
            if self.direction in ['bw', 'bi']:
                if i == 0:
                    bw_cur_neigh_hidden = nn.Embedding.from_pretrained(embedded_node_reps, )(bw_neigh_sampled_indices)
                    bw_cur_edge_preps = nn.Embedding.from_pretrained(embedded_edge_reps, )(bw_neigh_sampled_edges)
                    bw_cur_neigh_hidden = torch.cat([bw_cur_neigh_hidden, bw_cur_edge_preps], dim=-1)
                    # bw_cur_neigh_hidden = bw_cur_edge_preps + bw_cur_indice_preps
                    # bw_cur_neigh_hidden = bw_cur_neigh_hidden * in_neigh_mask[:-1].unsqueeze(-1)
                else:
                    # bw_cur_edge_preps = nn.Embedding.from_pretrained(
                    #     torch.cat((bw_hidden, torch.zeros(1, dim_mul * self.hidden_dim, device=device)), dim=0))(
                    #     bw_neigh_sampled_edges)
                    bw_cur_neigh_hidden = nn.Embedding.from_pretrained(
                        torch.cat((bw_hidden, torch.zeros(1, dim_mul * self.hidden_dim, device=device)), dim=0))(
                        bw_neigh_sampled_indices)
                    # bw_cur_neigh_hidden = bw_cur_indice_preps + bw_cur_edge_preps
                    # bw_cur_neigh_hidden = bw_cur_neigh_hidden * in_neigh_mask[:-1].unsqueeze(-1)

                if i >= self.max_hops:  # maximun hops is 10
                    bw_aggregator = self.bw_aggregators[self.max_hops - 1]
                else:
                    bw_aggregator = self.bw_aggregators[i]
                bw_hidden = bw_aggregator(bw_hidden, bw_cur_neigh_hidden)

            # ======================================================================================================#
            if self.direction in ['fw', 'bi']:
                if i == 0:
                    fw_cur_neigh_hidden = nn.Embedding.from_pretrained(embedded_node_reps)(fw_neigh_sampled_indices)
                    fw_cur_edge_preps = nn.Embedding.from_pretrained(embedded_edge_reps)(fw_neigh_sampled_edges)
                    fw_cur_neigh_hidden = torch.cat([fw_cur_neigh_hidden, fw_cur_edge_preps], dim=-1)
                    # fw_cur_neigh_hidden = fw_cur_edge_preps + fw_cur_indice_preps
                    # fw_cur_neigh_hidden = fw_cur_neigh_hidden * out_neigh_mask[:-1].unsqueeze(-1)

                else:
                    # fw_cur_edge_preps = nn.Embedding.from_pretrained(
                    # torch.cat((fw_hidden, torch.zeros(1, dim_mul * self.hidden_dim, device=device)), dim=0))(
                    # fw_neigh_sampled_edges)
                    fw_cur_neigh_hidden = nn.Embedding.from_pretrained(
                        torch.cat((fw_hidden, torch.zeros(1, dim_mul * self.hidden_dim, device=device)), dim=0))(
                        fw_neigh_sampled_indices)
                    # fw_cur_neigh_hidden = fw_cur_edge_preps + fw_cur_indice_preps
                    # fw_cur_neigh_hidden = fw_cur_neigh_hidden * out_neigh_mask[:-1].unsqueeze(-1)

                if i >= self.max_hops:  # maximun hops is 10
                    fw_aggregator = self.fw_aggregators[self.max_hops - 1]
                else:
                    fw_aggregator = self.fw_aggregators[i]
                fw_hidden = fw_aggregator(fw_hidden, fw_cur_neigh_hidden)

        # [batch_size, num_nodes, 2 * hidden_dim]
        if self.direction == 'fw':
            graph_encoder_output = fw_hidden.reshape(batch_size, num_nodes, -1)
        elif self.direction == 'bw':
            graph_encoder_output = bw_hidden.reshape(batch_size, num_nodes, -1)
        else:
            fw_hidden = fw_hidden.reshape(batch_size, num_nodes, -1)
            bw_hidden = bw_hidden.reshape(batch_size, num_nodes, -1)
            # [batch_size, num_nodes, 4 * hidden_dim]
            graph_encoder_output = F.relu(torch.cat((fw_hidden, bw_hidden), dim=2))

        # Project node reps to output_dim first (node-wise)
        if self.fc_out:
            graph_encoder_output = self.fc_out(graph_encoder_output)  # [B, N, output_dim]

        # Attention pooling với root bias thay thế MaxPool
        # graph_hidden: [B, output_dim] -- sentence-level graph repr
        graph_hidden = self.attn_pool(graph_encoder_output, node_mask)

        return {
            "encoder_out":
                (graph_hidden, graph_encoder_output),
            "encoder_padding_mask": node_mask
        }


class ViGraphLayer(FairseqEncoder):
    def __init__(self, dictionary, embedding_dim, hidden_dim, dropout, pad_idx=1, alpha=0.1, n_loop=5,
                 first_layer=False, last_layer=False, n_heads=8):
        super(ViGraphLayer, self).__init__(dictionary)
        # self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        # self.dropout = dropout
        # self.last_layer = last_layer
        # if last_layer:
        #     self.W = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        #     nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #     self.a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        #     nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # else:
        #     self.W = nn.Parameter(torch.zeros(size=(embedding_dim, hidden_dim)))
        #     nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #     self.a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        #     nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #
        # self.W_fw = nn.Parameter(torch.zeros(size=(embedding_dim, hidden_dim)))
        # nn.init.xavier_uniform_(self.W_fw.data, gain=1.414)
        # # self.a_fw = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        # # nn.init.xavier_uniform_(self.a_fw.data, gain=1.414)
        #
        # self.W_bw = nn.Parameter(torch.zeros(size=(embedding_dim, hidden_dim)))
        # nn.init.xavier_uniform_(self.W_bw.data, gain=1.414)
        # # self.a_bw = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        # # nn.init.xavier_uniform_(self.a_bw.data, gain=1.414)
        # self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.leakyrelu = nn.LeakyReLU(alpha, inplace=False)
        # if last_layer:
        #     self.multihead_attn = multihead_attention.MultiheadAttention(embed_dim=embedding_dim // 8, num_heads=8,
        #                                                                  kdim=embedding_dim // 8, vdim=embedding_dim// 8)
        # else:
        #     self.multihead_attn = multihead_attention.MultiheadAttention(embed_dim=hidden_dim, num_heads=8,
        #                                                                  kdim=hidden_dim, vdim=hidden_dim)
        self.n_head = n_heads

        self.feat_dim = embedding_dim

        self.query_dim = embedding_dim
        self.key_dim = embedding_dim + embedding_dim

        self.merger = MergeLayer(self.query_dim, embedding_dim, embedding_dim, embedding_dim)

        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=n_heads,
                                                       dropout=dropout)

    def forward(self, embedded_node_reps, fw_hidden, fw_edge_hidden, bw_hidden, bw_edge_hidden, num_nodes, mask):
        # total_node, num_fw_neigh, dim = fw_hidden.shape
        # num_bw_neigh = bw_hidden.shape[1]
        # # total_node = fw_hidden.shape[0]
        # # for loop in range(0, self.n_loop):
        # h_i = torch.mm(embedded_node_reps, self.W)
        # h_fw = torch.mm(fw_hidden.reshape(-1, dim), self.W_fw)
        # # h_fw = torch.cat((embedded_node_reps,fw_hidden),dim=1)
        # h_bw = torch.mm(bw_hidden.reshape(-1, dim), self.W_bw)
        #
        # # if not self.last_layer:
        # #     h = torch.cat([h_i.unsqueeze(1), h_fw.reshape(total_node, -1, h_fw.shape[-1]),
        # #                    h_bw.reshape(total_node, -1, h_bw.shape[-1])], dim=1)
        # # else:
        # #     h = torch.cat([h_i.unsqueeze(1), h_fw.reshape(total_node, -1, h_fw.shape[-1]),
        # #                    h_bw.reshape(total_node, -1, h_bw.shape[-1])], dim=1)
        # # h = h.unsqueeze(dim=1)
        # # h = torch.sum(h, dim=1).reshape(num_nodes, -1, h.shape[-1]) # tgt_len, bsz, embed_dim
        # # h = torch.sum(h, dim=1).unsqueeze(dim=1).permute(1, 0, 2)  # tgt_len, bsz, embed_dim
        #
        # # h_fw = h_fw.reshape(total_node, -1, dim)
        # # h_bw = h_bw.reshape(total_node, -1, dim)
        #
        # h_fw = h_fw.reshape(total_node, num_fw_neigh, -1)
        # h_bw = h_bw.reshape(total_node, num_bw_neigh, -1)
        # h_neigh = torch.cat([h_fw, h_bw], dim=1)
        # attn_output, attn_weights = self.multihead_attn(h_i.unsqueeze(dim=0), h_neigh, h_neigh)
        # # attn_output, attn_weights = self.multihead_attn(h, h_fw, h_fw)
        #
        # # e = self.leakyrelu(torch.matmul(h, self.a).squeeze(2))
        #
        # # zero_vec = -9e15*torch.ones_like(e)
        # # alpha = torch.softmax(e, dim=1)
        # # alpha = torch.dropout(alpha, self.dropout, train=self.training)
        #
        # # h_: node-level representation
        # # h_ = torch.relu(alpha.unsqueeze(2) * h).reshape(-1, h.shape[-1])
        #
        # return attn_output, nn.functional.relu(h_fw), nn.functional.relu(h_bw)

        # return attn_output, nn.functional.relu(h_fw.reshape(total_node, num_fw_neigh, -1)), nn.functional.relu(
        #     h_bw.reshape(total_node, num_bw_neigh, -1))

        src_node_features_unrolled = torch.unsqueeze(embedded_node_reps, dim=1)

        query = src_node_features_unrolled.clone()
        key_fw = torch.cat([fw_hidden, fw_edge_hidden], dim=2)
        key_bw = torch.cat([bw_hidden, bw_edge_hidden], dim=2)
        key = torch.cat([key_fw, key_bw], dim=1)
        # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
        # Reshape tensors so to expected shape by multi head attention
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

        # Compute mask of which source nodes have no valid neighbors
        invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
        mask[invalid_neighborhood_mask.squeeze(), 0] = False

        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                                  )

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        attn_output = self.merger(attn_output, embedded_node_reps)
        return attn_output, attn_output_weights

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.feat_dim) + ' -> ' + str(self.feat_dim) + ')'


class ViGraphEncoder(FairseqEncoder):
    def __init__(self, dictionary, embedding_dim, hidden_dim, dropout, pad_idx=1, alpha=0.1, n_layers=1, n_highway=1,
                 n_heads=8):
        super(ViGraphEncoder, self).__init__(dictionary)
        self.direction = 'bi'
        self.pad_idx = pad_idx
        self.num_embeddings = len(dictionary)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.num_embeddings, embedding_dim, padding_idx=pad_idx)
        self.n_highways = n_highway
        if self.n_highways > 0:
            self.edge_highways = nn.ModuleList([Highway(embedding_dim, embedding_dim) for _ in range(self.n_highways)])
            self.indice_highways = nn.ModuleList(
                [Highway(embedding_dim, embedding_dim) for _ in range(self.n_highways)])
        else:
            self.edge_highways = None
            self.indice_highways = None
        self.dropout = dropout
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(
            ViGraphLayer(dictionary, embedding_dim, hidden_dim, dropout, pad_idx=1, alpha=alpha, n_loop=5,
                         n_heads=n_heads))
        for i in range(0, n_layers - 1):
            self.layers.append(
                ViGraphLayer(dictionary, embedding_dim, hidden_dim, dropout, pad_idx=1, alpha=alpha, n_loop=5,
                             n_heads=n_heads))
        # self.graph_out = ViGraphLayer(dictionary, 4 * n_layers * embedding_dim, hidden_dim, dropout, pad_idx=1, alpha=alpha,
        #                               n_loop=5,
        #                               last_layer=True)
        self.compact = nn.Linear(embedding_dim, hidden_dim)

        # Attention pooling với root bias thay thế MaxPool
        self.attn_pool = AttentionPooling(hidden_dim, root_bias=True)

    def __create_mask(self, graph_tokens):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        batch_size, num_nodes = nodes.shape
        node_mask = node_feats.eq(self.pad_idx)[:-1].reshape(batch_size, -1).t()
        if torch.cuda.is_available():
            padding_indx = int(torch.max(torch.max(in_indices, dim=0)[0]).cpu().numpy().tolist())
        else:
            padding_indx = int(torch.max(torch.max(in_indices, dim=0)[0]).numpy().tolist())
        in_indice_mask = in_indices != padding_indx
        out_indice_mask = out_indices != padding_indx
        return {
            "node_mask": node_mask,
            "in_neigh_mask": in_indice_mask,
            "out_neigh_mask": out_indice_mask,
        }

    def __prepare_info(self, graph_tokens, graph_lengths):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        mask = self.__create_mask(graph_tokens)
        node_mask, in_neigh_mask, out_neigh_mask = mask.values()
        batch_size, num_nodes = nodes.shape

        nodes = nodes.reshape(-1)  # [batch_size * num_nodes]
        edges = edges.reshape(-1)  # [ batch_size *num_node]

        # NB: total_nodes = batch_size * num_nodes
        # [total_nodes, embedding_dim]
        embedded_node_reps = torch.dropout(self.embeddings(node_feats), self.dropout, train=self.training)
        embedded_edge_reps = torch.dropout(self.embeddings(edge_feats), self.dropout, train=self.training)
        if self.n_highways > 0:
            for layer in range(self.n_highways):
                embedded_node_reps = self.indice_highways[layer](embedded_node_reps)
                embedded_edge_reps = self.edge_highways[layer](embedded_edge_reps)
        # [total_nodes, neigh_size]
        if self.direction in ['fw', 'bi']:
            fw_neigh_sampled_indices = nn.Embedding.from_pretrained(out_indices, )(nodes).type(torch.long)
            fw_neigh_sampled_edges = nn.Embedding.from_pretrained(out_edges, )(edges).type(torch.long)

            fw_edge_hidden = nn.Embedding.from_pretrained(embedded_edge_reps, )(fw_neigh_sampled_edges)
            fw_indice_hidden = nn.Embedding.from_pretrained(embedded_node_reps, )(fw_neigh_sampled_indices)
            fw_indice_hidden = fw_indice_hidden * out_neigh_mask[:-1].unsqueeze(-1)
            fw_edge_hidden = fw_edge_hidden * out_neigh_mask[:-1].unsqueeze(-1)

            # fw_hidden = torch.cat([fw_indice_hidden, fw_edge_hidden], dim=-1)
            fw_hidden = fw_indice_hidden + fw_edge_hidden
            # fw_hidden = torch.sum(fw_hidden, dim=1)
            # fw_hidden = torch.relu(fw_hidden)
            if self.direction == "fw":
                return embedded_node_reps, embedded_edge_reps, None, None, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, None
        if self.direction in ['bw', 'bi']:
            bw_neigh_sampled_edges = nn.Embedding.from_pretrained(in_edges, )(edges).type(torch.long)
            bw_neigh_sampled_indices = nn.Embedding.from_pretrained(in_indices, )(nodes).type(torch.long)

            bw_edge_hidden = nn.Embedding.from_pretrained(embedded_edge_reps, )(bw_neigh_sampled_edges)
            bw_indice_hidden = nn.Embedding.from_pretrained(embedded_node_reps, )(bw_neigh_sampled_indices)
            bw_indice_hidden = bw_indice_hidden * in_neigh_mask[:-1].unsqueeze(-1)
            bw_edge_hidden = bw_edge_hidden * in_neigh_mask[:-1].unsqueeze(-1)

            # bw_hidden = torch.cat([bw_indice_hidden, bw_edge_hidden], dim=-1)
            bw_hidden = bw_indice_hidden + bw_edge_hidden
            # bw_hidden = torch.sum(bw_hidden, dim=1)
            # bw_hidden = torch.relu(bw_hidden)
            if self.direction == "bw":
                return embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, None, None, None, bw_hidden
        return embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, bw_hidden

    def forward(self, graph_tokens, graph_lengths=None, **kwargs):
        ids, node_feats, edge_feats, nodes, edges, out_indices, out_edges, in_indices, in_edges = graph_tokens.values()
        mask = self.__create_mask(graph_tokens)
        node_mask, in_neigh_mask, out_neigh_mask = mask.values()
        feed_mask = torch.cat([out_neigh_mask[:-1], in_neigh_mask[:-1]], dim=1)
        feed_mask.requires_grad = False
        batch_size, num_nodes = nodes.shape
        # embedded_node_reps, embedded_edge_reps, bw_neigh_sampled_indices, bw_neigh_sampled_edges, fw_neigh_sampled_indices, fw_neigh_sampled_edges, fw_hidden, bw_hidden = self.__prepare_info(
        #     graph_tokens, graph_lengths)
        embedded_node_reps = torch.dropout(self.embeddings(node_feats), self.dropout, train=self.training)
        embedded_edge_reps = torch.dropout(self.embeddings(edge_feats), self.dropout, train=self.training)
        fw_neigh_sampled_indices = nn.Embedding.from_pretrained(out_indices, )(nodes).type(torch.long)
        fw_neigh_sampled_edges = nn.Embedding.from_pretrained(out_edges, )(edges).type(torch.long)
        bw_neigh_sampled_indices = nn.Embedding.from_pretrained(in_indices, )(nodes).type(torch.long)
        bw_neigh_sampled_edges = nn.Embedding.from_pretrained(in_edges, )(edges).type(torch.long)

        fw_node_reps = nn.Embedding.from_pretrained(embedded_node_reps)(fw_neigh_sampled_indices).reshape(
            batch_size * num_nodes, -1, self.embedding_dim)
        bw_node_reps = nn.Embedding.from_pretrained(embedded_node_reps)(bw_neigh_sampled_indices).reshape(
            batch_size * num_nodes, -1, self.embedding_dim)

        fw_edge_reps = nn.Embedding.from_pretrained(embedded_edge_reps)(fw_neigh_sampled_edges).reshape(
            batch_size * num_nodes, -1, self.embedding_dim)
        bw_edge_reps = nn.Embedding.from_pretrained(embedded_edge_reps)(bw_neigh_sampled_edges).reshape(
            batch_size * num_nodes, -1, self.embedding_dim)

        x_ = embedded_node_reps[:-1].clone()
        for layer in self.layers:
            x_, attn_weights = layer(x_, fw_node_reps, fw_edge_reps, bw_node_reps, bw_edge_reps, num_nodes, feed_mask)

        x = self.compact(x_)
        x = nn.functional.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)

        # Reshape về [B, N, hidden_dim] để pooling
        graph_encoder_output = x.reshape(batch_size, num_nodes, -1)  # [B, N, hidden_dim]

        # Attention pooling với root bias thay thế MaxPool
        # node_mask: [N, B] (True = padding) -- truyền thẳng vào AttentionPooling
        graph_hidden = self.attn_pool(graph_encoder_output, node_mask)  # [B, hidden_dim]

        return {
            "encoder_out":
                (graph_hidden, graph_encoder_output),
            "encoder_padding_mask": node_mask
        }


if __name__ == "__main__":
    pass
