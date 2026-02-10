import torch
import torch.nn as nn
from torch.nn import functional as F

from fairseq.models import FairseqEncoder
from fairseq.models.aggregators import MaxPoolingAggregator, MeanAggregator, GCNAggregator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        graph_hidden = torch.max(graph_encoder_output, dim=1)[0]  # [batch_size, 4*hidden_dim]

        if self.fc_out:
            graph_hidden = self.fc_out(graph_hidden)
            graph_encoder_output = self.fc_out(graph_encoder_output)
        # graph_hidden: [batch_size, num_nodes,hidden_dim]
        # graph_encoder_output: [batch_size, hidden_dim]
        return {
            "encoder_out":
                (graph_hidden, graph_encoder_output),
            "encoder_padding_mask": node_mask
        }


if __name__ == "__main__":
    pass
