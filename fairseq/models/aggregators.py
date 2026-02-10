import torch
from torch import nn
from torch.nn import functional as F
from fairseq.modules import MultiheadAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, input_neigh_dim=None, activation=torch.relu, dropout=0.1, bias=False,
                 concat=False,
                 device=None):

        super(MeanAggregator, self).__init__()
        self.dropout_rate = dropout
        self.bias = bias
        self.activation_fn = activation
        self.concat = concat
        if input_neigh_dim is None:
            input_neigh_dim = input_dim
        if torch.cuda.is_available():
            self.neigh_linear = nn.Linear(input_neigh_dim, output_dim, bias=bias).cuda()
            self.self_linear = nn.Linear(input_dim, output_dim, bias=bias).cuda()
        else:
            self.neigh_linear = nn.Linear(input_neigh_dim, output_dim, bias=bias)
            self.self_linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, prev_hidden, neigh_hidden):
        prev_hidden = F.dropout(prev_hidden, p=self.dropout_rate)
        neigh_hidden = F.dropout(neigh_hidden, p=self.dropout_rate)
        neigh_means = torch.mean(neigh_hidden, dim=1)
        from_neighs = self.neigh_linear(neigh_means)
        from_self = self.self_linear(prev_hidden)
        if not self.concat:
            output = from_self + from_neighs
        else:
            output = torch.cat([from_self, from_neighs], dim=1)

        return self.activation_fn(output)


class GCNAggregator(nn.Module):
    """
    Aggregating via mean and followed by matmul + non-linearity
    """

    def __init__(self, input_dim, output_dim, input_neigh_dim=None, dropout=0.2, bias=False, activation=torch.relu,
                 concat=False):
        super(GCNAggregator, self).__init__()
        self.dropout_rate = dropout
        self.activation_fn = activation
        self.bias = bias
        self.concat = concat
        if input_neigh_dim is None:
            input_neigh_dim = input_dim
        if torch.cuda.is_available():
            self.linear = nn.Linear(input_neigh_dim, output_dim, bias=bias).cuda()
        else:
            self.linear = nn.Linear(input_neigh_dim, output_dim, bias=bias)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, prev_hidden, neigh_hidden):
        neigh_hidden = F.dropout(neigh_hidden, p=self.dropout_rate)
        prev_hidden = F.dropout(prev_hidden, p=self.dropout_rate)
        synthesized_hidden = torch.cat([neigh_hidden, prev_hidden.unsqueeze(1)], dim=1)
        means = torch.mean(synthesized_hidden, dim=1)
        output = self.linear(means)
        return self.activation_fn(output)


class MaxPoolingAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, input_neigh_dim=None, hidden_dim=512, dropout=0.0, bias=False,
                 activation=torch.relu, concat=False):
        super(MaxPoolingAggregator, self).__init__()
        self.dropout_rate = dropout
        self.bias = bias
        self.activation_fn = activation
        self.concat = concat

        if input_neigh_dim is None:
            input_neigh_dim = input_dim

        if torch.cuda.is_available():
            self.mlp_layer = nn.Linear(input_neigh_dim, hidden_dim, bias=True).cuda(0)
            self.neigh_linear = nn.Linear(hidden_dim, output_dim, bias=bias).cuda()
            self.self_linear = nn.Linear(input_dim, output_dim, bias=bias).cuda()
        else:
            self.mlp_layer = nn.Linear(input_neigh_dim, hidden_dim, bias=True)
            self.neigh_linear = nn.Linear(hidden_dim, output_dim, bias=bias)
            self.self_linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_neigh_dim = input_neigh_dim

    def forward(self, prev_hidden, neigh_hidden):
        shape = neigh_hidden.shape
        tmp_neigh = neigh_hidden.reshape(-1, self.input_neigh_dim)
        neigh_hidden_reshaped = self.mlp_layer(tmp_neigh)
        tmp_neigh = neigh_hidden_reshaped.reshape(shape[0], shape[1], -1)
        tmp_neigh = torch.max(tmp_neigh, dim=1)[0]
        from_neigh = self.neigh_linear(tmp_neigh)
        from_self = self.self_linear(prev_hidden)
        if not self.concat:
            output = from_self + from_neigh
        else:
            output = torch.cat([from_self, from_neigh], dim=1)
        return self.activation_fn(output)


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 input_neigh_dim=None, dropout=0.2,
                 bias=False,
                 activation=None,
                 concat=False, alpha=0.2):
        super(AttentionAggregator, self).__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.a = nn.Linear(2 * output_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.414)

    def forward(self, prev_hidden, neigh_hidden, mask=None):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.linear1(prev_hidden)
        # compute attention
        attn_scores = (neigh_hidden * x.unsqueeze(1)).sum(dim=2)

        # don't attend over padding
        if mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=1)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * neigh_hidden).sum(dim=1)

        x = torch.tanh(self.a(torch.cat((x, prev_hidden), dim=1)))
        return x

class MultiheadAttn(nn.Module):
    def __init__(self, input_dim, output_dim,
                 input_neigh_dim=None, dropout=0.2,
                 bias=False,
                 activation=None,
                 concat=False, alpha=0.2, heads=8):
        super(MultiheadAttn, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_list = nn.ModuleList()
        for i in range(heads):
            self.attn_list.append(AttentionAggregator(input_dim, output_dim,dropout=dropout))
    def forward(self, prev_hidden, neigh_vec_hidden, aggtype='mean'):
        prev_hidden = F.relu(self.fc1(prev_hidden))
        attn = [layer(prev_hidden, neigh_vec_hidden) for layer in self.attn_list]
        if aggtype == 'cat':
            return F.elu(torch.cat(attn,dim=1), alpha=0.2)
        else:
            meant = F.elu(torch.mean(torch.stack(attn), dim=0), alpha=0.2)
            pre_result = torch.cat([prev_hidden, meant], dim=1)
            return F.relu(pre_result)
        
class NewMulihead(nn.Module):
    def __init__(self, input_dim, output_dim,
                 input_neigh_dim=None, dropout=0.2,
                 bias=False,
                 activation=None,
                 concat=False, alpha=0.2, heads=8):
        super(NewMulihead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.attn = MultiheadAttention(input_dim,heads, dropout=dropout, encoder_decoder_attention=False)
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, prev_hidden, neigh_hidden, mask=None):
        prev_hidden = self.linear(prev_hidden)
        neigh_hidden = neigh_hidden.permute(1,0,2)
        attn, _ = self.attn(query=neigh_hidden, key=neigh_hidden, value=neigh_hidden, key_padding_mask=mask)
        attn = attn.permute(1,0,2)
        attn = torch.mean(self.fc(attn), dim=1)

        return torch.relu(torch.cat([prev_hidden, attn], dim=1))
