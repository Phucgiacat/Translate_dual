import torch
from torch import nn
from torch.nn import functional as F

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
