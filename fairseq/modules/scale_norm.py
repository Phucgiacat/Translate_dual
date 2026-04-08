import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    """ScaleNorm inspired from https://arxiv.org/pdf/1910.05895.pdf"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
