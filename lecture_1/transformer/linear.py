import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super.__init__()

        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device = device, dtype = dtype))
        torch.nn.init.trunc_normal_(self.W.data, a = -3.0, b = 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bi, oi -> bi', x, self.W)