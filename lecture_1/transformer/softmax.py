import torch
from torch import nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.amax(x, dim, keepdim=True)
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim, keepdim=True)

    return exp_x / sum_exp
