import torch
from torch import nn

def clip_gradients(params, max_norm, eps = 1e-6):
    total_norm = 0
    for p in params:
        if p.grad is not None:
            grad = p.grad.data
            total_norm += torch.sum(grad ** 2).item()
    total_norm = torch.sqrt(total_norm + eps)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)
