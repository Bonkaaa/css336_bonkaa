import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super.__init__()
        self.eps = eps

        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)

        normed = x / rms

        result = torch.einsum('bsd, d -> bsd', normed, self.weight)
        return result.to(dtype=in_dtype)