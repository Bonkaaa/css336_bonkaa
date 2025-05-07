import torch
from torch import nn
import math

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, device = None, dtype = None):
        super.__init__()

        d_ff_raw = d_model * 8/3

        d_ff = int(math.ceil(d_ff_raw / 64) * 64)

        self.w1 = nn.Parameter(torch.empty(d_model, 2 * d_ff, device=device, dtype=dtype))
        self.b1 = nn.Parameter(torch.zero(2 * d_ff, device=device, dtype=dtype))

        self.w2 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.w1)
        nn.init.trunc_normal_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.einsum("bsd, dk -> bsk", x, self.w1) + self.b1

        value, gate = x_proj.chunk(2, dim=1)

        x_act = value * torch.nn.functional.relu(gate)

        out = torch.einsum("bsd, dk -> bsk", x_act, self.w2) + self.b2
        return out


