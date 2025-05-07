import torch
from torch import nn
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.amax(x, dim, keepdim=True)
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim, keepdim=True)

    return exp_x / sum_exp

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super.__init__()

        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device = device, dtype = dtype))
        torch.nn.init.trunc_normal_(self.W.data, a = -3.0, b = 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bi, oi -> bi', x, self.W)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    :param q: (..., seq_len_q, d_k)
    :param k: (..., seq_len_k, d_k)
    :param v: (..., seq_len_v, d_v)
    :param mask: optional (..., seq_len_q, seq_len_k) of bool
    :return: output (..., seq_len_q, d_v)
    """

    d_k = q.size(-1)

    scores = torch.einsum('...ik, ...jk -> ...ij', q, k) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_weights = softmax(scores, dim=-1)

    output = torch.einsum('...ik, ...jk -> ...ij', attn_weights, v)

    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module):
        super.__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (batch_size, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Apply RoPE
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Create casual mask
        casual_mask = torch.triu(torch.ones(seq_len, seq_len))

        # Compute scaled dot-product attention
        attn_output = scaled_dot_product_attention(q, k, v, mask = casual_mask)

        # Merge_heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_k)
        attn_output = torch.einsum('bnld-> bld', attn_output).reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)


