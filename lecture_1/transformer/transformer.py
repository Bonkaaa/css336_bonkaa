import torch
from torch import nn, optim
import math

# Multi head self attention
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

# RMS norm
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

# Feed forward
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

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super.__init__()
        self.norm1 = RMSNorm(d_model, eps = 1e-5)
        self.norm2 = RMSNorm(d_model, eps = 1e-5)

        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope = self)
        self.ff = SwiGLUFeedForward(d_model, device = None, dtype = None)

    def forward(self, x: torch.Tensor, token_positions: torch.tensor) -> torch.Tensor:
        # Sublayer 1: MHA with residual
        x = x + self.mha(self.norm1(x), token_positions)

        # Sublayer 2: FF with residual
        x = x + self.ff(self.norm2(x))

        return x

# Embedding
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()

        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))

        torch.nn.init.trunc_normal_(self.W.data, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: nn.Module
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.context_length = context_length
        self.num_layers = num_layers

        self.blocks = nn.ModuleList([
            Transformer_block(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        )

        self.final_norm = RMSNorm(d_model, eps = 1e-5)
        self.lm.head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        :param token_ids: (batch, seq_len)
        :return: (batch, seq_len, vocab_size)
        """

        batch_size, seq_len, _ = token_ids.shape

        # Embedding
        x = self.embedding(token_ids)

        # Positional information will be added via RoPE inside attention blocks
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # (1, seq_len)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, token_positions)

        # Norm + add to vocab
        x = self.final_norm(x)
        logits = self.lm.head(x) # (batch, seq_len, vocab_size)

        return logits










