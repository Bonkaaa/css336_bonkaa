import torch
from torch import nn
from softmax import *
import math

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