import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()

        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))

        torch.nn.init.trunc_normal_(self.W.data, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
