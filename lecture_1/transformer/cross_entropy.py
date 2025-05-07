import torch
from numpy.ma.core import soften_mask
from torch import nn

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy between logits and targets.
    """
    max_logits = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - max_logits
    exp_logits = torch.exp(logits_stable)
    softmax_denominator = exp_logits.sum(dim=-1, keepdim=True)

    softmax_probs = exp_logits / softmax_denominator

    # Gather probabilities of the correct tokens
    correct_token_probs = softmax_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Cross entropy loss
    loss = -torch.log(correct_token_probs)

    return loss.mean()