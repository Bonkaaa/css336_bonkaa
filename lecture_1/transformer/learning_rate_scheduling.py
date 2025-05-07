import torch
from torch import nn
import math

def cosine_learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha_t = t/T_w * alpha_max
    elif T_w <= t <= T_c:
        alpha_t = alpha_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w)) * math.pi) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t