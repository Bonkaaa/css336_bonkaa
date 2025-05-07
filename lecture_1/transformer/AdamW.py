import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                state = self.state[param]

                # Step count
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data) # m
                    state['exp_avg_sq'] = torch.zeros_like(param.data) # v

                state['step'] += 1
                step = state['step']
                m, v = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                weight_decay = group['weight_decay']

                # Update first moment estimate
                m.mul_(beta1).add_(1 - beta1, grad)

                # Update second moment estimate
                v.mul_(beta2).addcmul(1 - beta2, grad, grad)

                # Compute bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update
                denom = v.sqrt().add_(eps)
                param.data.addcdiv_(m, denom, value=-step_size)

                # Apply weight decay
                if weight_decay != 0:
                    param.data.add_(param.data, alpha = -lr * weight_decay)

        return loss

