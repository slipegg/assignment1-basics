from typing import Any, Dict
import torch
import torch.nn as nn
import math

from collections.abc import Callable, Iterable
from typing import Optional

from einops import einsum, rearrange
from jaxtyping import Float, Int

def cross_entropy(logits: Float[torch.Tensor, "batch seq_len vocab_size"], targets: Int[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, ""]:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    prob_sum = shifted_logits.exp().sum(dim=-1, keepdim=True)
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -(target_logits - prob_sum.log()).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-2, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                t, m, v = state['t'], state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t) 
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps) 
                p.data = p.data * (1 - lr * weight_decay)

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss
        
def learning_rate_schedule(step: int, warmup_steps: int = 1000, max_lr: float = 1e-2, min_lr: float = 1e-4, cosine_steps: int = 10000) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step <= cosine_steps:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - warmup_steps) / (cosine_steps - warmup_steps)))
    else:
        return min_lr
        
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    grads = [p.grad.flatten() for p in parameters if p.grad is not None]
    all_grads = torch.cat(grads, dim=0)
    grad_norm = torch.norm(all_grads)
    if grad_norm > max_norm:
        clip_coef = max_norm / (grad_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
