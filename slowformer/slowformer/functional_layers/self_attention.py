"""Implementation of Self Attention Mechanism"""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Performs self-attention"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.attn_proj = nn.Linear(d_model, 3 * d_model, bias=False)

    def forward(self, inputs: T.Tensor, is_causal: bool = False) -> T.Tensor:
        """Returns self-attended inputs"""
        Q, K, V = T.split(self.attn_proj(inputs), self.d_model)  # (N, d_model)

        Q_K = Q.T @ K if not is_causal else apply_causal_mask(Q.T @ K)  # (N, N)

        v_activations = F.softmax(Q_K / math.sqrt(self.d_model), dim=-1)  # (N, N)

        return v_activations @ V  # (N, d_model)


def apply_causal_mask(inputs: T.Tensor) -> T.Tensor:
    """Replaces all elements of upper triangle to '-inf' values"""
    n = inputs.size(0)
    tril_mask = T.ones(n, n, dtype=T.bool).tril()
    return inputs.masked_fill(tril_mask.logical_not(), float("-inf"))
