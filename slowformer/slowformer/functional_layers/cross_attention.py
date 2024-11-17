"""Implements cross-attention"""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross Attention"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q: T.Tensor, inputs: T.Tensor) -> T.Tensor:
        """Returns cross-attended inputs with Q"""
        Q, K, V = self.q_proj(q), self.k_proj(inputs), self.v_proj(inputs)

        Q_K = Q.T @ K

        v_activations = F.softmax(Q_K / math.sqrt(self.d_model), dim=-1)

        return v_activations @ V
