"""Implements MultiHeaded Attention mechanism"""

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.linear_transform = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor | None = None
    ) -> T.Tensor:
        """Computes scaled dot-product attention"""

        Q_K = Q @ K.T if not mask else apply_mask(Q @ K.T, mask)

        v_activations = F.softmax(Q_K / math.sqrt(self.d_model), dim=-1)  # (N, N)

        return v_activations @ V  # (N, d_model)

    def forward(
        self,
        query: T.Tensor,
        key: T.Tensor,
        value: T.Tensor,
        mask: T.Tensor | None,
        is_causal: bool = False,
    ) -> T.Tensor:
        batch_size, seq_length, _ = query.size()

        Q, K, V = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        mask = add_causal_mask(mask, seq_length) if is_causal else mask

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.linear_transform(attn_output)


def apply_mask(inputs: T.Tensor, mask: T.Tensor) -> T.Tensor:
    """Replaces all elements of mask == 0 to '-inf' values"""
    return inputs.masked_fill(mask.logical_not(), float("-inf"))


def add_causal_mask(mask: T.Tensor | None, seq_length: int) -> T.Tensor:
    """Returns mask concatenated with causal mask"""
    causal_mask = T.tril(seq_length, seq_length, dtype=T.bool)
    mask = mask & causal_mask if mask else causal_mask
    return mask
