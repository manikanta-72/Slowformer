"""Implements MultiHeaded Attention mechanism"""

from typing import Optional

import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout_ratio: float, d_keys: Optional[int]=None, d_values: Optional[int]=None):
        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys or d_model
        self.d_values = d_values or d_model

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(self.d_model, self.d_keys, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_keys, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_values, bias=False)

        self.linear_transform = nn.Linear(self.d_values, d_model)

        # Additional implemtation for scaled additive attention
        # self.alignment_model = nn.Linear()

    def scaled_dot_product_attention(
        self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor | None = None
    ) -> T.Tensor:
        """
        Computes scaled dot-product attention
        Batched & split across multiple heads
        """

        # handle mask dim for multi-headed attention
        mask = mask.unsqueeze(1) if mask else mask

        Q_K = Q @ K.transpose(-2,-1) if not mask else apply_mask(Q @ K.transpose(-2,-1), mask)

        v_activations = self.dropout(F.softmax(Q_K / math.sqrt(self.d_keys), dim=-1))  # (B, N_H, N, N)

        return v_activations @ V  # (B, N_H, N, H_Dim)
    
    def scaled_additive_attention(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor | None = None) -> T.Tensor:
        """
        Computes scaled additive attention
        In paper it was mentioned that dot-product attention outperforms additive attention
        additive attention paper: https://arxiv.org/pdf/1409.0473
        """
        return T.tensor([])

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

        attn_output = self.dropout(self.scaled_dot_product_attention(Q, K, V, mask))
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        ) # flatten and resize 

        return self.linear_transform(attn_output)


class MultiHeadCrossAttention(nn.Module):
    """
    Derived MultiHeadAttention to compute cross-attention
    """
    def __init__(self, d_model, num_heads, d_keys = None, d_values = None):
        self.attn = MultiHeadAttention(d_model, num_heads, d_keys, d_values)

    def forward(self, query: T.Tensor, key: T.Tensor, value: T.Tensor, mask: T.Tensor) -> T.Tensor:
        return self.attn(query, key, value, mask)


class MultiHeadSelfAttention(nn.Module):
    """
    Derived MultiHeadAttention to compute self-attention
    """
    def __init__(self, d_model, num_heads, d_keys = None, d_values = None):
        self.attn = MultiHeadAttention(d_model, num_heads, d_keys, d_values)

    def forward(self, inputs: T.Tensor, mask:T.Tensor, is_causal:bool) -> T.Tensor:
        return self.attn(inputs, inputs, inputs, mask, is_causal)


def apply_mask(inputs: T.Tensor, mask: T.Tensor) -> T.Tensor:
    """Replaces all elements of mask == 0 to '-inf' values"""
    return inputs.masked_fill(mask.logical_not(), float("-inf"))


def add_causal_mask(mask: T.BoolTensor | None, seq_length: int) -> T.BoolTensor:
    """Returns mask concatenated with causal mask"""
    causal_mask = T.ones(seq_length, seq_length, dtype=T.bool).tril()
    mask = mask & causal_mask if mask else causal_mask
    return mask
