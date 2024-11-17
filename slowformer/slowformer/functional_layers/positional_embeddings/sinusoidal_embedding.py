"""Implements positional embedding layer for the transformer"""

import math

import torch as T
import torch.nn as nn
from loguru import logger

FREQUENCY = 10_000


class SinusoidalEmbedding(nn.Module):
    """Enriches Positional information (constant sinusoidal vectors)"""

    def __init__(self, d_model: int, max_positions: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        pos_info = compute_sinusoidal_vectors(max_positions, d_model)
        self.register_buffer("pos_info", pos_info)

    def forward(self, inputs: T.Tensor) -> T.Tensor:
        """Returns positional information enriched inputs"""
        return inputs + self.pos_info[:, : inputs.size(1)]


def compute_sinusoidal_vectors(max_positions: int, vector_len: int) -> T.Tensor:
    """Computes constant vector using sinusoidal wave functions"""
    pos_vectors = T.zeros(max_positions, vector_len)
    positions = T.arange(0, max_positions, dtype=T.float).unsqueeze(1)
    mul_coeff = T.exp(
        T.arange(0, vector_len, 2) * (-1 * math.log(FREQUENCY) / vector_len)
    )
    pos_vectors[:, 0::2] = T.sin(positions * mul_coeff)
    pos_vectors[:, 1::2] = T.cos(positions * mul_coeff)

    logger.info(f"created {max_positions} positional embeddings with sin/cos function")
    return pos_vectors
