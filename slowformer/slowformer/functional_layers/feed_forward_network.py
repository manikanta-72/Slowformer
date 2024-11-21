"""Implement Point-Wise Feed Forward Networks"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(T.Module):
    """Applies linear transformations to the inputs"""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.linear_transform_1 = nn.Linear(d_model, hidden_dim)
        self.linear_transform_2 = nn.Linear(hidden_dim, d_model)
        self.activation = F.relu

    def forward(self, inputs: T.Tensor) -> T.Tensor:
        """Returns inputs after applying linear transformations"""
        return self.linear_transform_2(self.activation(self.linear_transform_1(inputs)))
