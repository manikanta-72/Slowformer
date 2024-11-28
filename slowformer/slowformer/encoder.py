
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from slowformer.functional_layers import multihead_attention, feed_forward_network

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout_ratio: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.multi_head_attn = multihead_attention.MultiHeadSelfAttention(d_model=self.d_model, num_heads=n_heads)
        self.feed_forward = feed_forward_network.FeedForwardNN(d_model=self.d_model, hidden_dim=2048)

    def forward(self, inputs: T.Tensor, mask: T.Tensor | None) -> T.Tensor:
        
        inputs_attn = self.multi_head_attn(inputs, mask)
        inputs_attn_res = self.dropout(F.layer_norm(inputs + inputs_attn, normalized_shape=(self.d_model,)))

        inputs_attn_res_ffn = self.feed_forward(inputs_attn_res)
        inputs_attn_res_ffn_res = self.dropout(F.layer_norm(inputs_attn_res + inputs_attn_res_ffn, normalized_shape=(self.d_model,)))

        return inputs_attn_res_ffn_res
