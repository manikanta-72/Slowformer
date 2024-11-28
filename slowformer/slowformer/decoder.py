
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from slowformer.functional_layers import multihead_attention, feed_forward_network

class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model

        self.multi_head_self_attn = multihead_attention.MultiHeadSelfAttention(d_model, num_heads)
        self.multi_head_cross_attn = multihead_attention.MultiHeadCrossAttention(d_model, num_heads)
        self.feed_forward = feed_forward_network.FeedForwardNN(d_model, hidden_dim=2048)

    def forward(self, inputs: T.Tensor, encoder_state: T.Tensor, mask: T.Tensor | None) -> T.Tensor:
        inputs_self = self.multi_head_self_attn(inputs, mask, is_causal=True)
        inputs_self_res = F.layer_norm(inputs_self + inputs, normalized_shape=(self.d_model,))

        inputs_self_res_cross = self.multi_head_cross_attn(encoder_state, encoder_state, inputs_self_res, mask)
        inputs_self_res_cross_res = F.layer_norm(inputs_self_res_cross + inputs_self_res, normalized_shape=(self.d_model,))

        inputs_self_res_cross_res_ffn = self.feed_forward(inputs_self_res_cross_res)
        inputs_self_res_cross_res_ffn_res = F.layer_norm(inputs_self_res_cross_res_ffn, normalized_shape=(self.d_model,))

        return inputs_self_res_cross_res_ffn_res
