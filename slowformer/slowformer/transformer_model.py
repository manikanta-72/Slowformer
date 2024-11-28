
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from slowformer.encoder import EncoderBlock
from slowformer.decoder import DecoderBlock

from slowformer.functional_layers.positional_embeddings import sinusoidal_embedding

class Transformer(nn.Module):

    def __init__(self, d_model:int, max_positions: int, n_heads: int, num_encoder_blocks: int, num_decoder_blocks: int, vocab_size: int) -> None
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedder = sinusoidal_embedding.SinusoidalEmbedding(self.d_model, max_positions)

        self.encoder = nn.Sequential(*[EncoderBlock(d_model, max_positions, n_heads) for _ in range(num_encoder_blocks)])
        self.decoder = nn.Sequential(*[DecoderBlock(d_model, max_positions, n_heads) for _ in range(num_decoder_blocks)])

        self.lm_proj = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: T.Tensor, labels: T.Tensor, input_mask: T.Tensor | None = None, labels_mask: T.Tensor | None = None) -> T.Tensor:

        input_embds = self.embedder(inputs)
        label_embds = self.embedder(labels)

        # position embedding
        input_pos_embds = self.pos_embedder(input_embds)
        label_pos_embds = self.pos_embedder(label_embds)

        encoder_outputs = self.encoder(input_pos_embds, input_mask)

        decoder_outputs = self.decoder(label_pos_embds, encoder_outputs, labels_mask)

        lm_outputs = self.lm_proj(decoder_outputs)
        
        return F.softmax(lm_outputs, dim=-1)
