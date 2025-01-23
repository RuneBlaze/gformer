import math
from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from constants import (
    MAX_SEQUENCE_LENGTH,
    MAX_TAXA,
    VOCAB_SIZE,
)


class DistanceMatrixMLP(nn.Module):
    """
    Single hidden layer MLP with SiLU activation for processing distance matrix encodings.
    Input dimension is calculated as: 8 * (n*(n-1)/2) where n is number of taxa
    and 8 is the binary distance encoding dimension.
    """

    def __init__(self, num_taxa: int, hidden_dim: int, output_dim: int):
        """
        Args:
            num_taxa: Number of taxa in the tree
            hidden_dim: Size of the hidden layer
            output_dim: Number of output features
        """
        super().__init__()

        # Calculate input dimension based on number of taxa
        num_distances = (num_taxa * (num_taxa - 1)) // 2
        input_dim = 8 * num_distances  # 8 is binary distance encoding dimension

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # First layer with SiLU activation
        x = F.silu(self.fc1(x))

        # Output layer (no activation)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for non-recurrent transformers.
    Only used for output sequence positions.
    """

    def __init__(self, d_model: int, max_len: int = MAX_SEQUENCE_LENGTH):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]
            start_pos: Starting position for the sequence
        """
        return x + self.pe[start_pos : start_pos + x.size(1)]


@dataclass
class ModelConfig:
    embedding_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    tree_embedding_dim: int
    max_sequence_length: int

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(**config["model"])  # Only use the "model" section


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex rotation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor.
    """
    # Reshape using einops for better readability
    x_complex = einops.rearrange(x.float(), '... (d r) -> ... d r', r=2)
    x_complex = torch.view_as_complex(x_complex)
    
    # Extend freqs_cis for broadcasting
    freqs_cis = freqs_cis[:x.shape[-2]]  # Select up to sequence length
    
    # Apply rotation using complex multiply
    x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
    
    # Convert back to real and reshape
    x_out = torch.view_as_real(x_rotated)
    return einops.rearrange(x_out, '... d r -> ... (d r)')

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_length: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_length)
        self.register_buffer("rope_freqs", freqs_cis, persistent=False)
        
        self.scaling = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tgt_len = query.shape[:2]
        src_len = key.shape[1]
        
        # Linear projections and reshape
        q = einops.rearrange(
            self.q_proj(query),
            'b s (h d) -> b h s d',
            h=self.num_heads
        )
        k = einops.rearrange(
            self.k_proj(key),
            'b s (h d) -> b h s d',
            h=self.num_heads
        )
        v = einops.rearrange(
            self.v_proj(value),
            'b s (h d) -> b h s d',
            h=self.num_heads
        )
        
        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, self.rope_freqs)
        k = apply_rotary_emb(k, self.rope_freqs)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = einops.rearrange(output, 'b h s d -> b s (h d)')
        output = self.out_proj(output)
        
        return output

class RoPETransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        max_seq_length: int,
    ):
        super().__init__()
        
        # Self-attention with RoPE
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, max_seq_length)
        # Regular cross-attention (no RoPE needed)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block with RoPE
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt2, tgt2, tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # Cross-attention block
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # Feed-forward block
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(F.relu(self.linear1(tgt2)))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

class RotaryDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        max_seq_length: int,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RoPETransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                max_seq_length=max_seq_length,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        
        return output


class TreeTransformer(nn.Module):
    """
    Encoder-decoder transformer model for processing gene trees and generating species trees.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Token embedding layer
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim)

        # Tree embedding MLP
        self.tree_embedding = DistanceMatrixMLP(
            num_taxa=MAX_TAXA,  # Use MAX_TAXA from constants
            hidden_dim=config.embedding_dim * 2,
            output_dim=config.embedding_dim,
        )

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Replace standard decoder with RoPE decoder
        self.decoder = RotaryDecoder(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            num_layers=config.num_layers,
            max_seq_length=config.max_sequence_length,
        )

        # Output projection
        self.output_projection = nn.Linear(config.embedding_dim, VOCAB_SIZE)

        # Enable support for gradient checkpointing
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        ...
        # self.gradient_checkpointing = True
        # self.encoder.enable_gradient_checkpointing()
        # self.decoder.enable_gradient_checkpointing()

    def forward(
        self,
        tree_encodings: torch.Tensor,  # [batch_size, num_gene_trees, num_distances, 8]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_gene_trees, num_distances, bits = tree_encodings.shape

        # Reshape tree encodings to process each gene tree through MLP
        tree_encodings = einops.rearrange(
            tree_encodings, "b g d bits -> (b g) (d bits)", bits=8
        )
        encoded_trees = self.tree_embedding(tree_encodings)

        # Reshape back for encoder input
        encoder_input = einops.rearrange(
            encoded_trees, "(b g) e -> b g e", b=batch_size, g=num_gene_trees
        )

        # Create padding mask based on zero vectors
        # True indicates positions that should be masked (padded)
        tree_padding_mask = (tree_encodings.abs().sum(dim=-1) == 0).view(
            batch_size, num_gene_trees
        )

        # Run Transformer encoder
        memory = self.encoder(encoder_input, src_key_padding_mask=tree_padding_mask)

        if output_tokens is not None:
            # Training mode
            decoder_input = self.token_embedding(output_tokens)
            
            # Generate causal mask if not provided
            if attention_mask is None:
                seq_length = output_tokens.size(1)
                attention_mask = torch.triu(
                    torch.ones(seq_length, seq_length), diagonal=1
                ).bool()
                attention_mask = attention_mask.to(output_tokens.device)

            # Use gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.decoder, decoder_input, memory, attention_mask
                )
            else:
                hidden_states = self.decoder(decoder_input, memory, tgt_mask=attention_mask)
        else:
            # Inference mode - start with memory
            hidden_states = memory

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return logits
