import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import yaml


from constants import (
    MAX_TAXA,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    MLP_HIDDEN_DIM,
    TREE_EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH,
    LEFT_PAREN,
    RIGHT_PAREN,
    END_OF_INPUT,
    END_OF_OUTPUT,
)


class DistanceMatrixMLP(nn.Module):
    """
    Single hidden layer MLP with SiLU activation for processing distance matrix encodings.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim: Number of input features (8 for binary distance encoding)
            hidden_dim: Size of the hidden layer
            output_dim: Number of output features
        """
        super().__init__()

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
    vocab_size: int

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(**config["model"])  # Only use the "model" section


class TreeTransformer(nn.Module):
    """
    Encoder-decoder transformer model for processing gene trees and generating species trees.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Tree embedding MLP
        self.tree_embedding = DistanceMatrixMLP(
            input_dim=8,  # Binary distance encoding dimension
            hidden_dim=config.embedding_dim * 2,
            output_dim=config.embedding_dim,  # Must match embedding_dim for transformer
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.embedding_dim, config.max_sequence_length)

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        # Output projection
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)

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
        tree_encodings: torch.Tensor,  # [batch_size, max_taxa, input_dim]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            tree_encodings: Tensor containing distance matrix encodings for each tree
            output_tokens: Target sequence tokens (for training)
            attention_mask: Causal attention mask for decoder self-attention

        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size = tree_encodings.size(0)
        max_taxa = tree_encodings.size(1)

        # Create padding mask for trees (1 for real data, 0 for padding)
        # We assume zero vectors are padding
        tree_padding_mask = (tree_encodings.sum(dim=-1) != 0).any(dim=-1)  # [batch_size, max_taxa]
        
        # Embed trees using MLP
        tree_embeddings = self.tree_embedding(
            tree_encodings.view(-1, tree_encodings.size(-1))
        )
        encoder_input = tree_embeddings.view(batch_size, max_taxa, -1)

        # Run encoder with padding mask
        memory = self.encoder(
            encoder_input,
            src_key_padding_mask=~tree_padding_mask  # PyTorch expects True for padding positions
        )

        if output_tokens is not None:
            # Training mode
            token_embeddings = self.token_embedding(output_tokens)
            
            # Add positional encoding to decoder input
            decoder_input = self.pos_encoding(token_embeddings)

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
                    self.decoder,
                    decoder_input,
                    memory,
                    attention_mask
                )
            else:
                hidden_states = self.decoder(
                    decoder_input,
                    memory,
                    attention_mask
                )

        else:
            # Inference mode - start with just embedding of first token
            hidden_states = memory

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return logits
