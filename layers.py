import math
from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from constants import (
    EMBEDDING_DIM,
    EOS,
    INTERNAL_NODE,
    MAX_SEQUENCE_LENGTH,
    MAX_TAXA,
    MLP_HIDDEN_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    PAD,
    TREE_EMBEDDING_DIM,
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
        self.output_projection = nn.Linear(config.embedding_dim, VOCAB_SIZE)


    def forward(
        self,
        tree_encodings: torch.Tensor,  # [batch_size, num_gene_trees, num_distances, 8]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_gene_trees, num_distances, bits = tree_encodings.shape

        

        # Reshape tree encodings to process each gene tree through MLP
        tree_encodings = einops.rearrange(
            tree_encodings,
            'b g d bits -> (b g) (d bits)',
            bits=8
        )
        encoded_trees = self.tree_embedding(tree_encodings)
        
        # Reshape back for encoder input
        encoder_input = einops.rearrange(
            encoded_trees,
            '(b g) e -> b g e',
            b=batch_size,
            g=num_gene_trees
        )

        # Create padding mask based on zero vectors
        # True indicates positions that should be masked (padded)
        tree_padding_mask = (tree_encodings.abs().sum(dim=-1) == 0).view(batch_size, num_gene_trees)

        # Run Transformer encoder
        memory = self.encoder(
            encoder_input,
            src_key_padding_mask=tree_padding_mask
        )

        if output_tokens is not None:
            # Training mode
            token_embeddings = self.token_embedding(output_tokens)
            decoder_input = self.pos_encoding(token_embeddings)

            # Generate causal mask if not provided
            if attention_mask is None:
                seq_length = output_tokens.size(1)
                attention_mask = torch.triu(
                    torch.ones(seq_length, seq_length), diagonal=1
                ).bool()
                attention_mask = attention_mask.to(output_tokens.device)

            # Use gradient checkpointing if enabled
            hidden_states = self.decoder(
                    decoder_input,
                    memory,
                    attention_mask
                )
        else:
            # Inference mode - start with memory
            hidden_states = memory

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return logits