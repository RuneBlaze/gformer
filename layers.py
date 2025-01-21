import math
from dataclasses import dataclass

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


class GeneTreeEncoder(nn.Module):
    """
    Light-weight encoder-only model to project individual gene trees to a fixed dimension.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, MAX_SEQUENCE_LENGTH)

        # Encoder layers for within-sequence attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection after pooling
        self.pooling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, gene_tree_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_tree_tokens: Tensor of shape [batch_size, max_tree_tokens]
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Create padding mask (True indicates positions that should be masked)
        padding_mask = gene_tree_tokens == PAD  # [batch_size, max_tree_tokens]
        # Embed and add positional encoding
        embeddings = self.token_embedding(gene_tree_tokens)  # [batch_size, max_tree_tokens, embedding_dim]
        embeddings = self.pos_encoding(embeddings)
        # Let tokens attend within each sequence
        encoded = self.encoder(
            embeddings, 
            src_key_padding_mask=padding_mask
        )  # [batch_size, max_tree_tokens, embedding_dim]
        # Mean pool over sequence length (excluding padding)
        mask = ~padding_mask.unsqueeze(-1)  # [batch_size, max_tree_tokens, 1]
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch_size, 1, 1]
        pooled = (encoded * mask).sum(dim=1) / valid_counts.squeeze(-1)  # [batch_size, embedding_dim]
        # Set fully padded sequences to 0
        zero_idx = (mask.sum(dim=[1, 2]) == 0)  # [batch_size]
        pooled[zero_idx] = 0
        # Final projection
        return self.pooling(pooled)  # [batch_size, embedding_dim]


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
    Both gene trees and species trees are tokenized using the same vocabulary.
    First encodes each gene tree to a fixed-dimension vector using GeneTreeEncoder,
    then processes these encodings to generate the species tree.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Remove gene_tree_projection as it's redundant
        self.gene_tree_encoder = GeneTreeEncoder(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=2,
        )

        # Token embedding layer (for species tree only)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim)

        # Positional encoding (for species tree only)
        self.pos_encoding = PositionalEncoding(
            config.embedding_dim, config.max_sequence_length
        )

        # Encoder layers (processes encoded gene trees)
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

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
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
        gene_tree_tokens: torch.Tensor,  # [batch_size, num_gene_trees, max_tree_tokens]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_gene_trees, max_tree_tokens = gene_tree_tokens.shape

        # Reshape to process all trees at once
        flat_gene_trees = gene_tree_tokens.view(-1, max_tree_tokens)
        
        # Encode each gene tree to a fixed-dimension vector
        gene_tree_encodings = self.gene_tree_encoder(flat_gene_trees)  # [batch_size * num_gene_trees, embedding_dim]
        
        # Reshape back to [batch_size, num_gene_trees, embedding_dim]
        gene_tree_encodings = gene_tree_encodings.view(batch_size, num_gene_trees, -1)

        # Create padding mask for encoded gene trees
        # If ALL positions in the original gene tree were padding, mask that tree's encoding
        trees_padding_mask = gene_tree_tokens.all(dim=-1)

        # Run main encoder on the gene tree encodings with padding mask
        memory = self.encoder(
            gene_tree_encodings, src_key_padding_mask=trees_padding_mask
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
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.decoder, decoder_input, memory, attention_mask
                )
            else:
                hidden_states = self.decoder(decoder_input, memory, attention_mask)
        else:
            # Inference mode - start with memory
            hidden_states = memory

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return logits
