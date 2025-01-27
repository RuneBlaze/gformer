import math
from dataclasses import dataclass

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
        return cls(**config["model"])


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    optimizer: dict
    mixed_precision: bool
    gradient_checkpointing: bool
    gradient_accumulation_steps: int = 1
    save_interval: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Ensure numeric values are properly converted
        training_config = config["training"]
        training_config["learning_rate"] = float(training_config["learning_rate"])
        training_config["grad_clip"] = float(training_config["grad_clip"])
        training_config["batch_size"] = int(training_config["batch_size"])
        training_config["warmup_steps"] = int(training_config["warmup_steps"])

        return cls(**training_config)


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
        self.pos_encoding = PositionalEncoding(
            config.embedding_dim, config.max_sequence_length
        )

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            batch_first=True,
            norm_first=False,
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
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(config.embedding_dim, VOCAB_SIZE)

    def _check_nan(self, tensor: torch.Tensor, step_name: str) -> None:
        """Helper method to check for NaN values and report the step where they occur with detailed diagnostics."""
        if torch.isnan(tensor).any():
            nan_mask = torch.isnan(tensor)
            nan_count = nan_mask.sum().item()

            # Get basic stats
            basic_info = (
                f"NaN detected in {step_name}!\n"
                f"Total NaNs: {nan_count}\n"
                f"Tensor shape: {tensor.shape}\n"
            )

            # Analyze NaN distribution across dimensions
            dim_info = "NaN distribution across dimensions:\n"
            for dim in range(tensor.dim()):
                nan_per_slice = nan_mask.sum(
                    dim=tuple(d for d in range(tensor.dim()) if d != dim)
                )
                nan_indices = torch.where(nan_per_slice > 0)[0].tolist()
                dim_info += f"Dimension {dim}: {len(nan_indices)}/{tensor.shape[dim]} slices contain NaNs\n"
                if len(nan_indices) < 10:  # Only show indices if there aren't too many
                    dim_info += f"    Indices with NaNs: {nan_indices}\n"

            # For 3D tensors (batch, sequence, features), analyze batch-wise patterns
            if tensor.dim() == 3:
                batch_info = "Batch analysis:\n"
                for b in range(tensor.shape[0]):
                    batch_nan_count = nan_mask[b].sum().item()
                    if batch_nan_count > 0:
                        batch_info += (
                            f"Batch {b}: {batch_nan_count} NaNs "
                            f"({batch_nan_count / (tensor.shape[1] * tensor.shape[2]):.1%} of elements)\n"
                        )

            # Get statistics of non-NaN values
            valid_values = tensor[~nan_mask]
            if len(valid_values) > 0:
                value_info = (
                    f"Non-NaN value range: [{valid_values.min():.3f}, {valid_values.max():.3f}]\n"
                    f"Non-NaN mean: {valid_values.mean():.3f}, std: {valid_values.std():.3f}\n"
                )
            else:
                value_info = "No valid values found - tensor contains all NaNs\n"

            # Check if the NaNs form a specific pattern (e.g., all in one feature dimension)
            pattern_info = "Pattern analysis:\n"
            if tensor.dim() == 3:
                feature_nan_count = nan_mask.sum(dim=(0, 1))
                seq_nan_count = nan_mask.sum(dim=(0, 2))
                if torch.all(feature_nan_count == feature_nan_count[0]):
                    pattern_info += (
                        "NaNs appear consistently across feature dimension\n"
                    )
                if torch.all(seq_nan_count == seq_nan_count[0]):
                    pattern_info += (
                        "NaNs appear consistently across sequence dimension\n"
                    )

            full_message = (
                f"{basic_info}\n{dim_info}\n{value_info}\n"
                f"{batch_info if tensor.dim() == 3 else ''}\n{pattern_info}"
            )
            raise ValueError(full_message)

    def forward(
        self,
        tree_encodings: torch.Tensor,  # [batch_size, num_gene_trees, num_distances, 8]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self._check_nan(tree_encodings, "input tree_encodings")

        batch_size, num_gene_trees, num_distances, bits = tree_encodings.shape
        # Reshape tree encodings to process each gene tree through MLP
        tree_encodings = einops.rearrange(
            tree_encodings, "b g d bits -> (b g) (d bits)", bits=8
        )
        self._check_nan(tree_encodings, "reshaped tree_encodings")

        encoded_trees = self.tree_embedding(tree_encodings)
        self._check_nan(encoded_trees, "MLP output (encoded_trees)")

        # Reshape back for encoder input
        encoder_input = einops.rearrange(
            encoded_trees, "(b g) e -> b g e", b=batch_size, g=num_gene_trees
        )
        self._check_nan(encoder_input, "encoder_input")

        # Create padding mask based on zero vectors
        # True indicates positions that should be masked (padded)
        tree_padding_mask = (tree_encodings.abs().sum(dim=-1) == 0).view(
            batch_size, num_gene_trees
        )

        # Run Transformer encoder
        memory = self.encoder(encoder_input, src_key_padding_mask=tree_padding_mask)
        self._check_nan(memory, "encoder output")

        if output_tokens is not None:
            # Training mode
            token_embeddings = self.token_embedding(output_tokens)
            self._check_nan(token_embeddings, "token embeddings")

            decoder_input = self.pos_encoding(token_embeddings)
            self._check_nan(decoder_input, "positional encoded embeddings")

            # Generate causal mask if not provided
            if attention_mask is None:
                seq_length = output_tokens.size(1)
                attention_mask = torch.triu(
                    torch.ones(seq_length, seq_length), diagonal=1
                ).bool()
                attention_mask = attention_mask.to(output_tokens.device)

            hidden_states = self.decoder(decoder_input, memory, attention_mask)
            self._check_nan(hidden_states, "decoder output")
        else:
            # Inference mode - start with memory
            hidden_states = memory

        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        self._check_nan(logits, "final logits")

        return logits
