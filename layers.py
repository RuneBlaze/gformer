import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
MAX_TAXA = 256  # Maximum number of taxa (0-255)
VOCAB_SIZE = MAX_TAXA + 4  # Taxa + (, ), EOS, EOI tokens
EMBEDDING_DIM = 768  # Following GPT2-small
NUM_HEADS = 12  # Following GPT2-small
NUM_LAYERS = 12  # Following GPT2-small
MLP_HIDDEN_DIM = 3072  # Following GPT2-small (4x embedding dim)
TREE_EMBEDDING_DIM = 768  # Same as model dimension for simplicity
MAX_SEQUENCE_LENGTH = 1024  # Following GPT2-small

# Special tokens
LEFT_PAREN = MAX_TAXA  # ( token
RIGHT_PAREN = MAX_TAXA + 1  # ) token
END_OF_INPUT = MAX_TAXA + 2  # EOI token
END_OF_OUTPUT = MAX_TAXA + 3  # EOS token


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


class TreeTransformer(nn.Module):
    """
    Decoder-only transformer model (like GPT) for processing gene trees and generating species trees.
    """

    def __init__(self):
        super().__init__()

        # Token embedding layer
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        # Tree embedding MLP
        self.tree_embedding = DistanceMatrixMLP(
            input_dim=8,  # Binary distance encoding dimension
            hidden_dim=EMBEDDING_DIM * 2,
            output_dim=TREE_EMBEDDING_DIM,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(EMBEDDING_DIM)

        # Decoder-only transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=MLP_HIDDEN_DIM,
            batch_first=True,
            norm_first=True,  # Pre-LN architecture for better stability
        )
        # Remove cross-attention by setting memory to None
        decoder_layer.multihead_attn = None
        decoder_layer.linear1 = nn.Linear(EMBEDDING_DIM, MLP_HIDDEN_DIM)
        decoder_layer.linear2 = nn.Linear(MLP_HIDDEN_DIM, EMBEDDING_DIM)

        self.transformer = nn.ModuleList([decoder_layer for _ in range(NUM_LAYERS)])

        # Output projection
        self.output_projection = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(
        self,
        tree_encodings: torch.Tensor,  # [batch_size, num_trees, input_dim]
        output_tokens: torch.Tensor = None,  # [batch_size, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            tree_encodings: Tensor containing distance matrix encodings for each tree
            output_tokens: Target sequence tokens (for training)
            attention_mask: Causal attention mask

        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size = tree_encodings.size(0)
        num_trees = tree_encodings.size(1)

        # Embed trees using MLP
        tree_embeddings = self.tree_embedding(
            tree_encodings.view(-1, tree_encodings.size(-1))
        )
        tree_embeddings = tree_embeddings.view(batch_size, num_trees, -1)

        if output_tokens is not None:
            # Training mode
            token_embeddings = self.token_embedding(output_tokens)

            # Concatenate tree embeddings with token embeddings
            sequence = torch.cat([tree_embeddings, token_embeddings], dim=1)

            # Add positional encoding (only to output tokens)
            sequence[:, num_trees:] = self.pos_encoding(
                sequence[:, num_trees:], start_pos=0
            )

        else:
            # Inference mode - only tree embeddings
            sequence = tree_embeddings

        # Generate causal mask if not provided
        if attention_mask is None:
            seq_length = sequence.size(1)
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length), diagonal=1
            ).bool()
            attention_mask = attention_mask.to(sequence.device)

        # Pass through decoder layers
        hidden_states = sequence
        for decoder_layer in self.transformer:
            hidden_states = decoder_layer(
                hidden_states,
                memory=None,  # No encoder-decoder attention
                tgt_mask=attention_mask,
            )

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return logits
