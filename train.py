import argparse
import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from layers import END_OF_INPUT, END_OF_OUTPUT, VOCAB_SIZE, TreeTransformer


def encode_distance_matrix(distance_matrix: np.ndarray) -> torch.Tensor:
    """
    Encode the upper triangular part of a distance matrix (excluding diagonal) into binary representation.

    Args:
        distance_matrix: NxN numpy array containing uint8 distances

    Returns:
        torch.Tensor: Binary encoding of shape (num_upper_elements, 8) containing the flattened upper triangular values
    """
    N = distance_matrix.shape[0]
    assert distance_matrix.shape == (N, N), "Input must be a square matrix"

    # Create a mask for upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()

    # Convert to torch tensor if not already
    distances = torch.from_numpy(distance_matrix).to(torch.uint8)

    # Create binary encoding matrix
    binary_encoding = torch.zeros(N, N, 8, dtype=torch.float32)

    # Get binary representation for each bit position (0-7)
    for bit in range(8):
        bit_value = (distances & (1 << bit)) >> bit
        binary_encoding[:, :, bit] = bit_value.float()

    # Apply SiLU activation to the binary encodings
    binary_encoding = F.silu(binary_encoding)

    # Extract only the upper triangular elements
    flat_encoding = binary_encoding[mask]

    return flat_encoding


@dataclass
class InputPair:
    gtrees: list[str]
    species_tree: str

    def to_distance_matrix(self) -> np.ndarray:
        # Later implemented.
        ...


class TreeDataset(Dataset):
    def __init__(self, jsonl_path: str, max_sequence_length: int = 1024):
        self.max_sequence_length = max_sequence_length
        self.data: List[InputPair] = []

        # Load data from jsonl file
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(
                    InputPair(gtrees=item["gtrees"], species_tree=item["species_tree"])
                )

    def __len__(self) -> int:
        return len(self.data)

    def encode_trees(self, pair: InputPair) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input trees
        encoded_trees = []
        for tree in pair.gtrees:
            distance_matrix = pair.to_distance_matrix()  # You'll implement this
            encoded = encode_distance_matrix(distance_matrix)
            encoded_trees.append(encoded)

        # Stack encoded trees
        tree_tensor = torch.stack(encoded_trees, dim=0)

        # Tokenize output species tree (you'll need to implement this)
        species_tokens = tokenize_newick(pair.species_tree)

        # Add EOI and EOS tokens
        species_tokens = torch.cat(
            [
                torch.tensor([END_OF_INPUT]),
                species_tokens,
                torch.tensor([END_OF_OUTPUT]),
            ]
        )

        return tree_tensor, species_tokens

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.data[idx]
        return self.encode_trees(pair)


def create_causal_mask(
    seq_length: int, num_trees: int, device: torch.device
) -> torch.Tensor:
    """Create causal attention mask that allows trees to attend to each other"""
    mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)

    # Allow trees to attend to each other
    mask[:num_trees, :num_trees] = False

    # Create causal mask for output sequence
    mask[num_trees:, num_trees:] = torch.triu(
        torch.ones(
            seq_length - num_trees,
            seq_length - num_trees,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

    return mask


def train_epoch(
    model: TreeTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (tree_encodings, target_seq) in enumerate(progress_bar):
        tree_encodings = tree_encodings.to(device)
        target_seq = target_seq.to(device)

        batch_size = tree_encodings.size(0)
        num_trees = tree_encodings.size(1)

        # Create attention mask
        attention_mask = create_causal_mask(
            target_seq.size(1) + num_trees, num_trees, device
        )

        # Forward pass
        logits = model(
            tree_encodings=tree_encodings,
            output_tokens=target_seq,
            attention_mask=attention_mask,
        )

        # Compute loss (only on output sequence, not tree encodings)
        loss = F.cross_entropy(
            logits[:, num_trees:-1, :].reshape(-1, VOCAB_SIZE),
            target_seq[:, 1:].reshape(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train TreeTransformer model")
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--val-data", type=str, required=True, help="Path to validation data JSONL file"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create save directory
    import os

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize model and move to device
    device = torch.device(args.device)
    model = TreeTransformer().to(device)

    # Create datasets and dataloaders
    train_dataset = TreeDataset(args.train_data)
    val_dataset = TreeDataset(args.val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tree_encodings, target_seq in val_loader:
                tree_encodings = tree_encodings.to(device)
                target_seq = target_seq.to(device)

                num_trees = tree_encodings.size(1)
                attention_mask = create_causal_mask(
                    target_seq.size(1) + num_trees, num_trees, device
                )

                logits = model(
                    tree_encodings=tree_encodings,
                    output_tokens=target_seq,
                    attention_mask=attention_mask,
                )

                loss = F.cross_entropy(
                    logits[:, num_trees:-1, :].reshape(-1, VOCAB_SIZE),
                    target_seq[:, 1:].reshape(-1),
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch} - Validation loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()


# Helper function to be implemented
def tokenize_newick(newick_str: str) -> torch.Tensor:
    """Convert Newick string to sequence of token indices"""
    # You'll need to implement this based on your tokenization scheme
    ...
