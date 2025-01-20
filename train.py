import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from layers import END_OF_INPUT, END_OF_OUTPUT, VOCAB_SIZE, TreeTransformer, ModelConfig
from data import TreeDataset


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

    @staticmethod
    def newick_to_distance_matrix(newick_str: str) -> np.ndarray:
        """
        Convert a single Newick format tree string to a topological distance matrix.
        Each entry d[i,j] represents the number of edges between taxa i and j,
        ignoring edge lengths.
        
        Args:
            newick_str: Tree in Newick format
            
        Returns:
            np.ndarray: NxN distance matrix where N is number of taxa
        """
        # Parse newick string into TreeSwift tree
        tree = ts.read_tree_newick(newick_str)
        
        # Set all edge lengths to 1 for topological distance
        for node in tree.traverse_postorder():
            if not node.is_root():
                node.edge_length = 1
                
        # Get distance matrix as dictionary using TreeSwift's built-in method
        dist_dict = tree.distance_matrix(leaf_labels=True)
        
        # Get sorted list of taxa names to ensure consistent ordering
        taxa = sorted(dist_dict.keys())
        n = len(taxa)
        
        # Create numpy array from distance dictionary
        dist_matrix = np.zeros((n, n), dtype=np.uint8)
        for i, u in enumerate(taxa):
            for j, v in enumerate(taxa):
                if i != j:
                    # No need to round since all distances will be integers
                    dist_matrix[i,j] = int(dist_dict[u][v])
                    
        return dist_matrix


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


def get_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: TreeTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
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
        seq_length = target_seq.size(1)

        # Teacher forcing: use target sequence as input, shifted right
        decoder_input = target_seq[:, :-1]  # Remove last token
        target_output = target_seq[:, 1:]   # Remove first token

        # Create causal mask for decoder self-attention
        attention_mask = torch.triu(
            torch.ones(decoder_input.size(1), decoder_input.size(1)),
            diagonal=1
        ).bool().to(device)

        # Forward pass
        logits = model(
            tree_encodings=tree_encodings,
            output_tokens=decoder_input,
            attention_mask=attention_mask,
        )

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target_output.reshape(-1),
            ignore_index=END_OF_INPUT,  # Ignore padding tokens
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def compute_tree_accuracy(predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
    """Compute accuracy of tree topology prediction"""
    # Ignore padding tokens in comparison
    mask = target_tokens != END_OF_INPUT  # Using constant from layers module
    correct = (predicted_tokens == target_tokens) & mask
    return correct.float().sum() / mask.sum()


def validate(model, val_loader, device) -> dict:
    model.eval()
    metrics = {
        'loss': 0.0,
        'accuracy': 0.0
    }
    
    with torch.no_grad():
        for tree_encodings, target_seq in val_loader:
            tree_encodings = tree_encodings.to(device)
            target_seq = target_seq.to(device)

            num_trees = tree_encodings.size(1)
            seq_length = target_seq.size(1)
            
            sequence_loss = 0
            sequence_correct = 0
            total_tokens = 0
            
            # Evaluate each position in sequence
            for pos in range(1, seq_length):
                partial_target = target_seq[:, :pos]
                attention_mask = create_causal_mask(
                    pos + num_trees, num_trees, device
                )

                logits = model(
                    tree_encodings=tree_encodings,
                    output_tokens=partial_target,
                    attention_mask=attention_mask,
                )

                # Loss for next token prediction
                loss = F.cross_entropy(
                    logits[:, -1, :],
                    target_seq[:, pos],
                )
                sequence_loss += loss.item()
                
                # Accuracy for next token prediction
                pred = logits[:, -1, :].argmax(dim=-1)
                correct = (pred == target_seq[:, pos])
                sequence_correct += correct.sum().item()
                total_tokens += target_seq.size(0)

            metrics['loss'] += sequence_loss / (seq_length - 1)
            metrics['accuracy'] += sequence_correct / total_tokens
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(val_loader)
        
    return metrics


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description="Train TreeTransformer model")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        "--data", type=str, required=True, 
        help="Path to data file (.jsonl or .parquet)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Ratio of directories to use for validation (for parquet files)"
    )
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    train_parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    train_parser.add_argument('--grad-clip', type=float, default=1.0, 
                       help='Gradient clipping value')
    train_parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Learning rate warmup steps')
    train_parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    train_parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of processes for parallel preprocessing')

    # Count parameters subcommand
    count_parser = subparsers.add_parser('count-params', help='Count model parameters')
    count_parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    count_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to initialize model on",
    )

    args = parser.parse_args()

    if args.command == 'count-params':
        device = torch.device(args.device)
        config = ModelConfig.from_yaml(args.config)
        model = TreeTransformer(config).to(device)
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return

    elif args.command == 'train':
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)

        # Initialize model and move to device
        device = torch.device(args.device)
        config = ModelConfig.from_yaml(args.config)
        model = TreeTransformer(config).to(device)

        # Create datasets with automatic train/val splitting
        train_dataset = TreeDataset(
            args.data,
            split='train',
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers
        )
        
        val_dataset = TreeDataset(
            args.data,
            split='val', 
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers
        )

        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Add cosine learning rate scheduler with warmup
        total_steps = len(train_loader) * args.epochs
        scheduler = get_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

            # Validate
            val_metrics = validate(model, val_loader, device)
            logger.info(f"Epoch {epoch} - Validation loss: {val_metrics['loss']:.4f}")
            logger.info(f"Epoch {epoch} - Validation accuracy: {val_metrics['accuracy']:.4f}")

            # Save checkpoint if validation loss improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_metrics['loss'],
                        "val_accuracy": val_metrics['accuracy'],
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
