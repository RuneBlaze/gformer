import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import treeswift as ts
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from constants import MAX_TAXA, INTERNAL_NODE

try:
    from transformers.optimization import Adafactor

    HAVE_ADAFACTOR = True
except ImportError:
    HAVE_ADAFACTOR = False

from constants import EOS, PAD, VOCAB_SIZE
from data import TreeDataset
from layers import ModelConfig, TreeTransformer
from tokenizer import NewickTokenizer


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
                    dist_matrix[i, j] = int(dist_dict[u][v])

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
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_optimizer(model: torch.nn.Module, config: "TrainingConfig"):
    """Get memory efficient optimizer based on config"""
    optimizer_name = config.optimizer["name"].lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            fused=config.optimizer.get(
                "memory_efficient", False
            ),  # Use fused if available
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            fused=config.optimizer.get("memory_efficient", False),
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif optimizer_name == "adafactor" and HAVE_ADAFACTOR:
        return Adafactor(
            model.parameters(),
            lr=config.learning_rate,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_epoch(
    model: TreeTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: Accelerator,
    epoch: int,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Modified training loop to use Accelerate"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )

    for batch_idx, (tree_encodings, target_seq) in enumerate(progress_bar):
        with accelerator.accumulate(model):
            # Forward pass
            logits = model(
                tree_encodings=tree_encodings,
                output_tokens=target_seq[:, :-1],
                attention_mask=None,  # Will be auto-generated
            )

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_seq[:, 1:].reshape(-1),
                ignore_index=PAD,
            )

            # Backward pass handled by accelerator
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def compute_tree_accuracy(
    predicted_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> float:
    """Compute accuracy of tree topology prediction"""
    # Ignore padding tokens in comparison
    mask = target_tokens != EOS  # Updated from END_OF_INPUT
    correct = (predicted_tokens == target_tokens) & mask
    return correct.float().sum() / mask.sum()


def validate(model, val_loader, device) -> dict:
    model.eval()
    metrics = {"loss": 0.0, "accuracy": 0.0}

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
                attention_mask = (
                    torch.triu(torch.ones(pos, pos), diagonal=1).bool().to(device)
                )

                logits = model(
                    tree_encodings=tree_encodings,
                    output_tokens=partial_target,
                    attention_mask=attention_mask,
                )

                print(logits.shape)

                print(F.softmax(logits, dim=-1))

                # Loss for next token prediction
                loss = F.cross_entropy(
                    logits[:, -1, :],
                    target_seq[:, pos],
                )
                sequence_loss += loss.item()

                # Accuracy for next token prediction
                pred = logits[:, -1, :].argmax(dim=-1)
                correct = pred == target_seq[:, pos]
                sequence_correct += correct.sum().item()
                total_tokens += target_seq.size(0)

            metrics["loss"] += sequence_loss / (seq_length - 1)
            metrics["accuracy"] += sequence_correct / total_tokens

    # Average metrics
    for k in metrics:
        metrics[k] /= len(val_loader)

    return metrics


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    optimizer: dict
    mixed_precision: bool
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    save_interval: int

    @classmethod
    def from_yaml(cls, yaml_path: str) -> tuple["TrainingConfig", ModelConfig]:
        """Load both training and model configs from yaml file"""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Parse training config
        training_config = config["training"]
        train_cfg = cls(
            batch_size=int(training_config["batch_size"]),
            learning_rate=float(training_config["learning_rate"]),
            warmup_steps=int(training_config["warmup_steps"]),
            grad_clip=float(training_config["grad_clip"]),
            optimizer=training_config["optimizer"],
            mixed_precision=training_config.get("mixed_precision", False),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            gradient_accumulation_steps=int(
                training_config.get("gradient_accumulation_steps", 1)
            ),
            save_interval=int(training_config.get("save_interval", 1)),
        )

        # Parse model config
        model_cfg = ModelConfig.from_yaml(yaml_path)  # This already handles the "model" section

        return train_cfg, model_cfg


def show_example_completions(
    model, val_loader, device, tokenizer, num_examples=5
) -> None:
    """Show example completions from the model with constrained generation"""
    model.eval()

    with torch.no_grad():
        # Get a batch of data
        tree_encodings, target_seq = next(iter(val_loader))
        tree_encodings = tree_encodings.to(device)
        target_seq = target_seq.to(device)

        # Show completions for first num_examples in batch
        for i in range(min(num_examples, target_seq.size(0))):
            # Get ground truth completion
            gt_completion = tokenizer.decode(target_seq[i].cpu().tolist())

            # Start with just first token (usually INTERNAL_NODE)
            curr_seq = target_seq[i:i+1, :1]
            
            # Track which taxa have been seen
            seen_taxa = set()
            
            # Generate tokens until we hit EOS or max length
            for pos in range(512):
                # Create causal mask for current sequence length
                attention_mask = torch.triu(
                    torch.ones(curr_seq.size(1), curr_seq.size(1)), diagonal=1
                ).bool().to(device)

                # Get next token prediction
                logits = model(
                    tree_encodings=tree_encodings[i:i+1],  # Keep batch dimension
                    output_tokens=curr_seq,
                    attention_mask=attention_mask,
                )
                
                # Get probabilities for next token
                probs = F.softmax(logits[:, -1:, :], dim=-1)
                
                # Modify probabilities based on constraints
                if len(seen_taxa) < MAX_TAXA:
                    # Set EOS probability to 0 until all taxa are seen
                    probs[0, 0, EOS] = 0
                    
                    # Set probabilities of seen taxa to 0 to prevent repeats
                    for taxon in seen_taxa:
                        probs[0, 0, taxon] = 0
                    
                    # Renormalize probabilities
                    probs = probs / probs.sum()

                # Get most likely next token
                next_token = probs.argmax(dim=-1)
                next_token_val = next_token.item()

                # Update seen taxa if this is a taxon
                if 0 <= next_token_val < MAX_TAXA:
                    seen_taxa.add(next_token_val)

                # Break if we hit EOS and all taxa have been seen
                if next_token_val == EOS and len(seen_taxa) >= MAX_TAXA:
                    break

                # Add predicted token to sequence
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
            
            pred_completion = tokenizer.decode([INTERNAL_NODE] + curr_seq[0].cpu().tolist())

            # Print comparison
            print(f"\nExample {i + 1}:")
            print(f"Ground truth: {gt_completion}")
            print(f"Prediction:   {pred_completion}")
            print(f"Taxa seen: {sorted(list(seen_taxa))}")

            # Print first few logits for debugging
            if pos == 0:
                probs = F.softmax(logits[0, 0], dim=-1)
                top_k = 5
                values, indices = probs.topk(top_k)
                print("\nTop predictions for first position:")
                for prob, idx in zip(values, indices):
                    if idx == tokenizer.INTERNAL_NODE:
                        token_name = "INTERNAL_NODE"
                    elif idx == tokenizer.EOS:
                        token_name = "EOS"
                    else:
                        token_name = str(idx.item())
                    print(f"{token_name}: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train TreeTransformer model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to data file (.jsonl or .parquet)"
    )
    train_parser.add_argument(
        "--lr", type=float, help="Learning rate (overrides config file)"
    )
    train_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of directories to use for validation (for parquet files)",
    )
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    train_parser.add_argument(
        "--batch-size", type=int, help="Batch size (overrides config file)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
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
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        help="Gradient clipping value (overrides config file)",
    )
    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Learning rate warmup steps (overrides config file)",
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes for parallel preprocessing",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of gradient accumulation steps (overrides config file)",
        default=1,
    )
    train_parser.add_argument(
        "--save-interval",
        type=int,
        help="Save checkpoint every N epochs (overrides config file)",
    )
    train_parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    train_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tree-transformer",
        help="Weights & Biases project name",
    )
    train_parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training (overrides config file)",
    )

    # Count parameters subcommand
    count_parser = subparsers.add_parser("count-params", help="Count model parameters")
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

    if args.command == "count-params":
        device = torch.device(args.device)
        config = ModelConfig.from_yaml(args.config)
        model = TreeTransformer(config).to(device)
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return

    elif args.command == "train":
        # Load both configs
        training_config, model_config = TrainingConfig.from_yaml(args.config)

        # Override config values with command line arguments if provided
        if args.save_interval is not None:
            training_config.save_interval = args.save_interval
        if args.lr is not None:
            training_config.learning_rate = args.lr
        if args.batch_size is not None:
            training_config.batch_size = args.batch_size
        if args.warmup_steps is not None:
            training_config.warmup_steps = args.warmup_steps
        if args.grad_clip is not None:
            training_config.grad_clip = args.grad_clip
        if args.mixed_precision:
            training_config.mixed_precision = True
        if args.gradient_accumulation_steps is not None:
            training_config.gradient_accumulation_steps = args.gradient_accumulation_steps

        # Initialize accelerator with the proper gradient_accumulation_steps from config
        accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision="fp16" if training_config.mixed_precision else "no",
            log_with="wandb" if args.use_wandb else None,
        )

        # Setup logging
        logger = get_logger(__name__)

        # Set seed for reproducibility
        if args.seed is not None:
            set_seed(args.seed)

        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)

        # Initialize model with model config
        model = TreeTransformer(model_config)

        # Create datasets and dataloaders
        train_dataset = TreeDataset(
            args.data,
            split="train",
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        val_dataset = TreeDataset(
            args.data,
            split="val",
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size or training_config.batch_size,
            shuffle=True,
            num_workers=1,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size or training_config.batch_size,
            shuffle=False,
            num_workers=1,
        )

        # Use training config for training parameters
        optimizer = get_optimizer(model, training_config)

        # Adjust total steps for scheduler
        total_steps = (
            len(train_loader) // args.gradient_accumulation_steps
        ) * args.epochs
        scheduler = get_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps or training_config.warmup_steps,
        )

        # Prepare everything with accelerator
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Initialize wandb if requested
        if accelerator.is_main_process and args.use_wandb:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config={
                    "learning_rate": args.lr or training_config.learning_rate,
                    "batch_size": args.batch_size or training_config.batch_size,
                    "epochs": args.epochs,
                    "model_config": model_config.__dict__,
                },
            )

        # Training loop
        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                accelerator,
                epoch,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

                # Log metrics
                if args.use_wandb:
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }
                    )

                # Show example completions
                logger.info("Example completions:")
                show_example_completions(
                    accelerator.unwrap_model(model),
                    val_loader,
                    accelerator.device,
                    NewickTokenizer(),
                )

                # Save checkpoint if needed
                if training_config.save_interval > 0 and (epoch + 1) % training_config.save_interval == 0:
                    checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
                    accelerator.save_state(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        # End wandb run if used
        if accelerator.is_main_process and args.use_wandb:
            accelerator.end_training()


if __name__ == "__main__":
    main()
