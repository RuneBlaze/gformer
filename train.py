import argparse
import logging
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from transformers.optimization import Adafactor

    HAVE_ADAFACTOR = True
except ImportError:
    HAVE_ADAFACTOR = False

from constants import EOS, PAD, VOCAB_SIZE
from data import TreeDataset
from layers import ModelConfig, TreeTransformer


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
) -> float:
    """Modified training loop to use Accelerate's built-in gradient accumulation"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )

    for batch_idx, (tree_encodings, target_seq) in enumerate(progress_bar):
        # Use accelerator's accumulate context manager
        with accelerator.accumulate(model):
            # Forward pass
            logits = model(
                tree_encodings=tree_encodings,
                output_tokens=target_seq[:, :-1],
                attention_mask=None,
            )

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_seq[:, 1:].reshape(-1),
                ignore_index=PAD,
            )

            # Backward pass handled by accelerator
            accelerator.backward(loss)

            # Optimizer and scheduler steps are handled automatically
            # by the accumulate context manager
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
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        training_config = config["training"]
        return cls(
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


def decode_sequence(
    model: TreeTransformer,
    tree_encodings: torch.Tensor,
    initial_tokens: torch.Tensor,
    device: torch.device,
    max_length: int = None,
) -> torch.Tensor:
    """
    Generate a sequence from the model given initial tokens and tree encodings.

    Args:
        model: The TreeTransformer model
        tree_encodings: Tensor of shape [batch_size, num_trees, encoding_dim]
        initial_tokens: Starting tokens tensor of shape [batch_size, seq_len]
        device: Device to run generation on
        max_length: Maximum sequence length (defaults to 2x initial sequence length)

    Returns:
        torch.Tensor: Generated sequence of shape [batch_size, seq_len]
    """
    model.eval()

    if max_length is None:
        max_length = initial_tokens.size(1) * 2

    curr_seq = initial_tokens

    with torch.no_grad():
        # Generate tokens until we hit EOS or max length
        for pos in range(max_length - initial_tokens.size(1)):
            # Create causal mask for current sequence length
            attention_mask = (
                torch.triu(torch.ones(curr_seq.size(1), curr_seq.size(1)), diagonal=1)
                .bool()
                .to(device)
            )

            # Get next token prediction
            logits = model(
                tree_encodings=tree_encodings,
                output_tokens=curr_seq,
                attention_mask=attention_mask,
            )

            # Get most likely next token
            next_token = logits[:, -1:, :].argmax(dim=-1)

            # Break if we hit EOS
            if (next_token == EOS).all():
                break

            # Add predicted token to sequence
            curr_seq = torch.cat([curr_seq, next_token], dim=1)

    return curr_seq


def show_example_completions(
    model, val_loader, device, tokenizer, num_examples=5
) -> None:
    """Show example completions from the model"""
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
            curr_seq = target_seq[i : i + 1, :1]

            # Generate completion
            completed_seq = decode_sequence(
                model, tree_encodings[i : i + 1], curr_seq, device
            )

            # Decode prediction
            pred_completion = tokenizer.decode(completed_seq[0].cpu().tolist())

            # Print comparison
            print(f"\nExample {i + 1}:")
            print(f"Ground truth: {gt_completion}")
            print(f"Prediction:   {pred_completion}")

            # Print first few logits for debugging
            if i == 0:  # Only for first example
                with torch.no_grad():
                    logits = model(
                        tree_encodings=tree_encodings[i : i + 1],
                        output_tokens=curr_seq,
                        attention_mask=None,
                    )
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
    )
    train_parser.add_argument(
        "--save-interval",
        type=int,
        help="Save checkpoint every N epochs (overrides config file)",
    )
    train_parser.add_argument(
        "--init-from",
        type=str,
        help="Initialize model weights from this checkpoint path",
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
        # Initialize accelerator with gradient accumulation
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
        )

        # Set seed for reproducibility
        set_seed(args.seed)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)

        # Create save directory
        if accelerator.is_local_main_process:
            os.makedirs(args.save_dir, exist_ok=True)

        # Initialize model
        device = accelerator.device
        config = ModelConfig.from_yaml(args.config)
        model = TreeTransformer(config)

        # Load initial weights if specified
        if args.init_from:
            logger.info(f"Initializing model weights from {args.init_from}")
            checkpoint = torch.load(args.init_from, map_location=device)
            # Only load model weights, ignore other checkpoint contents
            model.load_state_dict(checkpoint["model_state_dict"])

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
            batch_size=args.batch_size or config.batch_size,
            shuffle=True,
            num_workers=1,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size or config.batch_size,
            shuffle=False,
            num_workers=1,
        )

        # Initialize optimizer and scheduler
        optimizer = get_optimizer(model, config)
        total_steps = (
            len(train_loader) // args.gradient_accumulation_steps
        ) * args.epochs
        scheduler = get_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps or config.warmup_steps,
        )

        # Prepare everything with accelerator
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                accelerator,
                epoch,
            )

            # Only log on main process
            if accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")
                logger.info("Example completions:")
                show_example_completions(model, val_loader, device, tokenizer)

                # Save checkpoint if we've hit the save interval
                if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                    checkpoint_path = os.path.join(
                        args.save_dir, f"model_epoch_{epoch}.pt"
                    )
                    # Use accelerator's save/load methods
                    accelerator.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": accelerator.unwrap_model(
                                model
                            ).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                        },
                        checkpoint_path,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
