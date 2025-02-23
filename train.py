import argparse
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import treeswift as ts
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    from transformers.optimization import Adafactor

    HAVE_ADAFACTOR = True
except ImportError:
    HAVE_ADAFACTOR = False

from constants import MAX_TAXA
from data import TreeDataset
from models import ModelConfig, FusedQuartetDecider, TrainingConfig


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


def train_epoch(model: FusedQuartetDecider,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                accelerator: Accelerator,
                epoch: int,
                grad_clip: float) -> float:
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        with accelerator.accumulate(model):
            # Extract inputs from the batch (each from TreeDataset in quartet mode)
            gtrees = batch['gtrees']  # shape: [B, MAX_GTREES, n_distances, 8]
            quartet_queries = batch['quartet_queries']  # shape: [B, num_quartets, 16]
            labels = batch['Y']  # shape: [B, num_quartets]

            # Forward pass through the fused model; output shape: [B, num_quartets, 3]
            logits = model(gtrees, quartet_queries)
            loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1))

            accelerator.backward(loss)
            # Apply gradient clipping if grad_clip is greater than zero
            if grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def val_epoch(model: FusedQuartetDecider,
              dataloader: DataLoader,
              accelerator: Accelerator,
              device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            gtrees = batch['gtrees'].to(device)
            quartet_queries = batch['quartet_queries'].to(device)
            labels = batch['Y'].to(device)

            logits = model(gtrees, quartet_queries)
            loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)  # shape: [B, num_quartets]
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Train FusedQuartetDecider model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to data file (.jsonl or .parquet)")
    train_parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    train_parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of directories to use for validation")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--num-workers", type=int, default=4, help="Number of processes for parallel preprocessing")

    # Count parameters subcommand
    count_parser = subparsers.add_parser("count-params", help="Count model parameters")
    count_parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")

    args = parser.parse_args()

    if args.command == "train":
        # Load configs
        model_config = ModelConfig.from_yaml(args.config)
        training_config = TrainingConfig.from_yaml(args.config)

        # Initialize accelerator with proper mixed precision and device settings
        accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision="fp16" if training_config.mixed_precision else "no",
            cpu=not torch.cuda.is_available(),  # use GPU if available
        )

        # Set seed for reproducibility
        set_seed(args.seed)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)

        # Create save directory
        if accelerator.is_local_main_process:
            os.makedirs(args.save_dir, exist_ok=True)

        device = accelerator.device
        # Instantiate the fused quartet decider model
        model = FusedQuartetDecider(model_config)

        # Create datasets with quartet classification enabled
        train_dataset = TreeDataset(
            args.data,
            split="train",
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            is_quartet_classification=True,
        )

        val_dataset = TreeDataset(
            args.data,
            split="val",
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            is_quartet_classification=True,
        )

        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=1)

        # Initialize optimizer and scheduler
        optimizer = get_optimizer(model, training_config)
        total_steps = (len(train_loader) // training_config.gradient_accumulation_steps) * args.epochs
        scheduler = get_scheduler(optimizer, total_steps=total_steps, warmup_steps=training_config.warmup_steps)

        # Prepare everything with accelerator
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

        # Training loop
        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                accelerator,
                epoch,
                training_config.grad_clip
            )

            if accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")
                val_metrics = val_epoch(model, val_loader, accelerator, device)
                logger.info(f"Epoch {epoch} - Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")

                # Save checkpoint if needed
                if training_config.save_interval > 0 and (epoch + 1) % training_config.save_interval == 0:
                    checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
                    accelerator.save({
                        "epoch": epoch,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    elif args.command == "count-params":
        config = ModelConfig.from_yaml(args.config)
        model = FusedQuartetDecider(config)
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return


if __name__ == "__main__":
    main()
