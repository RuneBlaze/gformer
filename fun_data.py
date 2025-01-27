from data import TreeDataset
from rich.tree import Tree
from rich import print as rprint
import numpy as np
import torch
from typing import Any
from layers import TreeTransformer, ModelConfig
from constants import MAX_SEQUENCE_LENGTH, MAX_TAXA, VOCAB_SIZE
from torch.nn.utils.rnn import pad_sequence


def print_pytree(obj: Any, tree: Tree = None, name: str = "PyTree") -> None:
    """Pretty prints a PyTree structure using rich, showing shapes and dtypes of arrays/tensors."""
    if tree is None:
        tree = Tree(name)

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            branch = tree.add(f"[cyan]{type(obj).__name__}[{i}]")
            print_pytree(item, branch, "")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            branch = tree.add(f"[yellow]{key}")
            print_pytree(value, branch, "")
    elif isinstance(obj, torch.Tensor):
        tree.add(f"[red]Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})")
    else:
        tree.add(f"[blue]{type(obj).__name__}({obj})")


def recursively_print_shape(obj: Any) -> None:
    """Print PyTree structure with rich formatting."""
    tree = Tree("🌳 PyTree Structure")
    print_pytree(obj, tree)
    rprint(tree)


def test_model_for_nans():
    """Test if model produces NaN values with batch size 2 using real data."""
    # Create a simple model config
    config = ModelConfig(
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        mlp_hidden_dim=1024,
        tree_embedding_dim=256,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # Initialize model
    model = TreeTransformer(config)
    model.train()  # Set to training mode

    # Load dataset
    dataset = TreeDataset("overfit_dataset.parquet")

    # Get two data points and collate them
    data_point1 = dataset[0]
    data_point2 = dataset[1]

    # Collate tree encodings
    tree_encodings = torch.stack(
        [
            data_point1[0],  # tree tensor from first data point
            data_point2[0],  # tree tensor from second data point
        ]
    )

    # Collate and pad species tokens
    target_seq = pad_sequence(
        [data_point1[1], data_point2[1]], batch_first=True, padding_value=0
    )

    # Create attention mask exactly as in training
    seq_length = target_seq.size(1) - 1  # -1 because we'll use [:,:-1] for input
    attention_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

    # Run forward pass exactly as in training
    logits = model(
        tree_encodings=tree_encodings,
        output_tokens=target_seq[:, :-1],
        attention_mask=attention_mask,
    )

    # Check shapes match training expectation
    rprint("[bold green]Shape Check:[/bold green]")
    rprint(f"Input target_seq shape: {target_seq.shape}")
    rprint(f"Input tokens shape (after slice): {target_seq[:, :-1].shape}")
    rprint(f"Target tokens shape (for loss): {target_seq[:, 1:].shape}")
    rprint(f"Logits shape: {logits.shape}")

    # Compute loss exactly as in training
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        target_seq[:, 1:].reshape(-1),
        ignore_index=0,  # PAD token
    )
    rprint(f"Loss value: {loss.item()}")

    # Check for NaN values
    has_nans = torch.isnan(logits).any() or torch.isnan(loss).any()

    # Print results
    rprint("[bold green]Model NaN Test Results:[/bold green]")
    rprint(f"Tree encodings shape: {tree_encodings.shape}")
    rprint(
        f"Contains NaN in logits: {'[red]Yes[/red]' if torch.isnan(logits).any() else '[green]No[/green]'}"
    )
    rprint(
        f"Contains NaN in loss: {'[red]Yes[/red]' if torch.isnan(loss).any() else '[green]No[/green]'}"
    )

    if has_nans:
        # If there are NaNs, print where they occur
        nan_locations = torch.where(torch.isnan(logits))
        rprint("[red]NaN values found at positions:[/red]")
        rprint(nan_locations)

    return not has_nans


if __name__ == "__main__":
    success = test_model_for_nans()
    rprint(f"\nTest {'[green]passed[/green]' if success else '[red]failed[/red]'}")
