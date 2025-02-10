from typing import List, Tuple
import torch
import numpy as np
import treeswift as ts
from data import TreeDataset
from tree_utils import neighbor_joining, run_wastrid, rf_distance
from tokenizer import NewickTokenizer
from rich.console import Console
from utils import only_topology
from rich.table import Table
import statistics

console = Console()

def distance_matrix_from_binary(binary_matrix: torch.Tensor) -> np.ndarray:
    """
    Convert binary-encoded distance matrix back to regular distance matrix.
    
    Args:
        binary_matrix: Shape (n_entries, 8) where n_entries is from upper triangular
        
    Returns:
        Full distance matrix as numpy array
    """
    # Calculate matrix size from number of upper triangular entries
    n_entries = binary_matrix.shape[0]
    n = int((1 + (1 + 8*n_entries)**0.5) / 2)
    
    # Initialize distance matrix
    dist_matrix = np.zeros((n, n), dtype=np.uint8)
    
    # Convert binary back to distances
    distances = np.zeros(n_entries, dtype=np.uint8)
    for bit in range(8):
        distances += binary_matrix[:, bit].numpy().astype(np.uint8) * (1 << bit)
    # Fill upper triangular part
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i,j] = distances[idx]
            dist_matrix[j,i] = distances[idx]  # Mirror since distance matrix is symmetric
            idx += 1
            
    return dist_matrix

def baseline_predict(dataset: TreeDataset, idx: int) -> Tuple[str, str, dict]:
    """
    Make a baseline prediction using NJ + WASTRID and calculate RF distance.
    
    Args:
        dataset: TreeDataset instance
        idx: Index to predict
        
    Returns:
        Tuple of (predicted species tree, true species tree, rf_metrics)
    """
    # Get data point
    tree_tensor, species_tokens = dataset[idx]
    
    # Get number of actual trees (non-padded)
    n_trees = (tree_tensor.sum(dim=(1,2)) != 0).sum().item()
    tree_tensor = tree_tensor[:n_trees]
    
    # Convert each gene tree matrix to Newick string using NJ
    newick_trees = []
    for i in range(n_trees):
        # Convert binary matrix back to distance matrix
        dist_matrix = distance_matrix_from_binary(tree_tensor[i])
        
        # Run NJ
        tree = neighbor_joining(dist_matrix)
        newick_trees.append(tree.newick())
    
    # Run WASTRID to get species tree prediction
    predicted_tree = run_wastrid(newick_trees)
    
    # Decode true species tree
    true_tree = dataset.tokenizer.decode(species_tokens.tolist())
    
    # Convert strings to TreeSwift objects and get only topology
    predicted_ts = ts.read_tree_newick(predicted_tree)
    true_ts = ts.read_tree_newick(true_tree)
    
    predicted_topology = only_topology(predicted_ts)
    true_topology = only_topology(true_ts)
    
    # Calculate RF distance using the topology-only trees
    rf_metrics = rf_distance(predicted_topology.newick(), true_topology.newick(), normalize=True)
    
    return predicted_topology.newick(), true_topology.newick(), rf_metrics

if __name__ == "__main__":
    # Lists to store metrics
    rf_distances = []
    rf_normalized = []
    false_positives = []
    false_negatives = []
    
    console.print("\n[cyan]Running baseline on 1000 examples...[/cyan]")
    
    # Initialize dataset
    dataset = TreeDataset("/Users/lbq/goof/teedeelee/assets/processed_family.pkl")
    
    # Test on 1000 examples
    for idx in range(1000):
        try:
            predicted, true, rf_metrics = baseline_predict(dataset, idx)
            rf_distances.append(rf_metrics['rf_distance'])
            rf_normalized.append(rf_metrics['rf_normalized'])
            false_positives.append(rf_metrics['false_positives'])
            false_negatives.append(rf_metrics['false_negatives'])
            
            if (idx + 1) % 100 == 0:
                console.print(f"Processed {idx + 1} examples...")
                
        except Exception as e:
            console.print(f"[red]Error processing example {idx}: {e}[/red]")
    
    # Create summary statistics table
    table = Table(title="Baseline Results (1000 examples)")
    
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    
    metrics = {
        "RF Distance": rf_distances,
        "Normalized RF": rf_normalized,
        "False Positives": false_positives,
        "False Negatives": false_negatives
    }
    
    for metric_name, values in metrics.items():
        table.add_row(
            metric_name,
            f"{statistics.mean(values):.4f}",
            f"{statistics.median(values):.4f}",
            f"{statistics.stdev(values):.4f}",
            f"{min(values):.4f}",
            f"{max(values):.4f}"
        )
    
    console.print("\n")
    console.print(table)
