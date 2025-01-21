import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import treeswift as ts
from rich.progress import track
from rich.console import Console
from rich import print as rprint
import argparse
import pyarrow.parquet as pq
import random
from multiprocessing import Pool, cpu_count
from constants import MAX_TAXA, MAX_GTREES, PAD
from tokenizer import NewickTokenizer

console = Console()

@dataclass
class InputPair:
    gtrees: list[str]
    stree: str

    @staticmethod
    def newick_to_distance_matrix(newick_str: str) -> np.ndarray:
        """
        Convert a single Newick format tree string to a topological distance matrix.
        Each entry d[i,j] represents the number of edges between taxa i and j,
        ignoring edge lengths.
        """
        tree = ts.read_tree_newick(newick_str)
        
        # Set all edge lengths to 1 for topological distance
        for node in tree.traverse_postorder():
            if not node.is_root():
                node.edge_length = 1
                
        # Get distance matrix as dictionary using TreeSwift's built-in method
        dist_dict = tree.distance_matrix(leaf_labels=True)
        
        # Get sorted list of taxa names to ensure consistent ordering
        taxa = sorted(dist_dict.keys())
        n = MAX_TAXA
        
        # Create numpy array from distance dictionary
        dist_matrix = np.zeros((n, n), dtype=np.uint8)
        for i, u in enumerate(taxa):
            for j, v in enumerate(taxa):
                if i != j:
                    dist_matrix[i,j] = int(dist_dict[u][v])
        return dist_matrix


def encode_distance_matrix(distance_matrix: np.ndarray) -> torch.Tensor:
    """
    Encode the upper triangular part of a distance matrix (excluding diagonal) into binary representation.
    """
    N = distance_matrix.shape[0]
    assert distance_matrix.shape == (N, N), "Input must be a square matrix"

    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    distances = torch.from_numpy(distance_matrix).to(torch.uint8)
    binary_encoding = torch.zeros(N, N, 8, dtype=torch.float32)

    for bit in range(8):
        bit_value = (distances & (1 << bit)) >> bit
        binary_encoding[:, :, bit] = bit_value.float()

    flat_encoding = binary_encoding[mask]
    return flat_encoding


class TreeDataset(Dataset):
    def __init__(self, data_source: str, max_sequence_length: int = 1024, 
                 split: str = 'train', val_ratio: float = 0.2, seed: int = 42,
                 num_workers: int = 4):
        """
        Initialize the dataset.
        
        Args:
            data_source: Path to either .jsonl or .parquet file
            max_sequence_length: Maximum sequence length for tokenization
            split: Either 'train' or 'val'
            val_ratio: Ratio of directories to use for validation
            seed: Random seed for reproducibility
            num_workers: Number of processes for parallel preprocessing
        """
        self.max_sequence_length = max_sequence_length
        self.data: List[InputPair] = []
        self.tokenizer = NewickTokenizer()
        self.cached_encodings = []
        self.num_workers = min(num_workers, cpu_count())

        random.seed(seed)
        
        if data_source.endswith('.parquet'):
            self._load_from_parquet(data_source, split, val_ratio)
        else:
            self._load_from_jsonl(data_source)
            
        console.print(f"Pre-encoding trees for {split} split using {self.num_workers} processes...")
        self._parallel_encode_trees()

    def _load_from_parquet(self, parquet_path: str, split: str, val_ratio: float):
        """Load data from parquet file with train/val split based on directories"""
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        # Get unique directories and check if we have enough for directory-based split
        all_dirs = sorted(df['directory'].unique())
        n_val = int(len(all_dirs) * val_ratio)
        
        if n_val >= 1:
            # Directory-based split if we have enough directories
            val_dirs = set(random.sample(all_dirs, n_val))
            
            # Filter based on split
            if split == 'train':
                df = df[~df['directory'].isin(val_dirs)]
            else:  # val
                df = df[df['directory'].isin(val_dirs)]
        else:
            # Fallback to random row-based split if too few directories
            all_indices = list(range(len(df)))
            n_val_samples = int(len(df) * val_ratio)
            val_indices = set(random.sample(all_indices, n_val_samples))
            
            if split == 'train':
                df = df[~df.index.isin(val_indices)]
            else:  # val
                df = df[df.index.isin(val_indices)]
        
        console.print(f"Loading {split} split with {len(df)} examples")

        import treeswift as ts

        
        
        for _, row in df.iterrows():
            stree = ts.read_tree_newick(row['species_tree'])
            stree.order("num_descendants_then_label")
            self.data.append(
                InputPair(gtrees=row['gtrees'], stree=stree.newick().lstrip("[&R] "))
            )

    def _load_from_jsonl(self, jsonl_path: str):
        """Load data from jsonl file (legacy support)"""
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(
                    InputPair(gtrees=item["gtrees"], stree=item["species_tree"])
                )

    def __len__(self) -> int:
        return len(self.data)

    def encode_trees(self, pair: InputPair) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_trees = []
        for tree in pair.gtrees:
            distance_matrix = InputPair.newick_to_distance_matrix(tree)
            encoded = encode_distance_matrix(distance_matrix)
            encoded_trees.append(encoded)

        tree_tensor = torch.stack(encoded_trees, dim=0)
        species_tokens = torch.tensor(self.tokenizer.encode(pair.stree))
        species_tokens = torch.cat(
            [
                species_tokens,
                torch.tensor([self.tokenizer.EOS]),
            ]
        )

        return tree_tensor, species_tokens

    def _encode_single_item(self, pair: InputPair) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single item for parallel processing"""
        return self.encode_trees(pair)

    def _parallel_encode_trees(self):
        """Parallel processing of tree encoding using chunks"""
        CHUNK_SIZE = 100
        with Pool(processes=self.num_workers) as pool:
            total = len(self.data)
            chunks = [self.data[i:i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
            
            with console.status(f"[bold green]Processing trees...") as status:
                processed = 0
                for chunk_results in pool.imap(
                    self._encode_chunk, 
                    chunks,
                    chunksize=1  # Each "chunk" here is already a batch of items
                ):
                    self.cached_encodings.extend(chunk_results)
                    processed += len(chunk_results)
                    console.print(f"Processed {processed}/{total} trees")

    def _encode_chunk(self, pairs: List[InputPair]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode a chunk of items"""
        return [self.encode_trees(pair) for pair in pairs]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tree_tensor, species_tokens = self.cached_encodings[idx]
        
        # Get actual number of trees
        num_gene_trees = min(tree_tensor.size(0), MAX_GTREES)
        
        # Create padded tensor directly with zeros
        padded_tree_tensor = torch.zeros(
            (MAX_GTREES, tree_tensor.size(1), tree_tensor.size(2)),
            dtype=torch.float32
        )
        
        # Copy actual tree encodings
        padded_tree_tensor[:num_gene_trees] = tree_tensor[:num_gene_trees]
        
        return padded_tree_tensor, species_tokens


def parse_args():
    parser = argparse.ArgumentParser(description='Tree Dataset Loader')
    parser.add_argument('data_path', type=str, help='Path to data file (.jsonl or .parquet)')
    parser.add_argument('--max-seq-length', type=int, default=1024,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                      help='Ratio of directories to use for validation (for parquet files)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of processes for parallel preprocessing')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create datasets
    if args.data_path.endswith('.parquet'):
        train_dataset = TreeDataset(
            args.data_path, 
            max_sequence_length=args.max_seq_length,
            split='train',
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers
        )
        val_dataset = TreeDataset(
            args.data_path, 
            max_sequence_length=args.max_seq_length,
            split='val',
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers
        )
        
        console.print(f"[green]Dataset loaded successfully!")
        console.print(f"Train size: {len(train_dataset)}")
        console.print(f"Val size: {len(val_dataset)}")
    else:
        # Legacy jsonl support
        dataset = TreeDataset(
            args.data_path, 
            args.max_seq_length,
            num_workers=args.num_workers
        )
        console.print(f"[green]Dataset loaded successfully!")
        console.print(f"Dataset size: {len(dataset)}")
    
    # Look at first few items
    console.print("\n[yellow]Sample items:[/yellow]")
    dataset_to_inspect = train_dataset if args.data_path.endswith('.parquet') else dataset
    
    for i in range(min(3, len(dataset_to_inspect))):
        tree_tensor, species_tokens = dataset_to_inspect[i]
        console.print(f"\n[cyan]Item {i}:[/cyan]")
        console.print(f"Tree tensor shape: {tree_tensor.shape}")
        console.print(f"Species tokens shape: {species_tokens.shape}")
        console.print(f"First few species tokens: {species_tokens[:10]}")
        console.print(f"Tree tensor: {tree_tensor[:, :10]}")
        # Add decoded species tree output
        decoded_stree = dataset_to_inspect.tokenizer.decode(species_tokens.tolist())
        console.print(f"Decoded species tree: {decoded_stree}") 