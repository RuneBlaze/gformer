import argparse
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Literal, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import treeswift as ts
from rich.console import Console
from torch.utils.data import Dataset
import xxhash

from constants import MAX_GTREES, MAX_TAXA, PAD, QUARTET_SAMPLES
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
        taxa_set = set(taxa)
        for i in range(MAX_TAXA):
            if str(i) not in taxa_set:
                raise ValueError(f"Taxon {i} not found in tree {newick_str}")
        n = MAX_TAXA

        # Create numpy array from distance dictionary
        dist_matrix = np.zeros((n, n), dtype=np.uint8)
        for i, u in enumerate(taxa):
            for j, v in enumerate(taxa):
                if i != j:
                    dist_matrix[i, j] = int(dist_dict[u][v])
        return dist_matrix

    @staticmethod
    def decide_quartet_topology(distance_matrix: np.ndarray, abcd: tuple[int, int, int, int]) -> Literal[0, 1, 2]:
        # (AB|CD, AC|BD, AD|BC)
        abcd = tuple(sorted(abcd))
        a, b, c, d = abcd
        
        # Calculate the three possible sums of pairwise distances
        ab_cd = distance_matrix[a,b] + distance_matrix[c,d]  # supports AB|CD
        ac_bd = distance_matrix[a,c] + distance_matrix[b,d]  # supports AC|BD
        ad_bc = distance_matrix[a,d] + distance_matrix[b,c]  # supports AD|BC
        
        # Find which sum is smallest
        sums = [ab_cd, ac_bd, ad_bc]
        return sums.index(min(sums))

    
    def generate_queries_and_gt(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        # Create a hash from the first 3 gtrees and stree for seeding
        hasher = xxhash.xxh64()
        for tree in self.gtrees[:3]:
            hasher.update(tree.encode())
        hasher.update(self.stree.encode())
        seed = hasher.intdigest()
        rng = np.random.RandomState(seed % 2**32)
        
        # Initialize output arrays
        queries = np.zeros((num_samples, 4), dtype=np.int32)
        gt = np.zeros(num_samples, dtype=np.int32)
        stree_matrix = self.newick_to_distance_matrix(self.stree)
        # Generate samples
        sample_idx = 0
        while sample_idx < num_samples:
            # Generate a random 4-combination
            quartet = tuple(sorted(rng.choice(MAX_TAXA, size=4, replace=False)))
            
            # Get topology from species tree (ground truth)
            topology = self.decide_quartet_topology(stree_matrix, quartet)
            
            # Add to outputs
            queries[sample_idx] = quartet
            gt[sample_idx] = topology
            sample_idx += 1
        
        # Convert queries to one-hot encoding
        queries_onehot = np.zeros((num_samples, MAX_TAXA * 4), dtype=bool)
        for i in range(num_samples):
            for j, taxon in enumerate(queries[i]):
                queries_onehot[i, j * MAX_TAXA + taxon] = True
            
        return queries_onehot, gt


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
    def __init__(
        self,
        data_source: str,
        max_sequence_length: int = 1024,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        num_workers: int = 4,
        num_quartets: int = QUARTET_SAMPLES,
        is_quartet_classification: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: Path to parquet file
            max_sequence_length: Maximum sequence length for tokenization
            split: Either 'train' or 'val'
            val_ratio: Ratio of directories to use for validation
            seed: Random seed for reproducibility
            num_workers: Number of processes for parallel preprocessing
            num_quartets: Number of quartet queries to generate per tree
            is_quartet_classification: If True, return quartet queries and labels instead of species tree
        """
        self.max_sequence_length = max_sequence_length
        self.data: List[InputPair] = []
        self.tokenizer = NewickTokenizer()
        self.cached_encodings = []
        self.num_workers = min(num_workers, cpu_count())
        self.is_quartet_classification = is_quartet_classification
        self.num_quartets = num_quartets

        random.seed(seed)
        self._load_from_parquet(data_source, split, val_ratio)

        console.print(
            f"Pre-encoding trees for {split} split using {self.num_workers} processes..."
        )
        self._parallel_encode_trees()

    def _load_from_parquet(self, parquet_path: str, split: str, val_ratio: float):
        """Load data from parquet file with train/val split based on directories"""
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Get unique directories and check if we have enough for directory-based split
        all_dirs = sorted(df["directory"].unique())
        n_val = int(len(all_dirs) * val_ratio)

        if n_val >= 1:
            # Directory-based split if we have enough directories
            val_dirs = set(random.sample(all_dirs, n_val))

            # Filter based on split
            if split == "train":
                df = df[~df["directory"].isin(val_dirs)]
            else:  # val
                df = df[df["directory"].isin(val_dirs)]
        else:
            # Fallback to random row-based split if too few directories
            all_indices = list(range(len(df)))
            n_val_samples = int(len(df) * val_ratio)
            val_indices = set(random.sample(all_indices, n_val_samples))

            if split == "train":
                df = df[~df.index.isin(val_indices)]
            else:  # val
                df = df[df.index.isin(val_indices)]

        console.print(f"Loading {split} split with {len(df)} examples")

        for _, row in df.iterrows():
            stree = ts.read_tree_newick(row["species_tree"])
            stree.order("num_descendants_then_label")
            self.data.append(
                InputPair(gtrees=row["gtrees"], stree=stree.newick().lstrip("[&R] "))
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

    def _encode_single_item(self, pair: InputPair) -> dict:
        """Encode a single item for parallel processing"""
        tree_tensor, species_tokens = self.encode_trees(pair)
        
        if self.is_quartet_classification:
            quartet_queries, gt = pair.generate_queries_and_gt(self.num_quartets)
            return {
                'tree_tensor': tree_tensor,
                'quartet_queries': quartet_queries,
                'gt': gt
            }
        return {'tree_tensor': tree_tensor, 'species_tokens': species_tokens}

    def _parallel_encode_trees(self):
        """Parallel processing of tree encoding using chunks"""
        CHUNK_SIZE = 100
        with Pool(processes=self.num_workers) as pool:
            total = len(self.data)
            chunks = [
                self.data[i : i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)
            ]

            with console.status("[bold green]Processing trees...") as status:
                processed = 0
                for chunk_results in pool.imap(
                    self._encode_chunk,
                    chunks,
                    chunksize=1,  # Each "chunk" here is already a batch of items
                ):
                    self.cached_encodings.extend(chunk_results)
                    processed += len(chunk_results)
                    console.print(f"Processed {processed}/{total} trees")

    def _encode_chunk(
        self, pairs: List[InputPair]
    ) -> List[dict]:
        """Encode a chunk of items"""
        return [self._encode_single_item(pair) for pair in pairs]

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
        encoded_data = self.cached_encodings[idx]
        tree_tensor = encoded_data['tree_tensor']

        # Get actual number of trees
        num_gene_trees = min(tree_tensor.size(0), MAX_GTREES)

        # Create padded tensor directly with zeros
        padded_tree_tensor = torch.zeros(
            (MAX_GTREES, tree_tensor.size(1), tree_tensor.size(2)), dtype=torch.float32
        )

        # Copy actual tree encodings
        padded_tree_tensor[:num_gene_trees] = tree_tensor[:num_gene_trees]

        if self.is_quartet_classification:
            return {
                'gtrees': padded_tree_tensor,
                'quartet_queries': torch.from_numpy(encoded_data['quartet_queries']).float(),
                'Y': torch.from_numpy(encoded_data['gt']).long()
            }
        
        return padded_tree_tensor, encoded_data['species_tokens']


def parse_args():
    parser = argparse.ArgumentParser(description="Tree Dataset Loader")
    parser.add_argument(
        "data_path", type=str, help="Path to parquet file"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of directories to use for validation (for parquet files)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes for parallel preprocessing",
    )
    parser.add_argument(
        "--quartet-mode",
        action="store_true",
        help="Show example in quartet classification mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree Dataset Loader")
    parser.add_argument(
        "data_path", type=str, help="Path to parquet file"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of directories to use for validation (for parquet files)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes for parallel preprocessing",
    )
    parser.add_argument(
        "--quartet-mode",
        action="store_true",
        help="Show example in quartet classification mode",
    )
    args = parser.parse_args()

    # Create datasets
    train_dataset = TreeDataset(
        args.data_path,
        max_sequence_length=args.max_seq_length,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        is_quartet_classification=args.quartet_mode,
    )
    val_dataset = TreeDataset(
        args.data_path,
        max_sequence_length=args.max_seq_length,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        is_quartet_classification=args.quartet_mode,
    )

    console.print(f"[green]Dataset loaded successfully!")
    console.print(f"Train size: {len(train_dataset)}")
    console.print(f"Val size: {len(val_dataset)}")

    # Look at first few items
    console.print("\n[yellow]Sample items:[/yellow]")
    for i in range(min(3, len(train_dataset))):
        item = train_dataset[i]
        console.print(f"\n[cyan]Item {i}:[/cyan]")
        
        if args.quartet_mode:
            console.print(f"Gene trees tensor shape: {item['gtrees'].shape}")
            console.print(f"Quartet queries shape: {item['quartet_queries'].shape}")
            console.print(f"Ground truth labels shape: {item['Y'].shape}")
            console.print(f"First few quartet queries:\n{item['quartet_queries'][:3]}")
            console.print(f"First few labels: {item['Y'][:3]}")
        else:
            tree_tensor, species_tokens = item
            console.print(f"Tree tensor shape: {tree_tensor.shape}")
            console.print(f"Species tokens shape: {species_tokens.shape}")
            console.print(f"First few species tokens: {species_tokens[:10]}")
            console.print(f"Tree tensor: {tree_tensor[:, :10]}")
            decoded_stree = train_dataset.tokenizer.decode(species_tokens.tolist())
            console.print(f"Decoded species tree: {decoded_stree}")
