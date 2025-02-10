from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import treeswift as ts
from rich.console import Console
from torch.utils.data import Dataset

from constants import MAX_GTREES, MAX_TAXA, PAD
from tokenizer import NewickTokenizer

import pickle as pkl
from itertools import islice
import treeswift as ts
from smallperm import PseudoRandomPermutation as PRP
import logging
from typing import Any
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass
from teedeelee import DistanceMatrix, SortBy
import numpy as np
from random import Random

console = Console()

@dataclass
class MSCDataPoint:
    distance_matrices: list[DistanceMatrix]
    species_tree: str

    def distance_matrices_numpy(self) -> np.ndarray:
        return np.array([self.to_numpy(mat) for mat in self.distance_matrices])
    
    @staticmethod
    def to_numpy(mat: DistanceMatrix) -> np.ndarray:
        ts_names = set(mat.taxon_set.names())
        for t in range(mat.ntaxa):
            if mat.get_taxon_name(t) not in ts_names:
                raise ValueError(f"Taxon {t} not in taxon set")
        # Get upper triangular entries (excluding diagonal) in row-major order
        triu_entries = []
        for i in range(mat.ntaxa):
            for j in range(i + 1, mat.ntaxa):
                distance = int(mat[str(i), str(j)])
                # Convert to 8-bit binary representation
                binary = [(distance >> bit) & 1 for bit in range(8)]
                triu_entries.append(binary)
        return np.array(triu_entries, dtype=np.bool_)

class MSCDataset:
    def __init__(
        self,
        pkl_path: str,
        k_min_max: tuple[int, int],
        m: int,
        seed: int,
    ) -> None:
        with open(pkl_path, 'rb') as f:
            self.family = pkl.load(f)
        self.nfamily = len(self.family)
        self.k_min, self.k_max = k_min_max
        self.m = m
        self.seed = seed
    
    def __len__(self) -> int:
        return 2 ** 32
    
    def __getitem__(self, idx: int) -> MSCDataPoint:
        # Initialize random state
        rng = Random((idx ^ self.seed) % 2**32)
        
        # Select which family and get trees
        which_family = idx % self.nfamily
        trees = self.family[which_family]
        
        # Get sorted taxon names and select subset
        sorted_names = sorted(trees.taxon_set.names())
        prp_names = PRP(len(sorted_names), rng.randint(0, 2**32))
        subset_names = [sorted_names[i] for i in islice(prp_names, self.m)]
        
        # Create name mapping
        mapper = {name: str(i) for i, name in enumerate(subset_names)}
        
        # Select random subset of trees
        k = rng.randint(self.k_min, self.k_max)
        prp = PRP(len(trees), rng.randint(0, 2**32))
        subindices = list(islice(prp, k))
        subset_trees = [trees[i] for i in subindices]
        
        # Restrict trees to subset of taxa and remap names
        subset_trees_restricted = [tree.restriction(subset_names).remap(mapper) for tree in subset_trees]
        
        # Process species tree
        species_tree = trees.get_species_tree()
        species_tree_restricted = species_tree.restriction(subset_names).remap(mapper).sort_by_multiple(
            [(SortBy.DescendantCount, True), (SortBy.LexicographicalOrder, True)]
        )
        
        # Generate distance matrices and return
        distance_matrices = [tree.get_distance_matrix() for tree in subset_trees_restricted]
        return MSCDataPoint(distance_matrices, species_tree_restricted.newick())



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
                    dist_matrix[i, j] = int(dist_dict[u][v])
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
    def __init__(
        self,
        data_source: str,
        seed: int = 42,
    ):
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
        self._data = None
        self.tokenizer = NewickTokenizer()
        self.data_source = data_source
        self.seed = seed
    def __len__(self) -> int:
        return 2 ** 32

    # def encode_trees(self, pair: InputPair) -> Tuple[torch.Tensor, torch.Tensor]:
    #     encoded_trees = []
    #     for tree in pair.gtrees:
    #         distance_matrix = InputPair.newick_to_distance_matrix(tree)
    #         encoded = encode_distance_matrix(distance_matrix)
    #         encoded_trees.append(encoded)

    #     tree_tensor = torch.stack(encoded_trees, dim=0)
    #     species_tokens = torch.tensor(self.tokenizer.encode(pair.stree))
    #     species_tokens = torch.cat(
    #         [
    #             species_tokens,
    #             torch.tensor([self.tokenizer.EOS]),
    #         ]
    #     )

    def _load_data(self) -> MSCDataset:
        return MSCDataset(self.data_source, k_min_max=(230, 299), m=16, seed=self.seed)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._data is None:
            self._data = self._load_data()
        unpacked = self._data[idx]

        tree_tensor = torch.from_numpy(unpacked.distance_matrices_numpy()).to(torch.float32)
        species_tokens = torch.tensor(self.tokenizer.encode(unpacked.species_tree))
        species_tokens = torch.cat(
            [
                species_tokens,
            ]
        )

        # Get actual number of trees
        num_gene_trees = min(tree_tensor.size(0), MAX_GTREES)

        # Create padded tensor directly with zeros
        padded_tree_tensor = torch.zeros(
            (MAX_GTREES, tree_tensor.size(1), tree_tensor.size(2)), dtype=torch.float32
        )

        # Copy actual tree encodings
        padded_tree_tensor[:num_gene_trees] = tree_tensor[:num_gene_trees]

        return padded_tree_tensor, species_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Tree Dataset Loader")
    parser.add_argument(
        "data_path", type=str, help="Path to data file (.jsonl or .parquet)"
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
    return parser.parse_args()


if __name__ == "__main__":
    dataset = TreeDataset(
        "/Users/lbq/goof/teedeelee/assets/processed_family.pkl"
    )

    for i in range(1694, 10000):
        try:
            tree_tensor, species_tokens = dataset[i]
        except Exception as e:
            print(f"Error at item {i}: {e}")
            raise e
        console.print(f"\n[cyan]Item {i}:[/cyan]")
        console.print(f"Tree tensor shape: {tree_tensor.shape}")
        console.print(f"Species tokens shape: {species_tokens.shape}")
        # console.print(f"First few species tokens: {species_tokens[:10]}")
        # console.print(f"Tree tensor: {tree_tensor[:, :10]}")
        # # Add decoded species tree output
        decoded_stree = dataset.tokenizer.decode(species_tokens.tolist())
        console.print(f"Decoded species tree: {decoded_stree}")
