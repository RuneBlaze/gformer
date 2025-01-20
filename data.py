import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import treeswift as ts
from tqdm import tqdm

from tokenizer import NewickTokenizer

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
        n = len(taxa)
        
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

    binary_encoding = F.silu(binary_encoding)
    flat_encoding = binary_encoding[mask]

    return flat_encoding


class TreeDataset(Dataset):
    def __init__(self, jsonl_path: str, max_sequence_length: int = 1024):
        self.max_sequence_length = max_sequence_length
        self.data: List[InputPair] = []
        self.tokenizer = NewickTokenizer()
        self.cached_encodings = []

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(
                    InputPair(gtrees=item["gtrees"], species_tree=item["species_tree"])
                )
        
        print("Pre-encoding trees...")
        for pair in tqdm(self.data, desc="Encoding trees"):
            tree_tensor, species_tokens = self.encode_trees(pair)
            self.cached_encodings.append((tree_tensor, species_tokens))

    def __len__(self) -> int:
        return len(self.data)

    def encode_trees(self, pair: InputPair) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_trees = []
        for tree in pair.gtrees:
            distance_matrix = InputPair.newick_to_distance_matrix(tree)
            encoded = encode_distance_matrix(distance_matrix)
            encoded_trees.append(encoded)

        tree_tensor = torch.stack(encoded_trees, dim=0)
        species_tokens = torch.tensor(self.tokenizer.encode(pair.species_tree))
        species_tokens = torch.cat(
            [
                torch.tensor([self.tokenizer.EOI]),
                species_tokens,
                torch.tensor([self.tokenizer.EOS]),
            ]
        )

        return tree_tensor, species_tokens

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cached_encodings[idx]


if __name__ == "__main__":
    # Debug section
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python data.py <path_to_jsonl_file>")
        sys.exit(1)
        
    jsonl_path = sys.argv[1]
    
    # Create dataset
    dataset = TreeDataset(jsonl_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Look at first few items
    for i in range(min(3, len(dataset))):
        tree_tensor, species_tokens = dataset[i]
        print(f"\nItem {i}:")
        print(f"Tree tensor shape: {tree_tensor.shape}")
        print(f"Species tokens shape: {species_tokens.shape}")
        print(f"First few species tokens: {species_tokens[:10]}") 