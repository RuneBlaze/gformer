from typing import Protocol
import torch
import treeswift as ts
import numpy as np
from smallperm import PseudoRandomPermutation as PRP
from teedeelee import SortBy
from itertools import islice
from typing import TypedDict
from random import Random
import pickle as pkl
from pathlib import Path

import teedeelee as tdl


TOPOLOGY_AB_CD = 0
TOPOLOGY_AC_BD = 1
TOPOLOGY_AD_BC = 2


class SingleGeneTreeLike(Protocol):
    def classify_quartet(self, indices: tuple[int, int, int, int]) -> int:
        """
        Given a quartet of species indices, return the index of the quartet topology.
        """

        ...

    def distance_matrix(self) -> torch.Tensor:
        """
        Return the distance matrix of the tree.
        """

        ...


class SingleTree(SingleGeneTreeLike):
    def __init__(self, newick_str: str | ts.Tree):
        """
        Initialize a SingleTree from a Newick string.

        Args:
            newick_str: Tree in Newick format
        """
        self.tree = ts.read_tree_newick(newick_str) if isinstance(newick_str, str) else newick_str
        # Cache distance matrix
        self._distance_matrix = None

    def distance_matrix(self) -> torch.Tensor:
        """
        Return the distance matrix of the tree as a torch tensor.
        Values are converted from {0,1} to {-1,1} for better training.
        """
        dm = self.tree.distance_matrix(leaf_labels=True)
        # Convert to numpy array
        n = len(list(n.label for n in self.tree.traverse_leaves()))
        triu_entries = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = int(dm[str(i)][str(j)])
                # Convert to 8-bit binary representation
                binary = [(distance >> bit) & 1 for bit in range(8)]
                triu_entries.append(binary)
        # Convert to torch tensor and map 0 to -1
        tensor = torch.tensor(triu_entries, dtype=torch.float32)
        return torch.where(tensor == 0, torch.tensor(-1.0), tensor)

    def classify_quartet(self, indices: tuple[int, int, int, int]) -> int:
        """
        Given a quartet of species indices (i,j,k,l), return:
        0 if ij|kl topology
        1 if ik|jl topology
        2 if il|jk topology

        Uses the four-point method on the distance matrix to determine the topology.
        """
        i, j, k, l = indices
        d = self.tree.distance_matrix(leaf_labels=True)

        # Calculate the three sums corresponding to the three possible topologies
        ij_kl = d[str(i)][str(j)] + d[str(k)][str(l)]
        ik_jl = d[str(i)][str(k)] + d[str(j)][str(l)]
        il_jk = d[str(i)][str(l)] + d[str(j)][str(k)]

        # The smallest sum corresponds to the correct topology
        sums = [ij_kl, ik_jl, il_jk]
        return sums.index(min(sums))


class GeneTreeDataPoint(TypedDict):
    distance_matrix: torch.Tensor  # shape [num_pairs, 8], dtype=torch.bool
    quartet_queries: torch.Tensor  # shape [num_queries, 4], dtype=torch.int32
    quartet_topologies: torch.Tensor  # shape [num_queries], dtype=torch.int32


def only_topology(
    tree: ts.Tree,
) -> ts.Tree:
    """Remove branch lengths and internal node labels, keeping only topology."""
    for node in tree.traverse_postorder():
        node.edge_length = 1
        if node.num_children() > 0:
            node.label = None
    return tree

def strip_topology(
    tree: ts.Tree,
) -> ts.Tree:
    """Remove branch lengths and internal node labels, keeping only topology."""
    for node in tree.traverse_postorder():
        node.edge_length = None
        if node.num_children() > 0:
            node.label = None
    return tree

def unroot(tree):
    """
    Unroots treeswift tree. Adapted from treeswift 'deroot' function.
    This one doesn't contract (A,B); to A;

    Parameters
    ----------
    tree: treeswift tree

    Returns unrooted treeswift tree
    """
    if tree.root == None:
        return tree
    if tree.root.num_children() == 2:
        [left, right] = tree.root.child_nodes()
        if not right.is_leaf():
            right.contract()
        elif not left.is_leaf():
            left.contract()
    tree.is_rooted = False
    return tree

class GeneTreeDataset:
    def __init__(
        self,
        pkl_path: str,
        m: int,
        seed: int,
        reference_path: str = "/Users/lbq/Downloads/S101/01/truegenetrees",
    ) -> None:
        self.pkl_path = pkl_path
        self._family = None  # Initialize as None for lazy loading
        self._reference_trees = None  # Initialize reference trees as None
        self.m = m
        self.seed = seed
        self.reference_path = reference_path

    @property
    def family(self):
        """Lazily load the family data from pickle when first accessed"""
        if self._family is None:
            with open(self.pkl_path, "rb") as f:
                self._family = pkl.load(f)
        return self._family

    @property
    def nfamily(self):
        """Get number of families, loading data if needed"""
        return len(self.family)

    @property
    def reference_trees(self):
        """Lazily load the reference trees when first accessed"""
        if self._reference_trees is None:
            with open(self.reference_path, "r") as f:
                self._reference_trees = [ts.read_tree_newick(line.strip()) for line in f]
        return self._reference_trees

    def __len__(self) -> int:
        return 2**32

    def _sample_quartet(self, rng: Random) -> tuple[int, int, int, int]:
        """Sample a random quartet of distinct integers from range(self.m)"""
        indices = []
        while len(indices) < 4:
            i = rng.randrange(self.m)
            if i not in indices:
                indices.append(i)
        return tuple(sorted(indices))

    def _sample_unique_quartets(
        self, rng: Random, n: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Sample n unique quartets using rejection sampling.
        Returns list of tuples, each containing 4 sorted integers.
        """
        seen = set()
        quartets = []
        max_attempts = n * 10  # Prevent infinite loop
        attempts = 0

        while len(quartets) < n and attempts < max_attempts:
            quartet = self._sample_quartet(rng)
            if quartet not in seen:
                seen.add(quartet)
                quartets.append(quartet)
            attempts += 1

        if len(quartets) < n:
            raise RuntimeError(
                f"Could only generate {len(quartets)} unique quartets after {max_attempts} attempts"
            )

        return quartets

    def _greedy_quartet_cover(self, rng: Random, n: int, m: int) -> list[tuple[int, int, int, int]]:
        """
        Sample n quartets by deterministically choosing least represented species,
        avoiding repeated quartets.
        
        Args:
            rng: Random number generator (unused)
            n: Number of quartets to sample
            m: Number of species
        
        Returns:
            List of quartets (tuples of 4 integers)
        """
        quartets = []
        species_counts = {i: 0 for i in range(m)}
        used_quartets = set()
        
        def get_next_quartet(available_species):
            # Try all possible combinations of the first 8 least-used species
            # to find an unused quartet
            candidates = available_species[:min(8, len(available_species))]
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    for k in range(j + 1, len(candidates)):
                        for l in range(k + 1, len(candidates)):
                            quartet = (candidates[i], candidates[j], candidates[k], candidates[l])
                            if quartet not in used_quartets:
                                return quartet
            return None
        
        while len(quartets) < n:
            # Sort species by (count, species_index)
            species_by_count = sorted(
                range(m),
                key=lambda x: (species_counts[x], x)
            )
            
            quartet = get_next_quartet(species_by_count)
            if quartet is None:
                # If we can't find any unused quartets, reset the counts
                species_counts = {i: 0 for i in range(m)}
                continue
            
            quartets.append(quartet)
            used_quartets.add(quartet)
            
            # Update counts
            for species in quartet:
                species_counts[species] += 1
        
        return quartets

    def __getitem__(self, idx: int) -> GeneTreeDataPoint:
        # Initialize random state
        rng = Random((idx ^ self.seed) % 2**32)

        # Randomly select a tree from reference_trees
        tree = self.reference_trees[rng.randrange(len(self.reference_trees))]
        
        # Apply only_topology before any operations
        tree = only_topology(tree)

        leaves = list(tree.traverse_leaves())
        leaves = [str(i) for i in range(len(leaves))]

        # Get all leaf names and select random subset
        selected_indices = rng.sample(range(len(leaves)), self.m)
        subset_names = [str(leaves[i]) for i in selected_indices]

        # Extract tree with subset of taxa
        pruned_tree = tree.extract_tree_with(subset_names)

        old_name2new_name = {old_name: str(i) for i, old_name in enumerate(subset_names)}
        pruned_tree.rename_nodes(old_name2new_name)
        
        # Unroot the tree before applying only_topology
        pruned_tree = unroot(pruned_tree)
        only_topology(pruned_tree)

        # Get distance matrix directly from TreeSwift tree
        dm = pruned_tree.distance_matrix(leaf_labels=True)
        n = len(subset_names)
        triu_entries = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = int(dm[str(i)][str(j)])
                # Convert to 8-bit binary representation
                binary = [(distance >> bit) & 1 for bit in range(8)]
                triu_entries.append(binary)
        # Convert to torch tensor and map 0 to -1
        distance_matrix = torch.tensor(triu_entries, dtype=torch.float32)
        distance_matrix = torch.where(distance_matrix == 0, torch.tensor(-1.0), distance_matrix)

        NUM_QUARTETS = 8

        # Replace the old quartet sampling with greedy cover sampling
        selected_quartets = self._greedy_quartet_cover(Random(0), int(NUM_QUARTETS * 1.5), self.m)
        # Get quartet topologies using four-point method
        topologies = []
        valid_quartets = []
        for quartet in selected_quartets:
            
            i, j, k, l = quartet
            # Get pairwise distances from the distance matrix we already computed
            ij_kl = dm[str(i)][str(j)] + dm[str(k)][str(l)]
            ik_jl = dm[str(i)][str(k)] + dm[str(j)][str(l)]
            il_jk = dm[str(i)][str(l)] + dm[str(j)][str(k)]
            
            # Check if quartet is resolved (all sums must be different)
            sums = [ij_kl, ik_jl, il_jk]
            if len(set(sums)) > 1:  # All sums are different
                topologies.append(sums.index(min(sums)))
                valid_quartets.append(quartet)
                if len(topologies) >= NUM_QUARTETS:  # We have enough resolved quartets
                    break

        # If we don't have enough resolved quartets, pad with the last valid quartet
        while len(topologies) < NUM_QUARTETS:
            topologies.append(topologies[-1])
            valid_quartets.append(valid_quartets[-1])

        quartet_queries = torch.tensor(valid_quartets, dtype=torch.int32)
        quartet_topologies = torch.tensor(topologies, dtype=torch.int32)

        return GeneTreeDataPoint(
            distance_matrix=distance_matrix,
            quartet_queries=quartet_queries,
            quartet_topologies=quartet_topologies,
        )


if __name__ == "__main__":
    WHERE = "/Users/lbq/goof/teedeelee/assets/processed_family.pkl"
    REFERENCE_PATH = "/Users/lbq/Downloads/S101/01/truegenetrees"
    
    dataset = GeneTreeDataset(WHERE, 16, 42, REFERENCE_PATH)
    data = dataset[1]
    print(data)