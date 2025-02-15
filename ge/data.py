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
    def __init__(self, newick_str: str | tdl.Tree):
        """
        Initialize a SingleTree from a Newick string.

        Args:
            newick_str: Tree in Newick format
        """
        self.tree = tdl.Tree(newick_str) if isinstance(newick_str, str) else newick_str
        # Cache distance matrix
        self._distance_matrix = None

    def distance_matrix(self) -> torch.Tensor:
        """
        Return the distance matrix of the tree as a torch tensor.
        Values are converted from {0,1} to {-1,1} for better training.
        """
        dm: tdl.DistanceMatrix = self.tree.get_distance_matrix()
        # Convert to numpy array
        n = dm.ntaxa
        triu_entries = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = int(dm[str(i), str(j)])
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
        d = self.tree.get_distance_matrix()

        # Calculate the three sums corresponding to the three possible topologies
        ij_kl = d[str(i), str(j)] + d[str(k), str(l)]
        ik_jl = d[str(i), str(k)] + d[str(j), str(l)]
        il_jk = d[str(i), str(l)] + d[str(j), str(k)]

        # The smallest sum corresponds to the correct topology
        sums = [ij_kl, ik_jl, il_jk]
        return sums.index(min(sums))


class GeneTreeDataPoint(TypedDict):
    distance_matrix: torch.Tensor  # shape [num_pairs, 8], dtype=torch.bool
    quartet_queries: torch.Tensor  # shape [num_queries, 4], dtype=torch.int32
    quartet_topologies: torch.Tensor  # shape [num_queries], dtype=torch.int32


class GeneTreeDataset:
    def __init__(
        self,
        pkl_path: str,
        m: int,
        seed: int,
    ) -> None:
        self.pkl_path = pkl_path
        self._family = None  # Initialize as None for lazy loading
        self.m = m
        self.seed = seed

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

    def __getitem__(self, idx: int) -> GeneTreeDataPoint:
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

        # Select a single random tree
        tree_idx = rng.randrange(len(trees))
        tree = trees[tree_idx]

        # Restrict tree to subset of taxa and remap names
        tree_restricted = tree.restriction(subset_names).remap(mapper)

        # Convert to SingleTree object for consistent interface
        single_tree = SingleTree(tree_restricted)

        # Generate distance matrix
        distance_matrix = single_tree.distance_matrix()

        # Generate random quartets using rejection sampling
        selected_quartets = self._sample_unique_quartets(rng, 100)

        # Convert quartets to tensor
        quartet_queries = torch.tensor(selected_quartets, dtype=torch.int32)

        # Get quartet topologies
        topologies = [
            single_tree.classify_quartet(quartet) for quartet in selected_quartets
        ]
        quartet_topologies = torch.tensor(topologies, dtype=torch.int32)

        return GeneTreeDataPoint(
            distance_matrix=distance_matrix,
            quartet_queries=quartet_queries,
            quartet_topologies=quartet_topologies,
        )


if __name__ == "__main__":
    WHERE = "/Users/lbq/goof/teedeelee/assets/processed_family.pkl"

    dataset = GeneTreeDataset(WHERE, 16, 42)
    data = dataset[0]
    print(data)
