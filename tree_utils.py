import treeswift
import random
from typing import Union, Set, List, Tuple
import numpy as np
import tempfile
import os
import subprocess

def get_tree_from_input(tree_input: Union[str, treeswift.Tree]) -> treeswift.Tree:
    """Convert string input to TreeSwift tree if needed."""
    if isinstance(tree_input, str):
        return treeswift.read_tree_newick(tree_input)
    return tree_input

def get_bit_representations(tree: treeswift.Tree) -> Tuple[List[int], Set[int]]:
    """Generate random bit representations for taxa in the tree."""
    taxa = set()
    for node in tree.traverse_leaves():
        taxa.add(node.label)
    
    taxa = sorted(list(taxa))  # Ensure consistent ordering
    n = len(taxa)
    taxa_to_idx = {taxon: idx for idx, taxon in enumerate(taxa)}
    
    # Generate random 64-bit integers for each taxon
    transl = [random.getrandbits(64) for _ in range(n)]
    universe = 0
    for x in transl:
        universe ^= x
        
    return transl, taxa_to_idx, universe

def get_bipartitions(tree: treeswift.Tree, transl: List[int], 
                     taxa_to_idx: dict, universe: int) -> Set[int]:
    """Calculate bipartitions using XOR method."""
    bipartitions = set()
    bit_reprs = {}
    
    for node in tree.traverse_postorder():
        if node.is_leaf():
            bit_reprs[node] = transl[taxa_to_idx[node.label]]
        else:
            # Skip root bipartition
            if node.is_root():
                continue
                
            # Calculate clade representation using XOR of children
            clade = 0
            for child in node.child_nodes():
                clade ^= bit_reprs[child]
            
            bit_reprs[node] = clade
            # Store the smaller of clade and its complement
            bipartitions.add(min(clade, universe ^ clade))
            
    return bipartitions

def rf_distance(tree1: Union[str, treeswift.Tree], 
                tree2: Union[str, treeswift.Tree], 
                normalize: bool = False) -> dict:
    """
    Calculate RF distance between two trees with triple verification.
    
    Args:
        tree1: First tree (Newick string or TreeSwift Tree object)
        tree2: Second tree (Newick string or TreeSwift Tree object)
        normalize: If True, return normalized RF distance
        
    Returns:
        Dictionary containing RF distance metrics
    """
    # Convert inputs to TreeSwift trees if needed
    tree1 = get_tree_from_input(tree1)
    tree2 = get_tree_from_input(tree2)
    
    # Run the comparison three times with different random bits
    results = []
    for _ in range(3):
        transl, taxa_to_idx, universe = get_bit_representations(tree1)
        
        ref_bips = get_bipartitions(tree1, transl, taxa_to_idx, universe)
        est_bips = get_bipartitions(tree2, transl, taxa_to_idx, universe)
        
        shared_bips = len(ref_bips & est_bips)
        fn_bips = len(ref_bips) - shared_bips
        fp_bips = len(est_bips) - shared_bips
        
        results.append((fp_bips, fn_bips, len(ref_bips), len(est_bips)))
    
    # Verify all three runs agree
    if not all(r == results[0] for r in results):
        # If they don't agree, retry the whole process
        return rf_distance(tree1, tree2, normalize)
    
    fp_bips, fn_bips, ref_edges, est_edges = results[0]
    rf_dist = fp_bips + fn_bips
    
    # Calculate normalized RF distance if requested
    if normalize:
        max_rf = ref_edges + est_edges
        rf_normalized = rf_dist / max_rf if max_rf > 0 else 0.0
    else:
        rf_normalized = None
    
    return {
        'rf_distance': rf_dist,
        'rf_normalized': rf_normalized,
        'false_positives': fp_bips,
        'false_negatives': fn_bips,
        'reference_edges': ref_edges,
        'estimated_edges': est_edges,
    }

def neighbor_joining(D: np.ndarray) -> treeswift.Tree:
    """
    Implement the Neighbor-Joining algorithm to construct a phylogenetic tree from a distance matrix.
    """
    # Convert input to float64 to prevent overflow
    D = D.astype(np.float64)
    n = D.shape[0]
    if n != D.shape[1]:
        raise ValueError("Distance matrix must be square")
        
    # Initialize nodes with labels 0..n-1
    nodes = [treeswift.Node(label=str(i)) for i in range(n)]
    
    # Iteratively join clusters until only two remain
    while len(nodes) > 2:
        n = len(nodes)
        # Compute row sums (total distance) for each cluster
        r = np.sum(D, axis=1)
        
        # Compute Q-matrix
        Q = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                q_val = (n - 2) * D[i, j] - r[i] - r[j]
                Q[i, j] = q_val
                Q[j, i] = q_val
                
        # Find the pair (i,j) with minimum Q-value
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i  # ensure i < j for consistency
            
        # Compute branch lengths for nodes i and j
        dij = D[i, j]
        li = 0.5 * dij + (r[i] - r[j]) / (2 * (n - 2))
        lj = dij - li
        
        # Set branch lengths (preventing negatives)
        nodes[i].edge_length = max(0, li)
        nodes[j].edge_length = max(0, lj)
        
        # Create a new node as parent of nodes[i] and nodes[j]
        new_node = treeswift.Node()
        new_node.add_child(nodes[i])
        new_node.add_child(nodes[j])
        
        # Build new distance matrix
        remaining_indices = [k for k in range(n) if k not in (i, j)]
        m = len(remaining_indices)
        new_D = np.zeros((m + 1, m + 1))
        
        # Distances from the new node to each remaining node
        for idx, k in enumerate(remaining_indices):
            new_dist = 0.5 * (D[i, k] + D[j, k] - dij)
            new_D[0, idx + 1] = new_dist
            new_D[idx + 1, 0] = new_dist
            
        # Fill in the distances between the remaining nodes
        for ii, k in enumerate(remaining_indices):
            for jj, l in enumerate(remaining_indices):
                new_D[ii + 1, jj + 1] = D[k, l]
                
        nodes = [new_node] + [nodes[k] for k in remaining_indices]
        D = new_D
        
    # When only two nodes remain, join them under a root node
    if len(nodes) == 2:
        branch_length = D[0, 1] / 2
        nodes[0].edge_length = branch_length
        nodes[1].edge_length = branch_length
        root = treeswift.Node()
        root.add_child(nodes[0])
        root.add_child(nodes[1])
    else:
        root = nodes[0]
        
    t = treeswift.Tree()
    t.root = root
    return t

def set_default_branch_lengths(tree: treeswift.Tree, default: float = 1.0) -> None:
    """Assign a default branch length to every node if not already set."""
    for node in tree.traverse_postorder():
        if node.edge_length is None:
            node.edge_length = default

def get_bipartition_sets(tree: treeswift.Tree) -> set:
    """
    Compute the set of bipartitions for a tree.
    Each bipartition is represented as a frozenset of leaf labels corresponding to
    the smaller side of the split (by size, or lexicographically if equal).
    """
    # Get the full set of leaves in the tree.
    full_taxa = set(leaf.label for leaf in tree.traverse_leaves())
    biparts = set()
    for node in tree.traverse_postorder():
        # Skip leaves and the root (since the root's split is trivial)
        if node.is_leaf() or node.is_root():
            continue
        # Compute the set of leaves descending from this node.
        leaves = set(leaf.label for leaf in node.traverse_leaves())
        # Ignore trivial bipartitions (empty or full set)
        if len(leaves) == 0 or len(leaves) == len(full_taxa):
            continue
        complement = full_taxa - leaves
        # Choose a canonical representation: if sizes differ, choose the smaller;
        # if equal, choose based on lexicographic order.
        if len(leaves) > len(complement):
            smaller = complement
        elif len(leaves) < len(complement):
            smaller = leaves
        else:
            if sorted(leaves) <= sorted(complement):
                smaller = leaves
            else:
                smaller = complement
        if len(smaller) > 1:
            biparts.add(frozenset(smaller))
    return biparts

def rf_difference_details(tree1: Union[str, treeswift.Tree],
                          tree2: Union[str, treeswift.Tree]) -> dict:
    """
    Compute the RF distance between two trees and print the bipartitions that differ.

    Args:
        tree1: Reference tree (Newick string or TreeSwift Tree object)
        tree2: Estimated tree (Newick string or TreeSwift Tree object)

    Returns:
        A dictionary containing the RF distance, false negatives, false positives,
        and the full sets of bipartitions from each tree.
    """
    t1 = get_tree_from_input(tree1)
    t2 = get_tree_from_input(tree2)
    bip1 = get_bipartition_sets(t1)
    bip2 = get_bipartition_sets(t2)

    false_negatives = bip1 - bip2  # Bipartitions present in tree1 but missing from tree2
    false_positives = bip2 - bip1  # Bipartitions present in tree2 but missing from tree1
    rf_dist = len(false_negatives) + len(false_positives)

    print("Bipartitions only in reference tree (false negatives):")
    for b in false_negatives:
        print(sorted(b))

    print("Bipartitions only in estimated tree (false positives):")
    for b in false_positives:
        print(sorted(b))

    return {
        'rf_distance': rf_dist,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'reference_bipartitions': bip1,
        'estimated_bipartitions': bip2,
    }

def run_wastrid(trees: Union[str, treeswift.Tree, List[Union[str, treeswift.Tree]]]) -> str:
    """
    Run wastrid on a tree or list of trees to produce a summary tree.
    """
    # Convert input to list if single tree
    if not isinstance(trees, list):
        trees = [trees]
        
    # Convert all trees to Newick strings and clean them
    newick_trees = []
    for tree in trees:
        if isinstance(tree, treeswift.Tree):
            tree_str = tree.newick()
        else:
            tree_str = str(tree)
        # Clean up the tree string
        tree_str = tree_str.replace("[&R]", "").strip()
        newick_trees.append(tree_str)
            
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tre', delete=False) as tmp:
        # Write all trees to temp file
        for tree in newick_trees:
            tmp.write(tree + '\n')
        tmp_path = tmp.name
        
    try:
        # Run wastrid and capture output
        print(f"Running wastrid on input file: {tmp_path}")
        print("Input trees:")
        for tree in newick_trees:
            print(f"  {tree}")
            
        result = subprocess.run(
            ['wastrid', '--input', tmp_path, '--mode', 'internode'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("\nwastrid stdout:")
        print(result.stdout)
        print("\nwastrid stderr:")
        print(result.stderr)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, 
                ['wastrid'], 
                result.stdout, 
                result.stderr
            )
        
        # Return the summary tree (last line of output)
        return result.stdout.strip().split('\n')[-1]
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    tree1 = "(((1,14),(9,6)),((((15,(0,10)),(5,4)),12),((((7,11),13),(2,8)),3)));"
    tree2 = tree1
    print(rf_distance(tree1, tree2))
    
    # Sanity test: reconstruct tree1 from its distance matrix
    tree1_obj = get_tree_from_input(tree1)
    # Set default branch lengths so that the distance matrix becomes meaningful
    set_default_branch_lengths(tree1_obj, default=1.0)
    dist_matrix = tree1_obj.distance_matrix(leaf_labels=True)
    
    # Convert distance dict to numpy array
    labels = sorted(dist_matrix.keys(), key=lambda x: int(x))
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    D = np.zeros((n, n))
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if label1 == label2:
                D[i,j] = 0
            else:
                D[i,j] = dist_matrix[label1][label2]
    
    # Reconstruct using neighbor joining
    reconstructed_tree = neighbor_joining(D)
    
    # Compare original and reconstructed trees
    comparison = rf_distance(tree1_obj, reconstructed_tree)
    print("\nReconstruction Test:")
    print(f"Original tree: {tree1_obj.newick()}")
    print(f"Reconstructed tree: {reconstructed_tree.newick()}")
    print(f"RF distance: {comparison['rf_distance']}")
    if comparison['rf_distance'] > 0:
        print(rf_difference_details(tree1_obj, reconstructed_tree))