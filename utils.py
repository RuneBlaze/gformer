import random
from pathlib import Path
from typing import Dict, List, Set
import multiprocessing as mp

import pandas as pd
import treeswift as ts
import typer
from smallperm import PseudoRandomPermutation as PRP
from tqdm import tqdm

from data import InputPair

# One-time pad constant for seed modification
OTP: int = 1000000007


def random_projection(
    gene_trees: List[str],
    species_tree: str,
    target_dimension: int,
    num_gene_trees: int,
    seed: int,
) -> InputPair:
    """
    Performs random projection on phylogenetic trees by selecting a subset of taxa.

    Args:
        gene_trees: List of Newick strings representing gene trees
        species_tree: Newick string representing the species tree
        target_dimension: Number of taxa to keep in the projection
        num_gene_trees: Number of gene trees to sample
        seed: Random seed for reproducibility

    Returns:
        InputPair containing projected gene trees and species tree
    """
    # Sample subset of gene trees
    subsetting_prp = PRP(len(gene_trees), seed ^ OTP)
    sampled_trees = [gene_trees[i] for i in subsetting_prp[:num_gene_trees]]

    # Parse Newick strings into TreeSwift objects
    gene_tree_objects = [ts.read_tree_newick(tree) for tree in sampled_trees]
    species_tree_object = ts.read_tree_newick(species_tree)

    # Collect all taxa from species tree
    taxa: Set[str] = {leaf.label for leaf in species_tree_object.traverse_leaves()}

    # Select random subset of taxa
    taxa_sorted = sorted(list(taxa))
    prp = PRP(len(taxa), seed)
    selected_taxa = [taxa_sorted[i] for i in prp[:target_dimension]]
    taxa_to_index = {taxon: i for i, taxon in enumerate(selected_taxa)}

    # Extract subtrees with selected taxa
    projected_gene_trees = []
    for tree in gene_tree_objects:
        subtree = tree.extract_tree_with(selected_taxa)
        subtree.rename_nodes(taxa_to_index)
        subtree = only_topology(subtree)
        subtree = unroot(subtree)
        projected_gene_trees.append(subtree)

    projected_species_tree = species_tree_object.extract_tree_with(selected_taxa)
    projected_species_tree.rename_nodes(taxa_to_index)

    projected_species_tree = only_topology(projected_species_tree)
    return InputPair(projected_gene_trees, projected_species_tree)


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


def only_topology(
    tree: ts.Tree,
) -> ts.Tree:
    for node in tree.traverse_postorder():
        node.edge_length = None
        if node.num_children() > 0:
            node.label = None
    return tree


def process_directory(
    input_dir: Path,
    target_dimension: int,
    num_gene_trees: int,
    seed: int,
) -> Dict:
    """
    Process a single directory and return results as a dictionary.
    """
    species_tree_path = input_dir / "s_tree.trees"
    gene_trees_path = input_dir / "truegenetrees"

    # Read input files
    with open(species_tree_path) as f:
        species_tree = f.read().strip()

    with open(gene_trees_path) as f:
        gene_trees = [line.strip() for line in f if line.strip()]

    # Perform random projection
    projected = random_projection(
        gene_trees=gene_trees,
        species_tree=species_tree,
        target_dimension=target_dimension,
        num_gene_trees=num_gene_trees,
        seed=seed,
    )

    return {
        "gtrees": [tree.newick() for tree in projected.gtrees],
        "species_tree": projected.stree.newick().lstrip("[&R] "),
        "directory": str(input_dir),
        "seed": seed,
        "num_trees": num_gene_trees,
        "target_dim": target_dimension,
    }


def process_directory_wrapper(args):
    """
    Standalone wrapper function for multiprocessing.
    Args should be (dir_num, base_dir, target_dimension, num_iterations)
    """
    dir_num, base_dir, target_dimension, num_iterations = args
    results = []
    input_dir = base_dir / f"{dir_num:02d}"
    if not input_dir.exists():
        print(f"Skipping non-existent directory: {input_dir}")
        return results

    for i in tqdm(range(num_iterations), desc=f"Directory {dir_num:02d}", leave=False):
        seed = i
        num_gene_trees = random.randint(100, 200)
        result = process_directory(
            input_dir=input_dir,
            target_dimension=target_dimension,
            num_gene_trees=num_gene_trees,
            seed=seed,
        )
        results.append(result)
    return results


def main(
    base_dir: Path = typer.Argument(
        ..., help="Base directory containing numbered subdirectories"
    ),
    target_dimension: int = typer.Option(
        16, "--target-dim", "-d", help="Number of taxa to keep in projection"
    ),
    num_iterations: int = typer.Option(
        1000, "--iterations", "-i", help="Number of iterations per directory"
    ),
    output_dir: Path = typer.Option(
        "projected_data", "--output", "-o", help="Output directory for parquet files"
    ),
    start_dir: int = typer.Option(1, "--start", help="Starting directory number"),
    end_dir: int = typer.Option(10, "--end", help="Ending directory number"),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        help="Number of iterations to process before writing to disk",
    ),
    num_processes: int = typer.Option(
        2, "--processes", "-p", help="Number of processes to use"
    ),
) -> None:
    """
    Process multiple directories and save results to sharded parquet files.
    Expects directories in format: {base_dir}/01, {base_dir}/02, etc.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create schema for the table
    schema = pa.schema(
        [
            ("gtrees", pa.list_(pa.string())),
            ("species_tree", pa.string()),
            ("directory", pa.string()),
            ("seed", pa.int64()),
            ("num_trees", pa.int64()),
            ("target_dim", pa.int64()),
        ]
    )

    # Create pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Prepare arguments for each directory
        dir_args = [
            (dir_num, base_dir, target_dimension, num_iterations)
            for dir_num in range(start_dir, end_dir + 1)
        ]

        # Process directories in parallel
        all_results = []
        shard_counter = 0

        for batch_results in tqdm(
            pool.imap_unordered(process_directory_wrapper, dir_args),
            total=len(dir_args),
            desc="Processing directories",
        ):
            all_results.extend(batch_results)

            # Write in chunks
            while len(all_results) >= chunk_size:
                chunk = all_results[:chunk_size]
                all_results = all_results[chunk_size:]

                # Write chunk to new parquet file
                df_chunk = pd.DataFrame(chunk)
                table = pa.Table.from_pandas(df_chunk, schema=schema)
                output_file = output_dir / f"shard_{shard_counter:04d}.parquet"
                pq.write_table(table, output_file)
                shard_counter += 1

    # Write any remaining results
    if all_results:
        df_chunk = pd.DataFrame(all_results)
        table = pa.Table.from_pandas(df_chunk, schema=schema)
        output_file = output_dir / f"shard_{shard_counter:04d}.parquet"
        pq.write_table(table, output_file)

    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    typer.run(main)
