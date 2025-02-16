from dataclasses import dataclass
from functools import reduce
from operator import xor
from random import getrandbits
from typing import Dict, Set, Tuple
from collections.abc import Iterator

import treeswift as ts


@dataclass
class RFMetrics:
    """Metrics for Robinson-Foulds distance between two trees"""

    normalized_false_negatives: (
        float  # proportion of reference bipartitions missing from estimated tree
    )
    normalized_false_positives: (
        float  # proportion of estimated bipartitions missing from reference tree
    )
    normalized_rf_distance: float  # overall normalized RF distance


def xor_clades(t: ts.Tree, transl: Dict[str, int]) -> Iterator[int]:
    calculated_fake_root = False
    for n in t.traverse_postorder():
        if n.is_leaf():
            n.bip = transl[n.label]
        else:
            if n.is_root():
                continue
            else:
                if n.parent.is_root() and n.parent.num_children() == 2:
                    if calculated_fake_root or any(
                        c.is_leaf() for c in n.parent.children
                    ):
                        continue
                    else:
                        calculated_fake_root = True
                n.bip = reduce(xor, (c.bip for c in n.children))
                yield n.bip


def fast_rf(lhs: str | ts.Tree, rhs: str | ts.Tree) -> RFMetrics:
    # lhs is reference tree
    # rhs is estimated tree
    
    # Convert strings to Tree objects if needed
    if isinstance(lhs, str):
        lhs = ts.read_tree_newick(lhs)
    if isinstance(rhs, str):
        rhs = ts.read_tree_newick(rhs)
        
    transl = {l.label: getrandbits(64) for l in lhs.traverse_leaves()}
    n = len(transl)
    universe = reduce(xor, transl.values(), 0)
    l_bips: Set[int] = set(min(c, universe ^ c) for c in xor_clades(lhs, transl))
    r_bips: Set[int] = set(min(c, universe ^ c) for c in xor_clades(rhs, transl))
    fn_bips = l_bips - r_bips
    fp_bips = r_bips - l_bips
    return RFMetrics(
        normalized_false_negatives=len(fn_bips) / len(l_bips),
        normalized_false_positives=len(fp_bips) / len(r_bips),
        normalized_rf_distance=(len(fn_bips) + len(fp_bips))
        / (len(l_bips) + len(r_bips)),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate RF distance between two trees"
    )
    parser.add_argument(
        "-l", "--lhs", help="Left hand side tree (reference)", required=True
    )
    parser.add_argument(
        "-r", "--rhs", help="Right hand side tree (estimated)", required=True
    )
    args = parser.parse_args()
    lhs = ts.read_tree_newick(args.lhs)
    rhs = ts.read_tree_newick(args.rhs)
    metrics = fast_rf(lhs, rhs)
    print(
        f"nFN {metrics.normalized_false_negatives}\nnFP {metrics.normalized_false_positives}\nnRF {metrics.normalized_rf_distance}"
    )
