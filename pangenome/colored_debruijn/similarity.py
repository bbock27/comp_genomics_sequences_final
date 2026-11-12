"""Pairwise genome similarity from a colored de Bruijn graph.

The graph (see build_debruijn_color) stores, for every k-mer of the
collection, the set of genomes that contain it. Two genomes are similar when
they share most of their k-mers. For genomes A and B, with K(A) the set of
distinct k-mers of A:

    similarity = |K(A) and K(B)| / |K(A) or K(B)|

the size of the intersection over the size of the union (the Jaccard index).
It is 1 for identical genomes and 0 when no k-mer is shared.

Method. One pass over the nodes groups them by color set, so each distinct
set of genomes is handled once however many k-mers carry it. A set with c
k-mers adds c to the count of every pair of genomes inside it.

Usage:
    python -m pangenome.colored_debruijn.similarity --collection data/synthetic_small/uneven_d0.01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pangenome.bwt.similarity import evaluate, output_dir, read_genomes
from simulator.genomes import genome_name

from .build_debruijn_color import build_graph

RESULTS = "results/debruijn_similarity"
DEFAULT_K = 31


def color_patterns(colors: dict) -> dict:
    """Count the k-mers carrying each distinct color set.

    Returns {genomes: count}, where genomes is a sorted tuple of genome
    numbers.
    """
    patterns = {}
    for members in colors.values():
        key = tuple(sorted(members))
        patterns[key] = patterns.get(key, 0) + 1
    return patterns


def shared_kmers(patterns: dict, d: int) -> np.ndarray:
    """Matrix S: S[a, b] = distinct k-mers in both genomes a and b;
    S[a, a] = distinct k-mers in genome a."""
    shared = np.zeros((d, d), dtype=np.int64)
    for members, count in patterns.items():
        index = np.array(members)
        shared[np.ix_(index, index)] += count
    return shared


def measures(shared: np.ndarray) -> dict:
    """Turn the matrix of shared k-mer counts into intersection, union, and
    similarity = intersection / union."""
    own = np.diag(shared)
    union = own[:, None] + own[None, :] - shared
    np.fill_diagonal(union, own)
    return {"shared": shared, "union": union, "similarity": shared / np.maximum(union, 1)}


def similarity_debruijn(genomes: list[str], k: int = DEFAULT_K) -> dict:
    """Shared k-mers, union, and similarity for every pair, with
    the size of the graph and how long each step took."""
    start = time.perf_counter()
    colors, edges = build_graph(genomes, k)
    built = time.perf_counter()
    patterns = color_patterns(colors)
    res = measures(shared_kmers(patterns, len(genomes)))
    compared = time.perf_counter()
    res["k"] = k
    res["nodes"] = len(colors)
    res["edges"] = sum(len(targets) for targets in edges.values())
    res["patterns"] = patterns
    res["build_seconds"] = built - start
    res["compare_seconds"] = compared - built
    return res


def write_results(out: Path, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d = len(res["shared"])
    names = [genome_name(i) for i in range(d)]
    with open(out / "pairs.tsv", "w") as fh:
        fh.write("genome_a\tgenome_b\tkmers_a\tkmers_b\tintersection\tunion\tsimilarity\n")
        for a in range(d):
            for b in range(a + 1, d):
                fh.write(f"{names[a]}\t{names[b]}\t{res['shared'][a, a]}\t{res['shared'][b, b]}\t{res['shared'][a, b]}\t"
                         f"{res['union'][a, b]}\t{res['similarity'][a, b]:.6g}\n")
    with open(out / "similarity_matrix.tsv", "w") as fh:
        fh.write("\t".join([""] + names) + "\n")
        for a in range(d):
            fh.write("\t".join([names[a]] + [f"{res['similarity'][a, b]:.6g}" for b in range(d)]) + "\n")


def print_summary(res: dict, genomes: list[str], collection: str, out: Path) -> None:
    d = len(genomes)
    iu = np.triu_indices(d, 1)
    print(f"{d} genomes, {sum(len(g) for g in genomes)} bases, k = {res['k']}: "
          f"{res['nodes']} nodes, {res['edges']} edges, {len(res['patterns'])} distinct color sets", file=sys.stderr)
    print(f"  building the graph {res['build_seconds']:.2f} s, comparing {res['compare_seconds']:.2f} s", file=sys.stderr)
    print(f"pairs: similarity from {res['similarity'][iu].min():.4f} to {res['similarity'][iu].max():.4f}",
          file=sys.stderr)
    # evaluate() ranks pairs by two distance-like keys (lower = closer); give
    # it 1 - similarity under both names
    apart = 1 - res["similarity"]
    ev = evaluate({"runs": res["shared"], "r_over_n": apart, "distance": apart}, Path(collection) / "pairwise.tsv")
    if ev:
        print(f"1 - similarity against the true distances ({ev['pairs']} pairs): Spearman {ev['spearman_distance']:.3f}, "
              f"nearest neighbour right for {ev['nearest_distance']}/{d} genomes", file=sys.stderr)
    print(f"wrote {out}/pairs.tsv, similarity_matrix.tsv", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pangenome.colored_debruijn.similarity", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True, help="directory with genomes.fasta")
    parser.add_argument("--num-genomes", type=int, help="use only the first N genomes")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"k-mer length (default {DEFAULT_K})")
    parser.add_argument("--out", help=f"output directory (default {RESULTS}/<collection>_nNNN_kKK)")
    args = parser.parse_args(argv)

    genomes = read_genomes(args.collection, args.num_genomes)
    default = output_dir(None, RESULTS, args.collection, len(genomes))
    out = Path(args.out) if args.out else default.with_name(f"{default.name}_k{args.k}")
    res = similarity_debruijn(genomes, args.k)
    write_results(out, res)
    print_summary(res, genomes, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
