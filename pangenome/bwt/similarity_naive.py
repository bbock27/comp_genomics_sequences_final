"""Pairwise genome similarity from BWT run counts: the naive version.

Same measures and output files as pangenome.bwt.similarity, computed the direct
way: for every pair of genomes, build the BWT of just those two and count its
runs. With d genomes that is d single-genome BWTs plus d(d-1)/2 two-genome
BWTs, so the cost grows with the square of the number of genomes, where the
other version builds one BWT of the whole collection.

It exists as a baseline for timing and as an independent check: both versions
must produce identical run counts.

Usage:
    python -m pangenome.bwt.similarity_naive --collection data/synthetic_small/uneven_d0.01
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .build import build_bwt, count_runs
from .similarity import measures, output_dir, parse_args, print_summary, read_genomes, write_results


def similarity_naive(genomes: list[str], progress: bool = False) -> dict:
    """Run counts, lengths, r/n, and normalised distance for every pair."""
    d = len(genomes)
    runs = np.zeros((d, d), dtype=np.int64)
    for a in range(d):
        runs[a, a] = count_runs(build_bwt([genomes[a]])[0])
    done = 0
    for a in range(d):
        for b in range(a + 1, d):
            runs[a, b] = runs[b, a] = count_runs(build_bwt([genomes[a], genomes[b]])[0])
            done += 1
            if progress and done % 10 == 0:
                print(f"  {done}/{d * (d - 1) // 2} pairs", file=sys.stderr, flush=True)
    return measures(runs, genomes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args("python -m pangenome.bwt.similarity_naive", __doc__, "results/bwt_similarity_naive", argv)
    genomes = read_genomes(args.collection, args.num_genomes)
    out = output_dir(args.out, "results/bwt_similarity_naive", args.collection, len(genomes))

    start = time.time()
    res = similarity_naive(genomes, progress=True)
    elapsed = time.time() - start

    write_results(out, res)
    print_summary(res, genomes, elapsed, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
