"""Pairwise genome similarity from BWT run counts: naive version, in parallel.

Same measures and output files as pangenome.bwt.similarity_naive. Every pair
of genomes still gets its own BWT, but the pairs are independent of each
other, so a pool of worker processes builds several at once.

Each worker holds the genome text plus the working arrays for one pair, so
memory grows with the number of workers.

Usage:
    python -m pangenome.bwt.similarity_naive_parallel --collection data/synthetic_small/uneven_d0.01 --workers 8
"""

from __future__ import annotations

import multiprocessing
import sys
import time

import numpy as np

from .build import build_bwt, count_runs
from .similarity import measures, output_dir, parse_args, print_summary, read_genomes, write_results

_GENOMES: list[str] = []  # set once in each worker process


def _start_worker(genomes: list[str]) -> None:
    global _GENOMES
    _GENOMES = genomes


def _count_pair(pair: tuple[int, int]) -> tuple[int, int, int]:
    """Runs in the BWT of genomes a and b (or of genome a alone when a == b)."""
    a, b = pair
    texts = [_GENOMES[a]] if a == b else [_GENOMES[a], _GENOMES[b]]
    return a, b, count_runs(build_bwt(texts)[0])


def similarity_naive_parallel(genomes: list[str], workers: int, progress: bool = False) -> dict:
    """Run counts, lengths, r/n, and normalised distance for every pair."""
    d = len(genomes)
    tasks = [(a, b) for a in range(d) for b in range(a, d)]
    runs = np.zeros((d, d), dtype=np.int64)
    workers = max(1, min(workers, len(tasks)))
    with multiprocessing.Pool(workers, initializer=_start_worker, initargs=(genomes,)) as pool:
        done = 0
        for a, b, r in pool.imap_unordered(_count_pair, tasks):
            runs[a, b] = runs[b, a] = r
            done += 1
            if progress and done % 20 == 0:
                print(f"  {done}/{len(tasks)} BWTs", file=sys.stderr, flush=True)
    return measures(runs, genomes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args("python -m pangenome.bwt.similarity_naive_parallel", __doc__,
                      "results/bwt_similarity_naive_parallel", argv, workers=True)
    genomes = read_genomes(args.collection, args.num_genomes)
    out = output_dir(args.out, "results/bwt_similarity_naive_parallel", args.collection, len(genomes))

    start = time.time()
    res = similarity_naive_parallel(genomes, args.workers, progress=True)
    elapsed = time.time() - start

    write_results(out, res)
    print_summary(res, genomes, elapsed, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
