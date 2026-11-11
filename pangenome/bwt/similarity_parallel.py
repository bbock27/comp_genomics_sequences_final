"""Pairwise genome similarity from BWT run counts: single BWT, parallel sweep.

Same measures and output files as pangenome.bwt.similarity. The BWT of the
whole collection is built the same way (on one core); only the sweep that
counts runs for every pair is spread over several cores.

How the sweep is split. The rows of the BWT are cut into equal chunks. A chunk
can be counted independently once it knows, for every genome, the row and
character where that genome last appeared before the chunk begins. A quick
serial pass records that state at each chunk boundary; the chunks then run in
parallel and their counts are added.

Usage:
    python -m pangenome.bwt.similarity_parallel --collection data/synthetic_small/uneven_d0.01 --workers 8
"""

from __future__ import annotations

import time

import numba
import numpy as np
from numba import njit, prange

from .build import build_bwt
from .similarity import measures, output_dir, parse_args, print_summary, read_genomes, write_results


@njit(cache=True)
def _boundary_state(bwt, doc, d, bounds):
    """For each chunk start, the row and character where every genome was last
    seen before it (-1 if not yet seen)."""
    chunks = len(bounds) - 1
    rows = np.full((chunks, d), -1, dtype=np.int64)
    chars = np.zeros((chunks, d), dtype=np.uint8)
    last_row = np.full(d, -1, dtype=np.int64)
    last_chr = np.zeros(d, dtype=np.uint8)
    for c in range(chunks):
        rows[c, :] = last_row
        chars[c, :] = last_chr
        for i in range(bounds[c], bounds[c + 1]):
            last_row[doc[i]] = i
            last_chr[doc[i]] = bwt[i]
    return rows, chars


@njit(parallel=True, cache=True)
def _pair_runs_parallel(bwt, doc, d, bounds, start_rows, start_chars):
    """The same counting loop as pangenome.bwt.similarity._pair_runs, run on
    every chunk at once; see that function for what runs[g, h] means."""
    chunks = len(bounds) - 1
    partial = np.zeros((chunks, d, d), dtype=np.int64)
    for c in prange(chunks):
        runs = np.zeros((d, d), dtype=np.int64)
        last_row = start_rows[c].copy()
        last_chr = start_chars[c].copy()
        for i in range(bounds[c], bounds[c + 1]):
            g = doc[i]
            ch = bwt[i]
            own_row = last_row[g]
            own_chr = last_chr[g]
            for h in range(d):
                if last_row[h] > own_row:
                    prev_row, prev_chr = last_row[h], last_chr[h]
                else:
                    prev_row, prev_chr = own_row, own_chr
                if prev_row < 0 or prev_chr != ch:
                    runs[g, h] += 1
            last_row[g] = i
            last_chr[g] = ch
        partial[c] = runs
    return partial.sum(axis=0)


def pairwise_runs_parallel(bwt: np.ndarray, doc: np.ndarray, d: int, workers: int) -> np.ndarray:
    """Symmetric matrix of run counts, as pangenome.bwt.similarity.pairwise_runs."""
    numba.set_num_threads(max(1, min(workers, numba.config.NUMBA_NUM_THREADS)))
    chunks = max(1, min(workers * 4, len(bwt)))  # a few chunks per core evens out the load
    bounds = np.linspace(0, len(bwt), chunks + 1).astype(np.int64)
    doc64 = doc.astype(np.int64)
    start_rows, start_chars = _boundary_state(bwt, doc64, d, bounds)
    part = _pair_runs_parallel(bwt, doc64, d, bounds, start_rows, start_chars)
    total = part + part.T
    np.fill_diagonal(total, np.diag(part))
    return total


def similarity_parallel(genomes: list[str], workers: int) -> dict:
    """Run counts, lengths, r/n, and normalised distance for every pair."""
    start = time.perf_counter()
    bwt, doc = build_bwt(genomes)
    built = time.perf_counter()
    runs = pairwise_runs_parallel(bwt, doc, len(genomes), workers)
    swept = time.perf_counter()
    res = measures(runs, genomes)
    res["collection_runs"] = int(np.count_nonzero(bwt[1:] != bwt[:-1]) + 1)
    res["collection_n"] = int(len(bwt))
    res["build_seconds"] = built - start
    res["sweep_seconds"] = swept - built
    return res


def main(argv: list[str] | None = None) -> int:
    args = parse_args("python -m pangenome.bwt.similarity_parallel", __doc__, "results/bwt_similarity_parallel", argv,
                      workers=True)
    genomes = read_genomes(args.collection, args.num_genomes)
    out = output_dir(args.out, "results/bwt_similarity_parallel", args.collection, len(genomes))

    start = time.time()
    res = similarity_parallel(genomes, args.workers)
    elapsed = time.time() - start

    write_results(out, res)
    print_summary(res, genomes, elapsed, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
