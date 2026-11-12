"""Pairwise genome similarity from BWT run counts: single BWT, built with SA-IS.

Same measures, sweep, and output files as pangenome.bwt.similarity. The only
difference is how the BWT of the collection is built: SA-IS (linear time, see
pangenome.bwt.build_sais) instead of prefix doubling.

Usage:
    python -m pangenome.bwt.similarity_sais --collection data/synthetic_small/uneven_d0.01
"""

from __future__ import annotations

import time

import numpy as np

from .build_sais import build_bwt_sais
from .similarity import measures, output_dir, pairwise_runs, parse_args, print_summary, read_genomes, write_results


def similarity_sais(genomes: list[str]) -> dict:
    """Run counts, lengths, r/n, and normalised distance for every pair."""
    start = time.perf_counter()
    bwt, doc = build_bwt_sais(genomes)
    built = time.perf_counter()
    runs = pairwise_runs(bwt, doc, len(genomes))
    swept = time.perf_counter()
    res = measures(runs, genomes)
    res["collection_runs"] = int(np.count_nonzero(bwt[1:] != bwt[:-1]) + 1)
    res["collection_n"] = int(len(bwt))
    res["build_seconds"] = built - start
    res["sweep_seconds"] = swept - built
    return res


def main(argv: list[str] | None = None) -> int:
    args = parse_args("python -m pangenome.bwt.similarity_sais", __doc__, "results/bwt_similarity_sais", argv)
    genomes = read_genomes(args.collection, args.num_genomes)
    out = output_dir(args.out, "results/bwt_similarity_sais", args.collection, len(genomes))

    start = time.time()
    res = similarity_sais(genomes)
    elapsed = time.time() - start

    write_results(out, res)
    print_summary(res, genomes, elapsed, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
