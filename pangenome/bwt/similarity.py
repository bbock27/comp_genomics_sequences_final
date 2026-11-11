"""Pairwise genome similarity from BWT run counts.

For genomes A and B, let n be the length of the two-genome collection and r
the number of runs in its BWT. Similar genomes put matching characters next
to each other in the BWT, so they add few runs:

    A and B identical   r(A,B) is about r(A)
    A and B unrelated   r(A,B) is about r(A) + r(B)

Two numbers are reported per pair (lower = more similar in both):

r_over_n   r(A,B) / n(A,B), the raw measure
distance   (r(A,B) - max(r(A), r(B))) / min(r(A), r(B))
           about 0 for identical genomes and about 1 for unrelated ones. It
           removes the part of r/n that only reflects how repetitive each
           genome is by itself.

Method. One BWT is built for the whole collection. The BWT of any pair is
that BWT with the other genomes' rows deleted (see pangenome.bwt.build), so a single
pass over the rows yields the run count of every pair at once: for the row's
genome g and each other genome h, the previous row of the (g, h) pair is
whichever of g and h appeared most recently.

Usage:
    python -m pangenome.bwt.similarity --collection data/synthetic_small/uneven_d0.01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from simulator.genomes import genome_name, load_collection

from .build import build_bwt


@njit(cache=True)
def _pair_runs(bwt, doc, d):
    """runs[g, h] = runs of the (g, h) BWT that begin at a row of genome g;
    runs[g, g] = runs of genome g alone."""
    runs = np.zeros((d, d), dtype=np.int64)
    last_row = np.full(d, -1, dtype=np.int64)
    last_chr = np.zeros(d, dtype=np.uint8)
    for i in range(len(bwt)):
        g = doc[i]
        c = bwt[i]
        own_row = last_row[g]
        own_chr = last_chr[g]
        for h in range(d):
            # previous row among genomes g and h
            if last_row[h] > own_row:
                prev_row, prev_chr = last_row[h], last_chr[h]
            else:
                prev_row, prev_chr = own_row, own_chr
            if prev_row < 0 or prev_chr != c:
                runs[g, h] += 1
        last_row[g] = i
        last_chr[g] = c
    return runs


def pairwise_runs(bwt: np.ndarray, doc: np.ndarray, d: int) -> np.ndarray:
    """Symmetric matrix R: R[a, b] = runs in the BWT of genomes a and b;
    R[a, a] = runs in the BWT of genome a alone."""
    part = _pair_runs(bwt, doc.astype(np.int64), d)
    total = part + part.T
    np.fill_diagonal(total, np.diag(part))
    return total


def measures(runs: np.ndarray, genomes: list[str]) -> dict:
    """Turn a matrix of run counts (pairs off the diagonal, single genomes on
    it) into lengths, r/n, and normalised distance."""
    length = np.array([len(g) + 1 for g in genomes], dtype=np.int64)  # with terminator
    n = length[:, None] + length[None, :]
    np.fill_diagonal(n, length)
    own = np.diag(runs)
    distance = (runs - np.maximum.outer(own, own)) / np.minimum.outer(own, own)
    np.fill_diagonal(distance, 0.0)
    return {"runs": runs, "n": n, "r_over_n": runs / n, "distance": distance}


def similarity(genomes: list[str]) -> dict:
    """Run counts, lengths, r/n, and normalised distance for every pair."""
    bwt, doc = build_bwt(genomes)
    res = measures(pairwise_runs(bwt, doc, len(genomes)), genomes)
    res["collection_runs"] = int(np.count_nonzero(bwt[1:] != bwt[:-1]) + 1)
    res["collection_n"] = int(len(bwt))
    return res


# --------------------------------------------------------------------------
# output and evaluation
# --------------------------------------------------------------------------

def write_results(out: Path, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d = len(res["runs"])
    names = [genome_name(i) for i in range(d)]
    with open(out / "pairs.tsv", "w") as fh:
        fh.write("genome_a\tgenome_b\tn\tr\tr_over_n\tr_a\tr_b\tdistance\n")
        for a in range(d):
            for b in range(a + 1, d):
                fh.write(f"{names[a]}\t{names[b]}\t{res['n'][a, b]}\t{res['runs'][a, b]}\t{res['r_over_n'][a, b]:.6g}\t"
                         f"{res['runs'][a, a]}\t{res['runs'][b, b]}\t{res['distance'][a, b]:.6g}\n")
    for key in ("r_over_n", "distance"):
        with open(out / f"{key}_matrix.tsv", "w") as fh:
            fh.write("\t".join([""] + names) + "\n")
            for a in range(d):
                fh.write("\t".join([names[a]] + [f"{res[key][a, b]:.6g}" for b in range(d)]) + "\n")


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share a rank), for Spearman correlation."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    values, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / np.bincount(inverse))[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def evaluate(res: dict, truth_file: Path) -> dict | None:
    """Compare with the simulator's pairwise.tsv (true distance per pair)."""
    if not truth_file.exists():
        return None
    d = len(res["runs"])
    truth = np.full((d, d), np.nan)
    for line in truth_file.read_text().splitlines()[1:]:
        a, b, *_, dist = line.split("\t")
        a, b = int(a[1:]), int(b[1:])
        if a < d and b < d:
            truth[a, b] = truth[b, a] = float(dist)
    iu = np.triu_indices(d, 1)
    if len(iu[0]) < 2:
        return None
    out = {"pairs": len(iu[0])}
    for key in ("r_over_n", "distance"):
        m = res[key].copy()
        out[f"spearman_{key}"] = spearman(m[iu], truth[iu])
        np.fill_diagonal(m, np.inf)
        t = truth.copy()
        np.fill_diagonal(t, np.inf)
        # nearest neighbour counts as right if it is (one of) the truly nearest
        best = m.argmin(axis=1)
        out[f"nearest_{key}"] = int(sum(t[a, best[a]] <= t[a].min() * (1 + 1e-9) for a in range(d)))
    return out


def parse_args(prog: str, doc: str, default_dir: str, argv: list[str] | None) -> argparse.Namespace:
    """Command-line options shared by both similarity tools."""
    parser = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True, help="directory with genomes.fasta")
    parser.add_argument("--num-genomes", type=int, help="use only the first N genomes")
    parser.add_argument("--out", help=f"output directory (default {default_dir}/<collection>_nNNN)")
    return parser.parse_args(argv)


def read_genomes(collection: str, num_genomes: int | None) -> list[str]:
    """Read genomes.fasta from a collection directory (the first N records)."""
    genomes = load_collection(collection, num_genomes).genomes
    if len(genomes) < 2:
        raise SystemExit("need at least two genomes")
    return genomes


def output_dir(out: str | None, default_dir: str, collection: str, d: int) -> Path:
    if out:
        return Path(out)
    cdir = Path(collection)
    return Path(default_dir) / f"{cdir.parent.name}_{cdir.name}_n{d:03d}"


def print_summary(res: dict, genomes: list[str], elapsed: float, collection: str, out: Path) -> None:
    """Report timing, the range of both measures, and agreement with the
    simulator's true distances when the collection has a pairwise.tsv."""
    d = len(genomes)
    iu = np.triu_indices(d, 1)
    total = sum(len(g) + 1 for g in genomes)
    print(f"{d} genomes, {total} characters, built and compared in {elapsed:.1f} s", file=sys.stderr)
    if "collection_runs" in res:
        print(f"whole collection: r = {res['collection_runs']}, r/n = {res['collection_runs'] / res['collection_n']:.5f}",
              file=sys.stderr)
    print(f"pairs: r/n from {res['r_over_n'][iu].min():.5f} to {res['r_over_n'][iu].max():.5f}; "
          f"distance from {res['distance'][iu].min():.4f} to {res['distance'][iu].max():.4f}", file=sys.stderr)
    ev = evaluate(res, Path(collection) / "pairwise.tsv")
    if ev:
        for key in ("r_over_n", "distance"):
            print(f"against the true distances ({ev['pairs']} pairs), {key}: Spearman {ev[f'spearman_{key}']:.3f}, "
                  f"nearest neighbour right for {ev[f'nearest_{key}']}/{d} genomes", file=sys.stderr)
    print(f"wrote {out}/pairs.tsv, r_over_n_matrix.tsv, distance_matrix.tsv", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args("python -m pangenome.bwt.similarity", __doc__, "results/bwt_similarity", argv)
    genomes = read_genomes(args.collection, args.num_genomes)
    out = output_dir(args.out, "results/bwt_similarity", args.collection, len(genomes))

    start = time.time()
    res = similarity(genomes)
    elapsed = time.time() - start

    write_results(out, res)
    print_summary(res, genomes, elapsed, args.collection, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
