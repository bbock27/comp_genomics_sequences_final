"""How the two BWT run-count similarity methods scale: time and memory against genome
size, number of genomes, and variety.

    python -m experiments.bwt_similarity_scaling run     # measure (about 15 minutes)
    python -m experiments.bwt_similarity_scaling plot    # draw the figure

Three sweeps, each varying one thing:

size     genome length, with 8 genomes at 1% divergence
count    number of genomes, each 100 kb at 1% divergence
variety  how different the genomes are, with 8 genomes of 400 kb: identical
         copies, four divergence levels, and unrelated random genomes

Genomes are independent mutated copies of the start of E. coli K-12 (the
simulator's star model), so every pair in a collection is equally distant and
the only thing changing along a sweep is the swept quantity.

For each point both methods are run twice: once timed, and once under
tracemalloc to record the peak memory of their working arrays. That peak
excludes the genome text itself, which both methods hold in memory.

Results go to results/bwt_similarity_scaling/measurements.tsv and scaling.png.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pangenome.bwt.similarity import similarity
from pangenome.bwt.similarity_naive import similarity_naive
from simulator.genomes import simulate_collection, simulate_control
from simulator.seq import clean, read_fasta

OUT = Path("results/bwt_similarity_scaling")
REFERENCE = "data/reference/ecoli_k12_mg1655.fasta"
SEED = 447
COLUMNS = ["sweep", "label", "genome_length", "num_genomes", "divergence", "method", "seconds", "peak_bytes"]

SIZE_SWEEP = [50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000]
COUNT_SWEEP = [2, 4, 8, 16, 32]
VARIETY_SWEEP = [("identical", 0.0), ("0.01%", 0.0001), ("0.1%", 0.001), ("1%", 0.01), ("10%", 0.1),
                 ("unrelated", None)]


def make_genomes(reference: str, length: int, count: int, divergence: float | None) -> list[str]:
    if divergence is None:
        return simulate_control(length, count, SEED)
    return simulate_collection(reference[:length], count, divergence, SEED)[0]


def measure(method: str, genomes: list[str]) -> tuple[float, int]:
    """Return (seconds, peak bytes of working memory) for one method."""
    compute = similarity if method == "single" else similarity_naive
    start = time.perf_counter()
    compute(genomes)
    seconds = time.perf_counter() - start
    tracemalloc.start()
    compute(genomes)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return seconds, peak


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = clean(read_fasta(REFERENCE)[0][1])[0]
    similarity(make_genomes(reference, 2000, 3, 0.01))  # compile the numba loop before timing

    points = [("size", f"{m}", m, 8, 0.01) for m in SIZE_SWEEP]
    points += [("count", f"{d}", 100_000, d, 0.01) for d in COUNT_SWEEP]
    points += [("variety", label, 400_000, 8, div) for label, div in VARIETY_SWEEP]

    with open(OUT / "measurements.tsv", "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for sweep, label, length, count, divergence in points:
            genomes = make_genomes(reference, length, count, divergence)
            for method in ("single", "naive"):
                seconds, peak = measure(method, genomes)
                row = [sweep, label, length, count, "-" if divergence is None else divergence, method,
                       f"{seconds:.3f}", peak]
                fh.write("\t".join(map(str, row)) + "\n")
                fh.flush()
                print(f"{sweep:8s}{label:>10s}  {method:6s}  {seconds:8.2f} s  {peak / 1e6:9.1f} MB",
                      file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"single": ("#2a78d6", "Single BWT + sweep"), "naive": ("#eb6834", "Naive pairwise BWTs")}


def load() -> list[dict]:
    lines = (OUT / "measurements.tsv").read_text().splitlines()
    return [dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:]]


def format_count(value: float, _pos: object = None) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def format_seconds(value: float, _pos: object = None) -> str:
    if value >= 600:
        return f"{value / 60:g} min"
    return f"{value:g} s"


def format_bytes(value: float, _pos: object = None) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g} GB"
    return f"{value / 1e6:g} MB"


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows = load()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), facecolor=SURFACE, sharey="row")
    sweeps = [("size", "genome_length", "Genome length (bases), 8 genomes"),
              ("count", "num_genomes", "Number of genomes, 100 kb each"),
              ("variety", None, "Divergence between genomes, 8 genomes of 400 kb")]
    measures_ = [("seconds", "Time", format_seconds), ("peak_bytes", "Peak working memory", format_bytes)]

    for r, (field, ylabel, yformat) in enumerate(measures_):
        for c, (sweep, xfield, xlabel) in enumerate(sweeps):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ends = {}
            for method, (color, _name) in SERIES.items():
                pts = [row for row in rows if row["sweep"] == sweep and row["method"] == method]
                xs = [float(p[xfield]) for p in pts] if xfield else list(range(len(pts)))
                ys = [float(p[field]) for p in pts]
                ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
                ends[method] = (xs, ys)
            ax.set_yscale("log")
            if xfield:
                ax.set_xscale("log", base=2)
                ax.set_xticks(ends["single"][0])
                ax.xaxis.set_major_formatter(FuncFormatter(format_count))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.minorticks_off()
                ax.set_yscale("log")
            else:
                pts = [row for row in rows if row["sweep"] == sweep and row["method"] == "single"]
                ax.set_xticks(range(len(pts)))
                ax.set_xticklabels([p["label"] for p in pts])
                ax.set_xlim(-0.4, len(pts) - 0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(yformat))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y", which="major", color=GRID, linewidth=1, zorder=0)
            ax.tick_params(length=0, which="both")
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if r == 1:
                ax.set_xlabel(xlabel, labelpad=8)
            if c == 0:
                ax.set_ylabel(ylabel, labelpad=8)

            # how many times more the naive method costs at the last point
            single, naive = ends["single"][1], ends["naive"][1]
            ratio = naive[-1] / single[-1]
            text = f"naive {ratio:.0f}× more" if ratio >= 1.5 else (
                f"single {1 / ratio:.0f}× more" if ratio <= 1 / 1.5 else "about equal")
            where = {"size": "at 1.6M bases", "count": "at 32 genomes", "variety": "for unrelated genomes"}[sweep]
            ax.text(0, 1.03, f"{text} {where}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
                    color=INK2)
            ax.margins(y=0.1)

    titles = ["Growing the genomes", "Adding genomes", "Making genomes more different"]
    for c, title in enumerate(titles):
        axes[0][c].set_title(title, loc="left", fontsize=11.5, color=INK, pad=24, fontweight="bold")

    handles = [plt.Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                          markeredgewidth=1.5, label=name) for color, name in SERIES.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.95), ncol=2, frameon=False,
               fontsize=10, labelcolor=INK2, handlelength=2.2, columnspacing=2.5)
    fig.suptitle("Pairwise genome similarity from BWT run counts: cost of the two methods", x=0.06, y=0.985,
                 ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.012, "Both axes are logarithmic except the divergence axis. Working memory excludes the genome "
             "text. One run per point; independent mutated copies of E. coli K-12.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.975, top=0.835, bottom=0.11, wspace=0.09, hspace=0.24)
    fig.savefig(OUT / "scaling.png", dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT / 'scaling.png'}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.bwt_similarity_scaling", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "plot"])
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
