"""Serial against parallel: the four BWT run-count similarity versions.

    python -m experiments.bwt_similarity_parallel run     # measure, then plot
    python -m experiments.bwt_similarity_parallel plot    # redraw from saved measurements

On one collection, measures wall-clock time for

single             one BWT, serial sweep            (pangenome.bwt.similarity)
single_parallel    one BWT, sweep on several cores  (pangenome.bwt.similarity_parallel)
naive              one BWT per pair, one at a time  (pangenome.bwt.similarity_naive)
naive_parallel     one BWT per pair, several at once (pangenome.bwt.similarity_naive_parallel)

with the parallel versions repeated at 2, 4, 8, ... workers up to the number
of cores. The single BWT is built once and reused for every sweep
measurement, since the build is the same serial code in both single versions;
their totals are that one build time plus each sweep time.

Results go to results/bwt_similarity_parallel_comparison/<collection>_nNNN/.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from pangenome.bwt.build import build_bwt
from pangenome.bwt.similarity import pairwise_runs, read_genomes
from pangenome.bwt.similarity_naive import similarity_naive
from pangenome.bwt.similarity_naive_parallel import similarity_naive_parallel
from pangenome.bwt.similarity_parallel import pairwise_runs_parallel

RESULTS = Path("results/bwt_similarity_parallel_comparison")
COLUMNS = ["method", "workers", "seconds", "build_seconds", "sweep_seconds"]


def worker_counts(limit: int) -> list[int]:
    counts, w = [], 2
    while w < limit:
        counts.append(w)
        w *= 2
    return counts + [limit]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(collection: str, num_genomes: int | None, max_workers: int, out: Path) -> None:
    genomes = read_genomes(collection, num_genomes)
    d = len(genomes)
    counts = worker_counts(max_workers)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(len(g) + 1 for g in genomes)
    (out / "machine.txt").write_text(
        f"host\t{platform.node()}\nprocessor\t{platform.processor() or platform.machine()}\n"
        f"cores\t{os.cpu_count()}\ncollection\t{collection}\ngenomes\t{d}\ncharacters\t{total}\n")
    log(f"{d} genomes, {total} characters, up to {max_workers} workers")

    # compile both numba loops on a tiny input so compilation is not timed
    warm_bwt, warm_doc = build_bwt(["ACGTACGT", "ACGTTCGT", "TTGACA"])
    pairwise_runs(warm_bwt, warm_doc, 3)
    pairwise_runs_parallel(warm_bwt, warm_doc, 3, 2)

    with open(out / "measurements.tsv", "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")

        def record(method: str, workers: int, seconds: float, build: float | None, sweep: float | None) -> None:
            row = [method, workers, f"{seconds:.3f}", "-" if build is None else f"{build:.3f}",
                   "-" if sweep is None else f"{sweep:.3f}"]
            fh.write("\t".join(map(str, row)) + "\n")
            fh.flush()
            log(f"  {method:16s} workers {workers:3d}  {seconds:9.2f} s"
                + ("" if sweep is None else f"   (build {build:.2f} s, sweep {sweep:.3f} s)"))

        start = time.perf_counter()
        bwt, doc = build_bwt(genomes)
        build = time.perf_counter() - start

        start = time.perf_counter()
        reference = pairwise_runs(bwt, doc, d)
        sweep = time.perf_counter() - start
        record("single", 1, build + sweep, build, sweep)

        for workers in [1] + counts:
            start = time.perf_counter()
            got = pairwise_runs_parallel(bwt, doc, d, workers)
            sweep = time.perf_counter() - start
            if not np.array_equal(got, reference):
                raise SystemExit(f"parallel sweep with {workers} workers disagrees with the serial sweep")
            record("single_parallel", workers, build + sweep, build, sweep)
        del bwt, doc

        for workers in reversed(counts):  # fastest first, so partial results are useful
            start = time.perf_counter()
            got = similarity_naive_parallel(genomes, workers)
            seconds = time.perf_counter() - start
            if not np.array_equal(got["runs"], reference):
                raise SystemExit(f"parallel naive with {workers} workers disagrees with the single-BWT sweep")
            record("naive_parallel", workers, seconds, None, None)

        start = time.perf_counter()
        got = similarity_naive(genomes)
        seconds = time.perf_counter() - start
        if not np.array_equal(got["runs"], reference):
            raise SystemExit("serial naive disagrees with the single-BWT sweep")
        record("naive", 1, seconds, None, None)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(out: Path) -> tuple[list[dict], dict]:
    lines = (out / "measurements.tsv").read_text().splitlines()
    rows = [dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:]]
    machine = dict(line.split("\t") for line in (out / "machine.txt").read_text().splitlines())
    return rows, machine


def series(rows: list[dict], method: str, field: str) -> tuple[list[int], list[float]]:
    points = sorted((int(r["workers"]), float(r[field])) for r in rows if r["method"] == method and r[field] != "-")
    return [p[0] for p in points], [p[1] for p in points]


def format_seconds(value: float, _pos: object = None) -> str:
    if value >= 600:
        return f"{value / 60:g} min"
    return f"{value:g} s"


def format_plain(value: float, _pos: object = None) -> str:
    return f"{value:g}"


def plot(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows, machine = load(out)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)

    single_serial = series(rows, "single", "seconds")[1][0]
    naive_serial = series(rows, "naive", "seconds")
    sx, sy = series(rows, "single_parallel", "seconds")
    nx, ny = series(rows, "naive_parallel", "seconds")
    if naive_serial[1]:
        nx, ny = [1] + nx, [naive_serial[1][0]] + ny
    sweep_x, sweep_y = series(rows, "single_parallel", "sweep_seconds")
    ticks = sorted(set(sx) | set(nx))

    def style(ax, ylabel: str) -> None:
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(format_plain))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(format_seconds))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.minorticks_off()
        ax.grid(axis="y", which="major", color=GRID, linewidth=1, zorder=0)
        ax.tick_params(length=0, which="both")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Workers (cores used)", labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.margins(x=0.06, y=0.16)

    def line(ax, xs, ys, color, label) -> None:
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3, label=label)

    # left: total time of both methods
    line(left, sx, sy, BLUE, "Single BWT, parallel sweep")
    line(left, nx, ny, ORANGE, "Naive pairwise BWTs, parallel")
    style(left, "Total time")
    left.annotate(f"{format_seconds(sy[-1])}", xy=(sx[-1], sy[-1]), xytext=(0, 9), textcoords="offset points",
                  ha="center", fontsize=9.5, color=INK2)
    left.annotate(f"{format_seconds(round(ny[-1], 1))}", xy=(nx[-1], ny[-1]), xytext=(0, 9),
                  textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
    left.annotate(f"{format_seconds(round(ny[0], 1))}", xy=(nx[0], ny[0]), xytext=(0, 9), textcoords="offset points",
                  ha="center", fontsize=9.5, color=INK2)
    faster = "naive" if ny[-1] < sy[-1] else "single BWT"
    ratio = max(ny[-1], sy[-1]) / min(ny[-1], sy[-1])
    left.set_title("Whole computation", loc="left", fontsize=11.5, color=INK, pad=24, fontweight="bold")
    left.text(0, 1.03, f"naive is {ny[0] / ny[-1]:.0f}× faster at {nx[-1]} workers than at 1; "
              f"{faster} is then {ratio:.1f}× faster", transform=left.transAxes, fontsize=9.5, color=INK2)
    left.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=2.2)

    # right: the sweep alone
    line(right, sweep_x, sweep_y, BLUE, "Sweep only")
    style(right, "Sweep time")
    right.set_title("The sweep alone (single-BWT method)", loc="left", fontsize=11.5, color=INK, pad=24,
                    fontweight="bold")
    build = float([r for r in rows if r["method"] == "single"][0]["build_seconds"])
    right.text(0, 1.03, f"{sweep_y[0] / sweep_y[-1]:.0f}× faster at {sweep_x[-1]} workers, but building the BWT "
               f"takes {format_seconds(round(build, 1))} on one core", transform=right.transAxes, fontsize=9.5,
               color=INK2)
    right.annotate(f"{sweep_y[0]:.2f} s", xy=(sweep_x[0], sweep_y[0]), xytext=(0, 9), textcoords="offset points",
                   ha="center", fontsize=9.5, color=INK2)
    right.annotate(f"{sweep_y[-1]:.2f} s", xy=(sweep_x[-1], sweep_y[-1]), xytext=(0, 9), textcoords="offset points",
                   ha="center", fontsize=9.5, color=INK2)

    fig.suptitle("BWT run-count similarity: what extra cores buy", x=0.06, y=0.98, ha="left", fontsize=14,
                 color=INK, fontweight="bold")
    fig.text(0.06, 0.015, f"{machine['genomes']} genomes, {int(machine['characters']) / 1e6:.0f} million characters "
             f"({machine['collection']}). {machine['cores']} cores, host {machine['host']}. Both axes logarithmic. "
             f"One run per point. Serial single-BWT total: {format_seconds(round(single_serial, 1))}.",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.8, bottom=0.17, wspace=0.22)
    fig.savefig(out / "parallel.png", dpi=170, facecolor=SURFACE)
    log(f"wrote {out / 'parallel.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.bwt_similarity_parallel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "plot"])
    parser.add_argument("--collection", default="data/synthetic/uneven_d0.01")
    parser.add_argument("--num-genomes", type=int)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)

    cdir = Path(args.collection)
    count = args.num_genomes or len(read_genomes(args.collection, None))
    out = RESULTS / f"{cdir.parent.name}_{cdir.name}_n{count:03d}"
    if args.command == "run":
        run(args.collection, args.num_genomes, args.max_workers, out)
    plot(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
