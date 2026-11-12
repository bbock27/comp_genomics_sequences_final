"""Midpoint figures with every similarity method side by side.

    python -m experiments.midpoint_figures_all_methods full     # full-size collection, about 40 minutes, 32 cores
    python -m experiments.midpoint_figures_all_methods scaling  # small sweeps, about 2 minutes
    python -m experiments.midpoint_figures_all_methods plot     # draw every figure from saved results

The methods, all computing the similarity of every pair of genomes:

debruijn             colored de Bruijn graph, shared k-mers / all k-mers
naive_bwt            a BWT per pair, prefix doubling, one core
naive_bwt_parallel   the same, pairs spread over 8 processes
single_bwt           one BWT of the collection (prefix doubling), one sweep
single_bwt_parallel  the same, sweep spread over 32 cores
sais_bwt             one BWT of the collection built with SA-IS, one sweep

The five BWT methods give identical run counts; full checks this.

full     runs each method once on data/synthetic/uneven_d0.01, each in its own
         process group, recording wall-clock time and peak memory. Memory is
         the largest total resident size of the group, sampled twice a second,
         so worker processes are counted (pages they share are counted once
         per process, which overstates the parallel naive method a little).
scaling  adds the single-BWT prefix-doubling method to the sweeps measured by
         experiments.debruijn_colored (same genomes, same machine), for the
         four one-core methods.
plot     writes the figures to results/midpoint_figures_all_methods/. The
         cores figure also uses experiments.bwt_similarity_parallel's
         measurements; figures that do not depend on the method are copied
         from results/midpoint_figures/ and results/debruijn_colored/.

Measurements already saved are kept: full and scaling only run what is missing.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments import debruijn_colored as small
from pangenome.bwt.similarity import read_genomes, similarity, spearman

OUT = Path("results/midpoint_figures_all_methods")
FULL = "data/synthetic/uneven_d0.01"
PARALLEL = Path("results/bwt_similarity_parallel_comparison/synthetic_uneven_d0.01_n016/measurements.tsv")
NAIVE_WORKERS = 8      # the naive method stops speeding up beyond this (see the cores figure)
SWEEP_WORKERS = 32

SURFACE, INK, INK2, MUTED, GRID, AXIS = small.SURFACE, small.INK, small.INK2, small.MUTED, small.GRID, small.AXIS
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# name, colour, label, cores, module, extra arguments. A parallel method keeps
# the colour of its one-core version and is drawn hatched or dashed.
METHODS = [
    ("debruijn", AQUA, "Colored de Bruijn graph", 1, "pangenome.colored_debruijn.similarity", []),
    ("naive_bwt", ORANGE, "Naive pairwise BWTs", 1, "pangenome.bwt.similarity_naive", []),
    ("naive_bwt_parallel", ORANGE, "Naive pairwise BWTs", NAIVE_WORKERS, "pangenome.bwt.similarity_naive_parallel",
     ["--workers", str(NAIVE_WORKERS)]),
    ("single_bwt", BLUE, "Single BWT, prefix doubling", 1, "pangenome.bwt.similarity", []),
    ("single_bwt_parallel", BLUE, "Single BWT, prefix doubling", SWEEP_WORKERS, "pangenome.bwt.similarity_parallel",
     ["--workers", str(SWEEP_WORKERS)]),
    ("sais_bwt", YELLOW, "Single BWT, SA-IS", 1, "pangenome.bwt.similarity_sais", []),
]
COLOR = {m[0]: m[1] for m in METHODS}
LABEL = {m[0]: m[2] for m in METHODS}
CORES = {m[0]: m[3] for m in METHODS}
SERIAL = ["debruijn", "naive_bwt", "single_bwt", "sais_bwt"]
FULL_COLUMNS = ["method", "cores", "collection", "genomes", "bases", "seconds", "peak_rss_bytes", "host"]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# full
# --------------------------------------------------------------------------

def group_rss(group: int) -> int:
    """Total resident memory, in bytes, of every process in a session."""
    done = subprocess.run(["ps", "-o", "rss=", "-g", str(group)], capture_output=True, text=True)
    return sum(int(v) for v in done.stdout.split()) * 1024


def run_in_group(arguments: list[str]) -> tuple[float, int]:
    """Run a Python module in its own session. Returns (seconds, peak total
    resident memory of the session in bytes)."""
    start = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-m"] + arguments, start_new_session=True)
    peak = 0
    while process.poll() is None:
        peak = max(peak, group_rss(process.pid))
        time.sleep(0.5)
    seconds = time.perf_counter() - start
    if process.returncode != 0:
        raise SystemExit(f"{arguments[0]} failed with status {process.returncode}")
    return seconds, peak


def full() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genomes = read_genomes(FULL, None)
    d = len(genomes)
    bases = sum(len(g) for g in genomes)
    del genomes
    saved = {}
    if (OUT / "full_scale.tsv").exists():
        for row in small.read_table(OUT / "full_scale.tsv"):
            saved[row["method"]] = [row[c] for c in FULL_COLUMNS]
    rows = []
    for name, _color, _label, cores, module, extra in METHODS:
        if name in saved:
            rows.append(saved[name])
            continue
        arguments = [module, "--collection", FULL, "--out", str(OUT / f"full_{name}")] + extra
        seconds, peak = run_in_group(arguments)
        rows.append([name, cores, FULL, d, bases, f"{seconds:.1f}", peak, platform.node()])
        log(f"full  {name:20s} {cores:2d} cores  {seconds:8.1f} s  {peak / 1e9:6.2f} GB")
        small.write_table(OUT / "full_scale.tsv", FULL_COLUMNS, rows)
    small.write_table(OUT / "full_scale.tsv", FULL_COLUMNS, rows)

    reference = small.matrix_from(OUT / "full_sais_bwt" / "distance_matrix.tsv")
    for name in ("naive_bwt", "naive_bwt_parallel", "single_bwt", "single_bwt_parallel"):
        if not np.array_equal(small.matrix_from(OUT / f"full_{name}" / "distance_matrix.tsv"), reference):
            raise SystemExit(f"{name} disagrees with sais_bwt")
    log("all five BWT methods wrote the same distance matrix")
    small.write_table(OUT / "accuracy_full.tsv", small.ACCURACY_COLUMNS,
                      small.accuracy_rows(FULL, "full", small.matrix_from(OUT / "full_debruijn" / "similarity_matrix.tsv"),
                                          reference))


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------

def run_single_bwt(genomes: list[str]) -> None:
    similarity(genomes)


def scaling() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["sweep", "label", "genome_length", "num_genomes", "divergence", "method", "seconds", "peak_bytes"]
    measured = small.read_table(small.OUT / "scaling.tsv")
    rows = [[r[c] for c in columns] for r in measured]
    if (OUT / "scaling.tsv").exists():
        rows += [[r[c] for c in columns] for r in small.read_table(OUT / "scaling.tsv") if r["method"] == "single_bwt"]
    have = {(r[0], r[1]) for r in rows if r[5] == "single_bwt"}
    reference = small.clean(small.read_fasta(small.REFERENCE)[0][1])[0]
    run_single_bwt(small.make_genomes(reference, 2000, 3, 0.01))  # compile the numba loop before timing
    points = {(r["sweep"], r["label"]): r for r in measured}
    for (sweep, label), r in points.items():
        if (sweep, label) in have:
            continue
        divergence = None if r["divergence"] == "-" else float(r["divergence"])
        genomes = small.make_genomes(reference, int(r["genome_length"]), int(r["num_genomes"]), divergence)
        seconds, peak = small.measure(run_single_bwt, genomes)
        rows.append([sweep, label, r["genome_length"], r["num_genomes"], r["divergence"], "single_bwt",
                     f"{seconds:.3f}", peak])
        log(f"scaling  {sweep:8s}{label:>10s}  single_bwt  {seconds:8.2f} s  {peak / 1e6:9.1f} MB")
    small.write_table(OUT / "scaling.tsv", columns, rows)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    log(f"wrote {OUT / name}")


def format_seconds(value: float, _pos: object = None) -> str:
    return f"{value:g} s"


def plot_full_scale(plt) -> None:
    rows = {r["method"]: r for r in small.read_table(OUT / "full_scale.tsv")}
    names = [m[0] for m in METHODS if m[0] in rows]
    first = rows[names[0]]
    seconds = {m: float(rows[m]["seconds"]) for m in names}
    memory = {m: float(rows[m]["peak_rss_bytes"]) / 1e9 for m in names}
    fastest = min(names, key=seconds.get)
    smallest = min(names, key=memory.get)
    fig = small.new_figure(plt, 13, 6.6, "Every method on the full-size collection: time and memory",
                           f"All 120 pairwise similarities of {first['genomes']} genomes, {int(first['bases']) / 1e6:.0f} "
                           f"million bases ({first['collection']}). Host {first['host']}, one run each. Hatched bars "
                           "use several cores.\nMemory is the peak total of all the method's processes. The graph is "
                           "plain Python dictionaries of strings; the BWTs are numpy arrays, with numba for SA-IS and "
                           "the sweeps.")
    panels = [(seconds, "Time", "{:.0f} s", f"fastest: {LABEL[fastest]}, {seconds[fastest]:.0f} s"),
              (memory, "Peak memory", "{:.1f} GB", f"smallest: {LABEL[smallest]}, {memory[smallest]:.1f} GB")]
    for i, (values, title, pattern, subtitle) in enumerate(panels):
        ax = fig.add_axes([0.3 + i * 0.36, 0.2, 0.25, 0.58])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(length=0)
        top = max(values.values())
        labels = []
        for y, name in enumerate(names):
            hatch = "////" if CORES[name] > 1 else None
            ax.barh([y], [values[name]], height=0.6, color=COLOR[name], hatch=hatch, edgecolor=SURFACE, linewidth=0,
                    zorder=3)
            ax.text(values[name] + top * 0.02, y, pattern.format(values[name]), va="center", fontsize=10.5, color=INK)
            cores = "1 core" if CORES[name] == 1 else f"{CORES[name]} cores"
            labels.append(f"{LABEL[name]}\n{cores}")
        ax.set_xlim(0, top * 1.3)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks(range(len(names)), labels if i == 0 else [""] * len(names), fontsize=10, color=INK2)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=26, fontweight="bold")
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)
    save(fig, "1_full_scale_all_methods.png")


def plot_scaling(plt, FuncFormatter, NullFormatter) -> None:
    rows = small.read_table(OUT / "scaling.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), facecolor=SURFACE, sharey="row")
    sweeps = [("size", "genome_length", "Genome length (bases), 8 genomes"),
              ("count", "num_genomes", "Number of genomes, 100 kb each"),
              ("variety", None, "Divergence between genomes, 8 genomes of 400 kb")]
    fields = [("seconds", "Time", format_seconds), ("peak_bytes", "Peak working memory", small.format_bytes)]
    brief = {"debruijn": "de Bruijn", "naive_bwt": "naive", "single_bwt": "single", "sais_bwt": "SA-IS"}
    for r, (field, ylabel, yformat) in enumerate(fields):
        for c, (sweep, xfield, xlabel) in enumerate(sweeps):
            ax = axes[r][c]
            small.style_axes(ax)
            last = {}
            for method in SERIAL:
                mine = [row for row in rows if row["sweep"] == sweep and row["method"] == method]
                xs = [float(p[xfield]) for p in mine] if xfield else list(range(len(mine)))
                ys = [float(p[field]) for p in mine]
                small.draw_line(ax, xs, ys, COLOR[method], LABEL[method])
                last[method] = ys[-1]
                labels = [p["label"] for p in mine]
            ax.set_yscale("log")
            if xfield:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.xaxis.set_major_formatter(FuncFormatter(small.format_count))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlim(-0.4, len(labels) - 0.6)
            ax.minorticks_off()
            ax.yaxis.set_major_formatter(FuncFormatter(yformat))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if r == 1:
                ax.set_xlabel(xlabel, labelpad=8)
            if c == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            parts = [f"{brief[m]} {yformat(float(f'{last[m]:.2g}'))}" for m in SERIAL]
            ax.text(0, 1.03, " · ".join(parts), transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK2)
            ax.margins(y=0.1)
    for c, title in enumerate(["Growing the genomes", "Adding genomes", "Making genomes more different"]):
        axes[0][c].set_title(title, loc="left", fontsize=11.5, color=INK, pad=24, fontweight="bold")
    handles = [plt.Line2D([0], [0], color=COLOR[m], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                          markeredgewidth=1.3, label=LABEL[m]) for m in SERIAL]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.95), ncol=4, frameon=False, fontsize=10,
               labelcolor=INK2, handlelength=2.2, columnspacing=2.5)
    fig.suptitle("Pairwise genome similarity: how the cost of each method grows", x=0.06, y=0.985, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.012, f"One core each. Both axes are logarithmic except the divergence axis. k = {small.DEFAULT_K}. "
             "Working memory excludes the genome text. One run per point; independent mutated copies of E. coli K-12.\n"
             "The line above each panel gives the values at its rightmost point. The graph is plain Python "
             "dictionaries of strings; the BWTs are numpy arrays.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.975, top=0.835, bottom=0.125, wspace=0.09, hspace=0.24)
    save(fig, "2_scaling_all_methods.png")


def plot_cores(plt, FuncFormatter, NullFormatter, LogLocator) -> None:
    measured = small.read_table(PARALLEL)
    full_rows = {r["method"]: r for r in small.read_table(OUT / "full_scale.tsv")}
    first = full_rows["debruijn"]
    fig = small.new_figure(plt, 12, 6.6, "What extra cores buy each method",
                           f"{first['genomes']} genomes, {int(first['bases']) / 1e6:.0f} million bases "
                           f"({first['collection']}), host {first['host']}. Both axes logarithmic. One run per point.\n"
                           "Solid lines: the two methods with a parallel version. Dashed lines: one-core methods, "
                           "drawn across for reference.")
    ax = fig.add_axes([0.08, 0.2, 0.62, 0.62])
    small.style_axes(ax)
    series = [("naive_parallel", "naive", "naive_bwt"), ("single_parallel", "single", "single_bwt")]
    nudge = {"debruijn": 7, "single_bwt": -7}  # these two end within a few seconds of each other
    ticks = set()
    for parallel_name, serial_name, method in series:
        points = {int(r["workers"]): float(r["seconds"]) for r in measured if r["method"] == parallel_name}
        serial = [float(r["seconds"]) for r in measured if r["method"] == serial_name]
        if serial and 1 not in points:
            points[1] = serial[0]
        xs = sorted(points)
        ticks |= set(xs)
        small.draw_line(ax, xs, [points[x] for x in xs], COLOR[method], LABEL[method])
        ax.annotate(f"{LABEL[method]}: {format_seconds(float(f'{points[xs[-1]]:.3g}'))}", xy=(xs[-1], points[xs[-1]]),
                    xytext=(10, nudge.get(method, 0)), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    ticks = sorted(ticks)
    for method in ("debruijn", "sais_bwt"):
        value = float(full_rows[method]["seconds"])
        ax.plot([ticks[0], ticks[-1]], [value, value], color=COLOR[method], linewidth=2, linestyle=(0, (4, 3)),
                zorder=2, label=f"{LABEL[method]} (1 core)")
        ax.annotate(f"{LABEL[method]}: {format_seconds(float(f'{value:.3g}'))}", xy=(ticks[-1], value),
                    xytext=(10, nudge.get(method, 0)), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(small.plain))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(format_seconds))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Cores used", labelpad=8, color=INK2)
    ax.set_ylabel("Total time", labelpad=8, color=INK2)
    ax.margins(x=0.05, y=0.12)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=2.6)
    save(fig, "3_cores_all_methods.png")


def plot_accuracy(plt, FuncFormatter) -> None:
    rows = small.read_table(OUT / "accuracy_full.tsv")
    truth = np.array([float(r["true_distance"]) for r in rows])
    debruijn = np.array([float(r["debruijn_similarity"]) for r in rows])
    bwt = np.array([float(r["bwt_distance"]) for r in rows])
    fig = small.new_figure(plt, 14, 6.2, "Every method ranks pairs of genomes by their true distance",
                           f"One point per pair, {len(rows)} pairs: 16 genomes of 4.6 Mb ({FULL}). True distance = "
                           "fraction of sites that differ, from the simulator.\nde Bruijn similarity = shared "
                           f"{small.DEFAULT_K}-mers / all {small.DEFAULT_K}-mers of the pair. BWT distance = (r(A,B) - "
                           "max(r(A), r(B))) / min(r(A), r(B)), r = runs in the BWT.")
    panels = [(truth, debruijn, AQUA, "True distance", "de Bruijn similarity", small.percent, "Colored de Bruijn graph",
               f"Spearman {spearman(truth, debruijn):.3f} (similarity falls as distance grows)"),
              (truth, bwt, INK2, "True distance", "BWT distance", small.percent, "All five BWT methods",
               f"Spearman {spearman(truth, bwt):.3f}; the five give identical values"),
              (bwt, debruijn, INK2, "BWT distance", "de Bruijn similarity", small.plain, "Measure against measure",
               f"Spearman {spearman(bwt, debruijn):.3f}")]
    for i, (xs, ys, color, xlabel, ylabel, xformat, title, subtitle) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.32, 0.2, 0.25, 0.56])
        small.style_axes(ax)
        ax.scatter(xs, ys, s=40, color=color, edgecolors=SURFACE, linewidths=1.1, zorder=3)
        ax.set_xlim(0, xs.max() * 1.06)
        ax.set_ylim(0, 1.0 if ylabel.startswith("de Bruijn") else ys.max() * 1.08)
        ax.xaxis.set_major_formatter(FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(FuncFormatter(small.plain))
        ax.set_xlabel(xlabel, labelpad=8, color=INK2)
        ax.set_ylabel(ylabel, labelpad=8, color=INK2)
        small.panel_title(ax, title, subtitle)
    save(fig, "4_accuracy_all_methods.png")


COPIES = [("results/debruijn_colored/3_growth_vs_bwt.png", "5_index_growth_graph_vs_bwt.png"),
          ("results/midpoint_figures/5_run_growth.png", "6_bwt_run_growth_full_size.png"),
          ("results/debruijn_colored/1_k_sweep.png", "7_debruijn_k_sweep.png"),
          ("results/debruijn_colored/2_color_spectrum.png", "8_debruijn_color_spectrum.png"),
          ("results/midpoint_figures/1_synthetic_tree.png", "9_synthetic_tree.png"),
          ("results/midpoint_figures/3_synthetic_heatmaps.png", "10_synthetic_heatmaps.png"),
          ("results/midpoint_figures/4_real_ecoli_heatmap.png", "11_real_ecoli_heatmap.png")]


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "hatch.color": SURFACE,
                         "hatch.linewidth": 1.6})
    OUT.mkdir(parents=True, exist_ok=True)
    plot_full_scale(plt)
    plot_scaling(plt, FuncFormatter, NullFormatter)
    plot_cores(plt, FuncFormatter, NullFormatter, LogLocator)
    plot_accuracy(plt, FuncFormatter)
    for source, name in COPIES:
        if Path(source).exists():
            shutil.copyfile(source, OUT / name)
            log(f"copied {source} to {OUT / name}")
        else:
            log(f"missing {source}: not copied")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.midpoint_figures_all_methods", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["full", "scaling", "plot"])
    args = parser.parse_args(argv)
    if args.command == "full":
        full()
    elif args.command == "scaling":
        scaling()
    else:
        plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
