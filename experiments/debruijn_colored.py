"""The colored de Bruijn graph: experiments by itself and against the BWT methods.

    python -m experiments.debruijn_colored run     # small collections, about 10 minutes
    python -m experiments.debruijn_colored full    # full-size collection, about 25 minutes, needs ~20 GB
    python -m experiments.debruijn_colored plot    # draw every figure from saved results

All methods answer the same question, how similar every pair of genomes is:

de Bruijn   similarity = shared k-mers / all k-mers of the pair (intersection
            over union), from pangenome.colored_debruijn
naive BWT   distance from the run count of a separate BWT per pair, built by
            prefix doubling, from pangenome.bwt.similarity_naive
SA-IS BWT   the same distance from one BWT of the whole collection, built by
            SA-IS, and one sweep, from pangenome.bwt.similarity_sais

The two BWT methods give identical run counts, so they differ in cost only.
Measurements already saved are kept: run and full only measure what is missing
from scaling.tsv and full_scale.tsv, so a method can be added later.

run   uses the 100 kb collections in data/synthetic_small (16 genomes each):

      k sweep     graph size and accuracy for a range of k
      spectrum    how many k-mers are in exactly 1, 2, ..., 16 genomes
      growth      graph nodes and BWT runs as genomes are added one at a time
      accuracy    both measures against the true distance of every pair
      scaling     time and memory of every method against genome length,
                  number of genomes, and divergence (as in
                  experiments.bwt_similarity_scaling: independent mutated
                  copies of the start of E. coli K-12)

full  runs each method once on data/synthetic/uneven_d0.01 (16 genomes of
      4.6 Mb), each in its own process, recording time and peak memory.

Results and figures go to results/debruijn_colored/.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pangenome.bwt.build import build_bwt, count_runs
from pangenome.bwt.similarity import read_genomes, spearman
from pangenome.bwt.similarity_naive import similarity_naive
from pangenome.bwt.similarity_sais import similarity_sais
from pangenome.colored_debruijn.build_debruijn_color import build_graph
from pangenome.colored_debruijn.similarity import DEFAULT_K, color_patterns, measures, shared_kmers, similarity_debruijn
from simulator.genomes import simulate_collection, simulate_control
from simulator.seq import clean, read_fasta

OUT = Path("results/debruijn_colored")
REFERENCE = "data/reference/ecoli_k12_mg1655.fasta"
SEED = 447
SMALL = "data/synthetic_small"
LOW, HIGH, CONTROL = "uneven_d0.001", "uneven_d0.01", "control"
FULL = "data/synthetic/uneven_d0.01"

K_SWEEP = [7, 9, 11, 13, 15, 21, 31, 41, 51, 63]
SIZE_SWEEP = [50_000, 100_000, 200_000, 400_000, 800_000]
COUNT_SWEEP = [2, 4, 8, 16, 32]
VARIETY_SWEEP = [("identical", 0.0), ("0.01%", 0.0001), ("0.1%", 0.001), ("1%", 0.01), ("10%", 0.1),
                 ("unrelated", None)]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def write_table(path: Path, columns: list[str], rows: list[list]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    log(f"wrote {path} ({len(rows)} rows)")


def read_table(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def true_distances(collection: str, d: int) -> np.ndarray:
    truth = np.zeros((d, d))
    for row in read_table(Path(collection) / "pairwise.tsv"):
        a, b = int(row["genome_a"][1:]), int(row["genome_b"][1:])
        truth[a, b] = truth[b, a] = float(row["distance"])
    return truth


# --------------------------------------------------------------------------
# run: small collections
# --------------------------------------------------------------------------

def k_sweep() -> None:
    rows = []
    for name in (LOW, HIGH):
        collection = f"{SMALL}/{name}"
        genomes = read_genomes(collection, None)
        d = len(genomes)
        truth = true_distances(collection, d)
        iu = np.triu_indices(d, 1)
        for k in K_SWEEP:
            start = time.perf_counter()
            colors, edges = build_graph(genomes, k)
            seconds = time.perf_counter() - start
            res = measures(shared_kmers(color_patterns(colors), d))
            edge_count = 0
            branching = 0
            for targets in edges.values():
                edge_count += len(targets)
                if len(targets) > 1:
                    branching += 1
            rank = -spearman(res["similarity"][iu], truth[iu])
            rows.append([name, k, len(colors), edge_count, branching, f"{seconds:.3f}",
                         f"{res['similarity'][iu].min():.6g}", f"{res['similarity'][iu].max():.6g}", f"{rank:.4f}"])
            log(f"k sweep  {name}  k={k:2d}  {len(colors):8d} nodes  {branching:7d} branching  Spearman {rank:.3f}")
    write_table(OUT / "k_sweep.tsv", ["collection", "k", "nodes", "edges", "branching_nodes", "build_seconds",
                                      "min_similarity", "max_similarity", "spearman"], rows)


def spectrum_and_growth() -> None:
    spectrum, growth = [], []
    for name in (LOW, HIGH, CONTROL):
        genomes = read_genomes(f"{SMALL}/{name}", None)
        d = len(genomes)
        colors, _edges = build_graph(genomes, DEFAULT_K)
        in_exactly = [0] * (d + 1)   # k-mers found in exactly j genomes
        first_seen = [0] * d         # k-mers whose lowest-numbered genome is j
        for members in colors.values():
            in_exactly[len(members)] += 1
            first_seen[min(members)] += 1
        for j in range(1, d + 1):
            spectrum.append([name, j, in_exactly[j]])
        nodes = 0
        bases = 0
        for j in range(1, d + 1):
            # a k-mer is in the graph of the first j genomes if one of them has it
            nodes += first_seen[j - 1]
            bases += len(genomes[j - 1])
            runs = count_runs(build_bwt(genomes[:j])[0])
            growth.append([name, j, bases, nodes, runs])
        log(f"spectrum and growth  {name}  {len(colors)} nodes, {growth[-1][4]} runs")
    write_table(OUT / "color_spectrum.tsv", ["collection", "genomes_sharing", "kmers"], spectrum)
    write_table(OUT / "growth.tsv", ["collection", "genomes", "bases", "nodes", "runs"], growth)


def accuracy_rows(collection: str, label: str, debruijn: np.ndarray, bwt: np.ndarray) -> list[list]:
    d = len(debruijn)
    truth = true_distances(collection, d)
    rows = []
    for a in range(d):
        for b in range(a + 1, d):
            rows.append([label, a, b, f"{truth[a, b]:.8g}", f"{debruijn[a, b]:.8g}", f"{bwt[a, b]:.8g}"])
    return rows


ACCURACY_COLUMNS = ["collection", "genome_a", "genome_b", "true_distance", "debruijn_similarity", "bwt_distance"]


def accuracy() -> None:
    rows = []
    for name in (LOW, HIGH):
        collection = f"{SMALL}/{name}"
        genomes = read_genomes(collection, None)
        debruijn = similarity_debruijn(genomes, DEFAULT_K)["similarity"]
        bwt = similarity_naive(genomes)["distance"]
        rows += accuracy_rows(collection, name, debruijn, bwt)
        log(f"accuracy  {name}")
    write_table(OUT / "accuracy.tsv", ACCURACY_COLUMNS, rows)


def run_debruijn(genomes: list[str]) -> None:
    similarity_debruijn(genomes, DEFAULT_K)


def run_naive_bwt(genomes: list[str]) -> None:
    similarity_naive(genomes)


def run_sais_bwt(genomes: list[str]) -> None:
    similarity_sais(genomes)


def measure(compute, genomes: list[str]) -> tuple[float, int]:
    """Return (seconds, peak bytes of working memory) for one method: one
    timed run, then one run under tracemalloc."""
    start = time.perf_counter()
    compute(genomes)
    seconds = time.perf_counter() - start
    tracemalloc.start()
    compute(genomes)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return seconds, peak


def make_genomes(reference: str, length: int, count: int, divergence: float | None) -> list[str]:
    if divergence is None:
        return simulate_control(length, count, SEED)
    return simulate_collection(reference[:length], count, divergence, SEED)[0]


def scaling() -> None:
    reference = clean(read_fasta(REFERENCE)[0][1])[0]
    points = [("size", f"{m}", m, 8, 0.01) for m in SIZE_SWEEP]
    points += [("count", f"{d}", 100_000, d, 0.01) for d in COUNT_SWEEP]
    points += [("variety", label, 400_000, 8, div) for label, div in VARIETY_SWEEP]
    methods = [("debruijn", run_debruijn), ("naive_bwt", run_naive_bwt), ("sais_bwt", run_sais_bwt)]
    columns = ["sweep", "label", "genome_length", "num_genomes", "divergence", "method", "seconds", "peak_bytes"]
    saved = {}
    if (OUT / "scaling.tsv").exists():
        for row in read_table(OUT / "scaling.tsv"):
            saved[row["sweep"], row["label"], row["method"]] = [row[c] for c in columns]
    run_sais_bwt(make_genomes(reference, 2000, 3, 0.01))  # compile the numba loops before timing
    rows = []
    for sweep, label, length, count, divergence in points:
        genomes = None
        for method, compute in methods:
            if (sweep, label, method) in saved:
                rows.append(saved[sweep, label, method])
                continue
            if genomes is None:
                genomes = make_genomes(reference, length, count, divergence)
            seconds, peak = measure(compute, genomes)
            rows.append([sweep, label, length, count, "-" if divergence is None else divergence, method,
                         f"{seconds:.3f}", peak])
            log(f"scaling  {sweep:8s}{label:>10s}  {method:10s}  {seconds:8.2f} s  {peak / 1e6:9.1f} MB")
    write_table(OUT / "scaling.tsv", columns, rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k_sweep()
    spectrum_and_growth()
    accuracy()
    scaling()


# --------------------------------------------------------------------------
# full: one full-size collection, each method in its own process
# --------------------------------------------------------------------------

def run_in_process(arguments: list[str]) -> tuple[float, int]:
    """Run a Python module as a separate process. Returns (seconds, peak
    resident memory in bytes)."""
    start = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-m"] + arguments)
    _pid, status, usage = os.wait4(process.pid, 0)
    seconds = time.perf_counter() - start
    if status != 0:
        raise SystemExit(f"{arguments[0]} failed with status {status}")
    scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux kilobytes
    return seconds, usage.ru_maxrss * scale


def matrix_from(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()[1:]
    return np.array([[float(v) for v in line.split("\t")[1:]] for line in lines])


def full() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genomes = read_genomes(FULL, None)
    d = len(genomes)
    bases = sum(len(g) for g in genomes)
    del genomes
    debruijn_out = OUT / "full_debruijn"
    bwt_out = OUT / "full_naive_bwt"
    sais_out = OUT / "full_sais_bwt"
    columns = ["method", "collection", "genomes", "bases", "seconds", "peak_rss_bytes", "host"]
    saved = {}
    if (OUT / "full_scale.tsv").exists():
        for row in read_table(OUT / "full_scale.tsv"):
            saved[row["method"]] = [row[c] for c in columns]
    rows = []
    jobs = [("debruijn", ["pangenome.colored_debruijn.similarity", "--collection", FULL, "--k", str(DEFAULT_K),
                          "--out", str(debruijn_out)]),
            ("naive_bwt", ["pangenome.bwt.similarity_naive", "--collection", FULL, "--out", str(bwt_out)]),
            ("sais_bwt", ["pangenome.bwt.similarity_sais", "--collection", FULL, "--out", str(sais_out)])]
    for method, arguments in jobs:
        if method in saved:
            rows.append(saved[method])
            continue
        seconds, peak = run_in_process(arguments)
        rows.append([method, FULL, d, bases, f"{seconds:.1f}", peak, platform.node()])
        log(f"full  {method:10s}  {seconds:8.1f} s  {peak / 1e9:6.2f} GB")
        write_table(OUT / "full_scale.tsv", columns, rows)
    write_table(OUT / "full_scale.tsv", columns, rows)
    same = np.array_equal(matrix_from(bwt_out / "distance_matrix.tsv"), matrix_from(sais_out / "distance_matrix.tsv"))
    log(f"naive and SA-IS BWT distance matrices identical: {same}")
    if not same:
        raise SystemExit("the two BWT methods disagree")
    write_table(OUT / "accuracy_full.tsv", ACCURACY_COLUMNS,
                accuracy_rows(FULL, "full", matrix_from(debruijn_out / "similarity_matrix.tsv"),
                              matrix_from(bwt_out / "distance_matrix.tsv")))


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# One colour per method in every figure, matching the BWT figures (blue there
# is the single-BWT method too).
DEBRUIJN, NAIVE_BWT, SAIS_BWT = AQUA, ORANGE, BLUE
ALL_METHODS = {"debruijn": (DEBRUIJN, "Colored de Bruijn graph"),
               "naive_bwt": (NAIVE_BWT, "Naive pairwise BWTs"),
               "sais_bwt": (SAIS_BWT, "Single BWT (SA-IS)")}
# The methods drawn in the figures. The SA-IS BWT is measured and saved by
# run and full, but left out of the figures; add "sais_bwt" here to show it.
PLOTTED = ["debruijn", "naive_bwt"]
METHODS = {m: ALL_METHODS[m] for m in PLOTTED}
SHORT = {"debruijn": "de Bruijn", "naive_bwt": "naive BWT", "sais_bwt": "SA-IS BWT"}
BRIEF = {"debruijn": "de Bruijn", "naive_bwt": "naive", "sais_bwt": "SA-IS"}
COLLECTION_NAMES = {LOW: "0.1% root-to-leaf divergence", HIGH: "1% root-to-leaf divergence",
                    CONTROL: "Unrelated random genomes"}


def format_count(value: float, _pos: object = None) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def format_seconds(value: float, _pos: object = None) -> str:
    return f"{value:g} s"


def format_bytes(value: float, _pos: object = None) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g} GB"
    return f"{value / 1e6:g} MB"


def percent(value: float, _pos: object = None) -> str:
    return f"{value * 100:g}%"


def plain(value: float, _pos: object = None) -> str:
    return f"{value:g}"


def new_figure(plt, width: float, height: float, title: str, footnote: str):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    fig.suptitle(title, x=0.05, y=0.965, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.05, 0.022, footnote, fontsize=8.5, color=MUTED)
    return fig


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", which="major", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, which="both")


def panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=26, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)


def draw_line(ax, xs, ys, color: str, label: str, size: float = 6) -> None:
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=size, markeredgecolor=SURFACE,
            markeredgewidth=1.3, solid_capstyle="round", solid_joinstyle="round", zorder=3, label=label)


def legend(ax, where: str) -> None:
    ax.legend(loc=where, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=2.2)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    log(f"wrote {OUT / name}")


def plot_k_sweep(plt, FuncFormatter, NullFormatter, LogLocator) -> None:
    rows = read_table(OUT / "k_sweep.tsv")
    fig = new_figure(plt, 14, 6.2, "Colored de Bruijn graph: choosing k",
                     f"16 genomes of 100 kb ({SMALL}), forward strand only. A branching node has edges to more than "
                     "one next k-mer.\nAccuracy = rank correlation (Spearman) between similarity and the true "
                     "distance of the 120 pairs, sign flipped so 1 is perfect.")
    panels = [("nodes", "Nodes (distinct k-mers)", "small k: every possible k-mer occurs, so the graph is full",
               format_count, True),
              ("branching_nodes", "Branching nodes", "fall as k passes the length of chance repeats", format_count,
               True),
              ("spearman", "Accuracy of the similarity", "poor until k-mers are long enough to be unique", plain,
               False)]
    series = [(HIGH, ORANGE), (LOW, BLUE)]
    for i, (field, title, subtitle, formatter, log_scale) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.32, 0.2, 0.26, 0.56])
        style_axes(ax)
        for name, color in series:
            mine = [r for r in rows if r["collection"] == name]
            xs = [int(r["k"]) for r in mine]
            ys = [float(r[field]) for r in mine]
            if log_scale:
                ys = [max(y, 0.5) for y in ys]  # zero cannot be drawn on a log axis
            draw_line(ax, xs, ys, color, COLLECTION_NAMES[name])
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.minorticks_off()
        else:
            ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))
        ax.set_xticks([7, 15, 21, 31, 41, 51, 63])
        ax.set_xlabel("k", labelpad=8, color=INK2)
        panel_title(ax, title, subtitle)
        if i == 0:
            legend(ax, "lower right")
    save(fig, "1_k_sweep.png")


def plot_spectrum(plt, FuncFormatter) -> None:
    rows = read_table(OUT / "color_spectrum.tsv")
    fig = new_figure(plt, 12.5, 6.2, "Colored de Bruijn graph: how widely each k-mer is shared",
                     f"k = {DEFAULT_K}. 16 genomes of 100 kb ({SMALL}). Each bar is the share of the graph's nodes whose "
                     "color set has that many genomes.\nThe tree shows through: a mutation on a branch makes new k-mers in the "
                     "1, 2, 4 or 8 genomes below it and leaves the old ones in the other 15, 14, 12 or 8.")
    for i, name in enumerate((LOW, HIGH)):
        mine = [r for r in rows if r["collection"] == name]
        xs = [int(r["genomes_sharing"]) for r in mine]
        counts = [int(r["kmers"]) for r in mine]
        total = sum(counts)
        ax = fig.add_axes([0.07 + i * 0.49, 0.2, 0.4, 0.56])
        style_axes(ax)
        ax.bar(xs, [c / total for c in counts], width=0.72, color=DEBRUIJN, zorder=3)
        ax.set_xticks(xs)
        ax.set_xlim(0.3, len(xs) + 0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(percent))
        ax.set_xlabel("Genomes that contain the k-mer", labelpad=8, color=INK2)
        if i == 0:
            ax.set_ylabel("Share of nodes", labelpad=8, color=INK2)
        panel_title(ax, COLLECTION_NAMES[name], f"{total:,} nodes: {counts[-1] / total:.0%} in all 16 genomes, "
                    f"{counts[0] / total:.0%} in only one")
    save(fig, "2_color_spectrum.png")


def plot_growth(plt) -> None:
    rows = read_table(OUT / "growth.tsv")
    fig = new_figure(plt, 14, 6.2, "Index size as genomes are added: graph nodes against BWT runs",
                     f"16 genomes of 100 kb ({SMALL}), added one at a time. Nodes for k = {DEFAULT_K}; runs counted in "
                     "the BWT of the first j genomes. Both divided by their value for one genome.\nThe dashed line "
                     "is the total length of the genomes on the same scale.")
    for i, name in enumerate((LOW, HIGH, CONTROL)):
        mine = [r for r in rows if r["collection"] == name]
        xs = [int(r["genomes"]) for r in mine]
        nodes = [int(r["nodes"]) / int(mine[0]["nodes"]) for r in mine]
        runs = [int(r["runs"]) / int(mine[0]["runs"]) for r in mine]
        bases = [int(r["bases"]) / int(mine[0]["bases"]) for r in mine]
        ax = fig.add_axes([0.06 + i * 0.32, 0.2, 0.25, 0.56])
        style_axes(ax)
        ax.plot(xs, bases, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2, label="Total length")
        draw_line(ax, xs, nodes, DEBRUIJN, "de Bruijn graph nodes", 5)
        draw_line(ax, xs, runs, NAIVE_BWT, "BWT runs", 5)
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_xlim(0.4, 16.6)
        ax.set_ylim(0, 17)
        ax.set_xlabel("Genomes in the collection", labelpad=8, color=INK2)
        if i == 0:
            ax.set_ylabel("Relative to one genome", labelpad=8, color=INK2)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), frameon=False, fontsize=9.5, labelcolor=INK2,
                      handlelength=2.2)
        panel_title(ax, COLLECTION_NAMES[name], f"at 16 genomes: nodes {nodes[-1]:.1f}×, runs {runs[-1]:.1f}×")
    save(fig, "3_growth_vs_bwt.png")


def accuracy_source() -> tuple[list[dict], str]:
    """The full-size pairs when the full run has been done, otherwise the
    small collection with the same divergence."""
    path = OUT / "accuracy_full.tsv"
    if path.exists():
        return read_table(path), f"16 genomes of 4.6 Mb ({FULL})"
    rows = [r for r in read_table(OUT / "accuracy.tsv") if r["collection"] == HIGH]
    return rows, f"16 genomes of 100 kb ({SMALL}/{HIGH})"


def plot_accuracy(plt, FuncFormatter) -> None:
    rows, source = accuracy_source()
    truth = np.array([float(r["true_distance"]) for r in rows])
    debruijn = np.array([float(r["debruijn_similarity"]) for r in rows])
    bwt = np.array([float(r["bwt_distance"]) for r in rows])
    fig = new_figure(plt, 14, 6.2, "Both indexes rank pairs of genomes by their true distance",
                     f"One point per pair, {len(rows)} pairs: {source}. True distance = fraction of sites that differ, "
                     f"from the simulator.\nde Bruijn similarity = shared {DEFAULT_K}-mers / all {DEFAULT_K}-mers of "
                     "the pair. BWT distance = (r(A,B) - max(r(A), r(B))) / min(r(A), r(B)), r = runs in the BWT.")
    panels = [(truth, debruijn, DEBRUIJN, "True distance", "de Bruijn similarity", percent, "de Bruijn graph",
               f"Spearman {spearman(truth, debruijn):.3f} (similarity falls as distance grows)"),
              (truth, bwt, NAIVE_BWT, "True distance", "BWT distance", percent, "Naive pairwise BWTs",
               f"Spearman {spearman(truth, bwt):.3f}"),
              (bwt, debruijn, INK2, "BWT distance", "de Bruijn similarity", plain, "Measure against measure",
               f"Spearman {spearman(bwt, debruijn):.3f}")]
    for i, (xs, ys, color, xlabel, ylabel, xformat, title, subtitle) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.32, 0.2, 0.25, 0.56])
        style_axes(ax)
        ax.scatter(xs, ys, s=40, color=color, edgecolors=SURFACE, linewidths=1.1, zorder=3)
        ax.set_xlim(0, xs.max() * 1.06)
        ax.set_ylim(0, 1.0 if ylabel.startswith("de Bruijn") else ys.max() * 1.08)
        ax.xaxis.set_major_formatter(FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.set_xlabel(xlabel, labelpad=8, color=INK2)
        ax.set_ylabel(ylabel, labelpad=8, color=INK2)
        panel_title(ax, title, subtitle)
    save(fig, "4_accuracy_vs_bwt.png")


def plot_scaling(plt, FuncFormatter, NullFormatter) -> None:
    rows = read_table(OUT / "scaling.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), facecolor=SURFACE, sharey="row")
    sweeps = [("size", "genome_length", "Genome length (bases), 8 genomes", "at 800k bases"),
              ("count", "num_genomes", "Number of genomes, 100 kb each", "at 32 genomes"),
              ("variety", None, "Divergence between genomes, 8 genomes of 400 kb", "for unrelated genomes")]
    fields = [("seconds", "Time", format_seconds), ("peak_bytes", "Peak working memory", format_bytes)]
    for r, (field, ylabel, yformat) in enumerate(fields):
        for c, (sweep, xfield, xlabel, where) in enumerate(sweeps):
            ax = axes[r][c]
            style_axes(ax)
            last = {}
            for method, (color, name) in METHODS.items():
                mine = [row for row in rows if row["sweep"] == sweep and row["method"] == method]
                if not mine:
                    continue
                xs = [float(p[xfield]) for p in mine] if xfield else list(range(len(mine)))
                ys = [float(p[field]) for p in mine]
                draw_line(ax, xs, ys, color, name)
                last[method] = ys[-1]
                labels = [p["label"] for p in mine]
            ax.set_yscale("log")
            if xfield:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.xaxis.set_major_formatter(FuncFormatter(format_count))
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
            parts = [f"{BRIEF[m]} {yformat(float(f'{last[m]:.2g}'))}" for m in METHODS if m in last]
            ax.text(0, 1.03, " · ".join(parts), transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK2)
            ax.margins(y=0.1)
    for c, title in enumerate(["Growing the genomes", "Adding genomes", "Making genomes more different"]):
        axes[0][c].set_title(title, loc="left", fontsize=11.5, color=INK, pad=24, fontweight="bold")
    handles = [plt.Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                          markeredgewidth=1.3, label=name) for color, name in METHODS.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.95), ncol=len(METHODS), frameon=False, fontsize=10,
               labelcolor=INK2, handlelength=2.2, columnspacing=2.5)
    fig.suptitle("Pairwise genome similarity: cost of the colored de Bruijn graph and the naive BWT method", x=0.06,
                 y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.012, f"Both axes are logarithmic except the divergence axis. k = {DEFAULT_K}. Working memory "
             "excludes the genome text. One run per point; independent mutated copies of E. coli K-12.\nThe line above "
             "each panel gives the values at its rightmost point. The graph is "
             "plain Python dictionaries of strings; the BWT is numpy arrays.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.975, top=0.835, bottom=0.125, wspace=0.09, hspace=0.24)
    save(fig, "5_scaling_vs_bwt.png")


def plot_full_scale(plt) -> None:
    path = OUT / "full_scale.tsv"
    if not path.exists():
        log("no full_scale.tsv yet: skipping the full-size figure (python -m experiments.debruijn_colored full)")
        return
    rows = {r["method"]: r for r in read_table(path)}
    names = [m for m in METHODS if m in rows]
    first = rows[names[0]]
    seconds = {m: float(rows[m]["seconds"]) for m in names}
    memory = {m: float(rows[m]["peak_rss_bytes"]) for m in names}
    fig = new_figure(plt, 12.5, 5.6, "Full size: time and memory of each method",
                     f"All 120 pairwise similarities of {first['genomes']} genomes, {int(first['bases']) / 1e6:.0f} "
                     f"million bases ({first['collection']}). Host {first['host']}, one core, one run each.\nMemory is "
                     f"the peak of the whole process. k = {DEFAULT_K}. The graph is plain Python dictionaries of "
                     "strings; the BWT is numpy arrays.")
    panels = [(seconds, "Time", "{:.0f} s"), (memory, "Peak memory", "{:.1f} GB")]
    for i, (values, title, pattern) in enumerate(panels):
        ax = fig.add_axes([0.3 + i * 0.37, 0.24, 0.25, 0.5])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(length=0)
        top = max(values.values())
        for y, method in enumerate(names):
            value = values[method]
            ax.barh([y], [value], height=0.56, color=METHODS[method][0], zorder=3)
            shown = value / 1e9 if title == "Peak memory" else value
            ax.text(value + top * 0.02, y, pattern.format(shown), va="center", fontsize=10.5, color=INK)
        ax.set_xlim(0, top * 1.3)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks(range(len(names)), [METHODS[m][1] for m in names] if i == 0 else [""] * len(names),
                      fontsize=10.5, color=INK2)
        ordered = sorted(values.values())
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=26, fontweight="bold")
        winner = SHORT[min(values, key=values.get)]
        ax.text(0, 1.04, f"{winner} lowest, {ordered[1] / ordered[0]:.1f}× below the next", transform=ax.transAxes,
                fontsize=10, color=INK2)
    save(fig, "6_full_scale_vs_bwt.png")


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    plot_k_sweep(plt, FuncFormatter, NullFormatter, LogLocator)
    plot_spectrum(plt, FuncFormatter)
    plot_growth(plt)
    plot_accuracy(plt, FuncFormatter)
    plot_scaling(plt, FuncFormatter, NullFormatter)
    plot_full_scale(plt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.debruijn_colored", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "full", "plot"])
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
    if args.command == "full":
        full()
    if args.command in ("run", "plot"):
        plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
