"""Figures for the midpoint presentation.

    python -m experiments.midpoint_figures run     # measure (about 8 minutes, needs ~20 GB for the real data)
    python -m experiments.midpoint_figures plot    # draw every figure from saved results

run   For each full-size collection: build the BWT with SA-IS, compute the
      pairwise similarity (written to results/bwt_similarity_sais/), and count
      the runs of the BWT of the first k genomes for every k. Timings and run
      counts go to results/midpoint_figures/.

plot  Writes these files to results/midpoint_figures/:

      1_synthetic_tree.png        the tree the synthetic genomes were simulated on
      2_distance_vs_truth.png     BWT distance against true distance, every pair
      3_synthetic_heatmaps.png    true and BWT distance matrices side by side
      4_real_ecoli_heatmap.png    50 real E. coli genomes, clustered by BWT distance
      5_run_growth.png            how the number of BWT runs grows as genomes are added
      6_method_times.png          time of each way of computing the similarity matrix
      7_scaling.png               copied from results/bwt_similarity_scaling/
      8_parallel.png              copied from results/bwt_similarity_parallel_comparison/

      Figure 6 also needs the measurements of experiments.bwt_similarity_parallel
      for the same collection, taken on the same machine.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from pangenome.bwt.build_sais import build_bwt_sais
from pangenome.bwt.similarity import measures, output_dir, pairwise_runs, read_genomes, spearman, write_results

OUT = Path("results/midpoint_figures")
SIMILARITY = "results/bwt_similarity_sais"
LOW = "data/synthetic/uneven_d0.001"
HIGH = "data/synthetic/uneven_d0.01"
CONTROL = "data/synthetic/control"
REAL = "data/real/ecoli"
COLLECTIONS = [LOW, HIGH, CONTROL, REAL]
PARALLEL = Path("results/bwt_similarity_parallel_comparison/synthetic_uneven_d0.01_n016")
SCALING = Path("results/bwt_similarity_scaling/scaling.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f3f7fc"]  # near to far


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

@njit(cache=True)
def _prefix_runs(bwt, doc, d):
    """runs[k] = runs in the BWT of the first k genomes, for k = 1..d.

    That BWT is the full one with the rows of genomes k, k+1, ... removed, so
    a row of genome g belongs to every prefix collection with k > g.
    """
    runs = np.zeros(d + 1, dtype=np.int64)
    last = np.full(d + 1, 255, dtype=np.uint8)
    for i in range(len(bwt)):
        c = bwt[i]
        for k in range(doc[i] + 1, d + 1):
            if last[k] != c:
                runs[k] += 1
                last[k] = c
    return runs


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_growth.tsv", "w") as growth, open(OUT / "timings.tsv", "w") as timings:
        growth.write("collection\tgenomes\tcharacters\truns\n")
        timings.write("collection\tgenomes\tcharacters\tbuild_seconds\tsweep_seconds\thost\n")
        for collection in COLLECTIONS:
            genomes = read_genomes(collection, None)
            d = len(genomes)
            start = time.perf_counter()
            bwt, doc = build_bwt_sais(genomes)
            built = time.perf_counter()
            runs = pairwise_runs(bwt, doc, d)
            swept = time.perf_counter()
            res = measures(runs, genomes)
            write_results(output_dir(None, SIMILARITY, collection, d), res)
            timings.write(f"{collection}\t{d}\t{len(bwt)}\t{built - start:.3f}\t{swept - built:.3f}\t{platform.node()}\n")
            timings.flush()

            prefix = _prefix_runs(bwt, doc.astype(np.int64), d)
            characters = 0
            for k in range(1, d + 1):
                characters += len(genomes[k - 1]) + 1
                growth.write(f"{collection}\t{k}\t{characters}\t{prefix[k]}\n")
            growth.flush()
            log(f"{collection}: {d} genomes, {len(bwt)} characters, build {built - start:.1f} s, "
                f"sweep {swept - built:.1f} s, r = {prefix[d]}")


# --------------------------------------------------------------------------
# reading results
# --------------------------------------------------------------------------

def read_table(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def read_matrix(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()[1:]
    return np.array([[float(v) for v in line.split("\t")[1:]] for line in lines])


def similarity_dir(collection: str, d: int) -> Path:
    return output_dir(None, SIMILARITY, collection, d)


def true_matrix(collection: str, d: int) -> np.ndarray:
    truth = np.zeros((d, d))
    for row in read_table(Path(collection) / "pairwise.tsv"):
        a, b = int(row["genome_a"][1:]), int(row["genome_b"][1:])
        truth[a, b] = truth[b, a] = float(row["distance"])
    return truth


# --------------------------------------------------------------------------
# trees
# --------------------------------------------------------------------------

def _parse_node(text: str, i: int) -> tuple[dict, int]:
    children = []
    if text[i] == "(":
        i += 1
        while True:
            child, i = _parse_node(text, i)
            children.append(child)
            if text[i] == ",":
                i += 1
            else:
                i += 1  # the closing bracket
                break
    j = i
    while text[j] not in ":,);":
        j += 1
    name = text[i:j]
    length = 0.0
    if text[j] == ":":
        i = j + 1
        j = i
        while text[j] not in ",);":
            j += 1
        length = float(text[i:j])
    return {"name": name, "length": length, "children": children}, j


def parse_newick(text: str) -> dict:
    return _parse_node(text.strip(), 0)[0]


def upgma(dist: np.ndarray) -> dict:
    """Average-linkage clustering of a distance matrix, as a tree whose leaves
    are named by row index."""
    d = len(dist)
    size = 2 * d - 1
    table = np.full((size, size), np.inf)
    table[:d, :d] = dist
    np.fill_diagonal(table, np.inf)
    nodes = [{"name": str(i), "length": 0.0, "children": []} for i in range(d)]
    height = [0.0] * d
    members = [1] * d
    active = list(range(d))
    for new in range(d, size):
        sub = table[np.ix_(active, active)]
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = active[a], active[b]
        top = table[a, b] / 2
        nodes[a]["length"] = top - height[a]
        nodes[b]["length"] = top - height[b]
        active.remove(a)
        active.remove(b)
        for c in active:
            table[new, c] = table[c, new] = (members[a] * table[a, c] + members[b] * table[b, c]) / (members[a] + members[b])
        nodes.append({"name": "", "length": 0.0, "children": [nodes[a], nodes[b]]})
        height.append(top)
        members.append(members[a] + members[b])
        active.append(new)
    return nodes[-1]


def place(node: dict, x: float, leaves: list, segments: list) -> float:
    """Lay a tree out left to right. Appends (name, x, y) per leaf, in order
    from the top, and the line segments to draw. Returns the node's y."""
    end = x + node["length"]
    if node["children"]:
        ys = [place(child, end, leaves, segments) for child in node["children"]]
        y = sum(ys) / len(ys)
        segments.append([(end, min(ys)), (end, max(ys))])
    else:
        y = float(len(leaves))
        leaves.append((node["name"], end, y))
    segments.append([(x, y), (end, y)])
    return y


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def percent(value: float, _pos: object = None) -> str:
    return f"{value * 100:g}%"


def plain(value: float, _pos: object = None) -> str:
    return f"{value:g}"


def new_figure(plt, width: float, height: float, title: str, footnote: str):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    fig.suptitle(title, x=0.05, y=0.965, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.05, 0.022, footnote, fontsize=8.5, color=MUTED)
    return fig


def style_axes(ax, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid == "y":
        ax.spines["left"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=26, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    log(f"wrote {OUT / name}")


def plot_tree(plt, LineCollection) -> None:
    tree = parse_newick((Path(HIGH) / "tree.nwk").read_text())
    leaves, segments = [], []
    place(tree, 0.0, leaves, segments)
    depth = max(x for _name, x, _y in leaves)

    fig = new_figure(plt, 11, 6.2, "Synthetic collection: 16 genomes simulated on an uneven tree",
                     "Every genome is E. coli K-12 with the mutations along its path from the root: 80% substitutions, "
                     "10% insertions, 10% deletions.\nBranch length = mutation rate per site. "
                     f"Collection {HIGH}: 1% root-to-leaf on average, each branch scaled by a random factor in [0.2, 1.8].")
    ax = fig.add_axes([0.07, 0.17, 0.86, 0.68])
    ax.set_facecolor(SURFACE)
    ax.add_collection(LineCollection(segments, colors=INK2, linewidths=2, capstyle="round"))
    for name, x, y in leaves:
        ax.plot([x], [y], marker="o", markersize=6, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(x + depth * 0.012, y, name, va="center", fontsize=10, color=INK2)
    root_y = segments[-1][0][1]  # the last segment added is the root's own branch
    ax.plot([0], [root_y], marker="o", markersize=6, color=INK2, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.text(depth * 0.012, root_y, "E. coli K-12", ha="left", va="center", fontsize=10, color=INK2)
    bar = 0.002
    ax.plot([0, bar], [len(leaves) + 0.2] * 2, color=INK, linewidth=2, solid_capstyle="butt")
    ax.text(bar / 2, len(leaves) + 0.75, "0.2% of sites mutated", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.set_xlim(-depth * 0.02, depth * 1.08)
    ax.set_ylim(len(leaves) + 1.2, -1.5)
    ax.axis("off")
    save(fig, "1_synthetic_tree.png")


def plot_distance_vs_truth(plt, FuncFormatter) -> None:
    fig = new_figure(plt, 12, 6.2, "BWT run-count distance tracks the true distance between genomes",
                     "One point per pair of genomes (120 pairs per collection). True distance = fraction of sites that "
                     "differ, from the simulator's variant log.\nBWT distance = (r(A,B) - max(r(A), r(B))) / "
                     "min(r(A), r(B)), where r is the number of runs in the BWT.")
    panels = [(LOW, "Closely related (0.1% root-to-leaf)", [0.07, 0.2, 0.4, 0.56]),
              (HIGH, "More diverged (1% root-to-leaf)", [0.56, 0.2, 0.4, 0.56])]
    for collection, title, box in panels:
        d = 16
        truth = true_matrix(collection, d)
        got = read_matrix(similarity_dir(collection, d) / "distance_matrix.tsv")
        iu = np.triu_indices(d, 1)
        ax = fig.add_axes(box)
        style_axes(ax)
        ax.scatter(truth[iu], got[iu], s=42, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.set_xlim(0, truth[iu].max() * 1.06)
        ax.set_ylim(0, got[iu].max() * 1.08)
        ax.xaxis.set_major_formatter(FuncFormatter(percent))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.set_xlabel("True distance (sites that differ)", labelpad=8, color=INK2)
        ax.set_ylabel("BWT distance", labelpad=8, color=INK2)
        nearest = nearest_right(got, truth)
        panel_title(ax, title, f"Spearman correlation {spearman(got[iu], truth[iu]):.3f}; "
                    f"nearest neighbour right for {nearest} of {d}")
    save(fig, "2_distance_vs_truth.png")


def nearest_right(got: np.ndarray, truth: np.ndarray) -> int:
    """Genomes whose nearest neighbour by `got` is (one of) the truly nearest."""
    d = len(got)
    g, t = got.copy(), truth.copy()
    np.fill_diagonal(g, np.inf)
    np.fill_diagonal(t, np.inf)
    best = g.argmin(axis=1)
    return int(sum(t[a, best[a]] <= t[a].min() * (1 + 1e-9) for a in range(d)))


def heatmap(fig, ax, matrix: np.ndarray, cmap, label: str, formatter, bar_box: list[float]) -> None:
    off = matrix[~np.eye(len(matrix), dtype=bool)]
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=off.max(), interpolation="nearest")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    bar_ax = fig.add_axes(bar_box)
    bar = fig.colorbar(image, cax=bar_ax, orientation="horizontal", format=formatter)
    bar.outline.set_visible(False)
    bar_ax.tick_params(length=0, labelsize=9, labelcolor=INK2)
    bar_ax.set_xlabel(label, fontsize=9.5, color=INK2, labelpad=6)


def plot_synthetic_heatmaps(plt, FuncFormatter, cmap) -> None:
    d = 16
    truth = true_matrix(HIGH, d)
    got = read_matrix(similarity_dir(HIGH, d) / "distance_matrix.tsv")
    names = [f"g{i:03d}" for i in range(d)]
    fig = new_figure(plt, 12, 7.2, "The BWT distance matrix reproduces the structure of the tree",
                     f"Collection {HIGH}, genomes in tree order. Darker = more similar. Each matrix has its own "
                     "colour scale; the diagonal is zero.")
    panels = [(truth, "True distance", "from the simulator's variant log", percent, "sites that differ", 0.07),
              (got, "BWT run-count distance", "computed from the genomes alone", plain, "BWT distance", 0.55)]
    for matrix, title, subtitle, formatter, label, left in panels:
        ax = fig.add_axes([left, 0.25, 0.39, 0.6])
        heatmap(fig, ax, matrix, cmap, label, FuncFormatter(formatter), [left + 0.085, 0.135, 0.22, 0.016])
        ax.set_xticks(range(d), names, rotation=90, fontsize=8, color=INK2)
        ax.set_yticks(range(d), names, fontsize=8, color=INK2)
        panel_title(ax, title, subtitle)
    save(fig, "3_synthetic_heatmaps.png")


def plot_real_heatmap(plt, FuncFormatter, LineCollection, cmap) -> None:
    info = read_table(Path(REAL) / "genomes.tsv")
    d = len(info)
    got = read_matrix(similarity_dir(REAL, d) / "distance_matrix.tsv")
    leaves, segments = [], []
    place(upgma(got), 0.0, leaves, segments)
    order = [int(name) for name, _x, _y in leaves]
    labels = []
    for i in order:
        strain = info[i]["strain"]
        labels.append(strain if strain != "-" else info[i]["accession"])
    iu = np.triu_indices(d, 1)

    fig = new_figure(plt, 11, 10.2, f"{d} real E. coli genomes, clustered by BWT run-count distance",
                     f"Complete RefSeq assemblies ({REAL}), a seeded random sample. Rows and columns ordered by "
                     "average-linkage clustering of the BWT distance.\nDarker = more similar. Pairwise distance ranges "
                     f"from {got[iu].min():.3f} to {got[iu].max():.2f}; there is no ground truth for these genomes.")
    tree_ax = fig.add_axes([0.05, 0.245, 0.15, 0.655])
    tree_ax.add_collection(LineCollection(segments, colors=INK2, linewidths=1.2, capstyle="round"))
    tree_ax.set_xlim(0, max(x for _n, x, _y in leaves))
    tree_ax.set_ylim(d - 0.5, -0.5)
    tree_ax.axis("off")
    ax = fig.add_axes([0.2, 0.245, 0.6, 0.655])
    heatmap(fig, ax, got[np.ix_(order, order)], cmap, "BWT distance", FuncFormatter(plain), [0.39, 0.115, 0.22, 0.01])
    ax.set_aspect("auto")
    ax.set_xticks(range(d), labels, rotation=90, fontsize=6.5, color=INK2)
    ax.yaxis.tick_right()
    ax.set_yticks(range(d), labels, fontsize=6.5, color=INK2)
    save(fig, "4_real_ecoli_heatmap.png")


def growth_series(rows: list[dict], collection: str) -> tuple[list[int], list[float], list[float]]:
    """(k, r(k) / r(1), n(k) / n(1)) for the first k genomes of a collection."""
    mine = [r for r in rows if r["collection"] == collection]
    k = [int(r["genomes"]) for r in mine]
    runs = [int(r["runs"]) / int(mine[0]["runs"]) for r in mine]
    chars = [int(r["characters"]) / int(mine[0]["characters"]) for r in mine]
    return k, runs, chars


def plot_run_growth(plt) -> None:
    rows = read_table(OUT / "run_growth.tsv")
    fig = new_figure(plt, 12.5, 6.4, "Adding related genomes adds few BWT runs",
                     "r = runs in the BWT of the first k genomes, divided by r of the first genome alone. The dashed "
                     "line is the text length n on the same scale.\nA run-length-compressed index takes space "
                     "proportional to r, not n.")

    left = fig.add_axes([0.07, 0.2, 0.4, 0.56])
    style_axes(left)
    k, _runs, chars = growth_series(rows, HIGH)
    left.plot(k, chars, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2, label="Text length n")
    series = [(CONTROL, "Unrelated random genomes", AQUA), (HIGH, "1% root-to-leaf divergence", ORANGE),
              (LOW, "0.1% root-to-leaf divergence", BLUE)]
    ends = []
    for collection, label, color in series:
        k, runs, _chars = growth_series(rows, collection)
        left.plot(k, runs, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                  markeredgewidth=1.2, zorder=3, label=label)
        left.annotate(f"{runs[-1]:.1f}×", xy=(k[-1], runs[-1]), xytext=(8, 0), textcoords="offset points",
                      va="center", fontsize=10, color=INK2)
        ends.append(runs[-1])
    left.set_xticks([1, 4, 8, 12, 16])
    left.set_xlim(0.4, 17.6)
    left.set_ylim(0, 17)
    left.set_xlabel("Genomes in the collection (k)", labelpad=8, color=INK2)
    left.set_ylabel("Relative to one genome", labelpad=8, color=INK2)
    left.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=2.2)
    panel_title(left, "Synthetic genomes", f"16 genomes: n grows 16×, r grows {ends[2]:.1f}× to {ends[1]:.1f}× "
                "when the genomes are related")

    right = fig.add_axes([0.56, 0.2, 0.4, 0.56])
    style_axes(right)
    k, runs, chars = growth_series(rows, REAL)
    right.plot(k, chars, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2, label="Text length n")
    right.plot(k, runs, color=BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE,
               markeredgewidth=1, zorder=3, label="BWT runs r")
    right.annotate(f"{runs[-1]:.1f}×", xy=(k[-1], runs[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                   fontsize=10, color=INK2)
    right.annotate(f"{chars[-1]:.1f}×", xy=(k[-1], chars[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                   fontsize=10, color=INK2)
    right.set_xticks([1, 10, 20, 30, 40, 50])
    right.set_xlim(-1, len(k) + 5)
    right.set_ylim(0, max(chars) * 1.06)
    right.set_xlabel("Genomes in the collection (k)", labelpad=8, color=INK2)
    right.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=2.2)
    panel_title(right, "Real E. coli genomes", f"{len(k)} genomes: n grows {chars[-1]:.0f}×, r grows {runs[-1]:.1f}×")
    save(fig, "5_run_growth.png")


def plot_method_times(plt) -> None:
    measured = read_table(PARALLEL / "measurements.tsv")
    machine = dict(line.split("\t") for line in (PARALLEL / "machine.txt").read_text().splitlines())
    sais = [r for r in read_table(OUT / "timings.tsv") if r["collection"] == HIGH][0]
    if sais["host"] != machine["host"]:
        log(f"warning: SA-IS was timed on {sais['host']} but the other methods on {machine['host']}")
    naive = [float(r["seconds"]) for r in measured if r["method"] == "naive"][0]
    single = [float(r["seconds"]) for r in measured if r["method"] == "single"][0]
    parallel = [(float(r["seconds"]), int(r["workers"])) for r in measured if r["method"] == "naive_parallel"]
    best_seconds, best_workers = min(parallel)
    sais_seconds = float(sais["build_seconds"]) + float(sais["sweep_seconds"])
    bars = [("A BWT per pair of genomes\n1 core", naive),
            (f"A BWT per pair of genomes\n{best_workers} cores (best)", best_seconds),
            ("One BWT, one sweep\nprefix doubling, 1 core", single),
            ("One BWT, one sweep\nSA-IS, 1 core", sais_seconds)]

    fig = new_figure(plt, 11, 5.8,
                     f"One BWT for the whole collection, built in linear time, is {naive / sais_seconds:.0f}× faster",
                     f"All 120 pairwise similarities of {machine['genomes']} genomes, "
                     f"{int(machine['characters']) / 1e6:.0f} million characters ({HIGH}). Host {machine['host']}, "
                     f"{machine['cores']} cores. One run each.\nEvery method produces the same matrix of run counts. "
                     "Python with numpy and numba.")
    ax = fig.add_axes([0.27, 0.2, 0.64, 0.62])
    style_axes(ax, grid="")
    ax.spines["bottom"].set_visible(False)
    positions = list(range(len(bars)))
    values = [seconds for _label, seconds in bars]
    ax.barh(positions, values, height=0.56, color=BLUE, zorder=3)
    for y, seconds in zip(positions, values):
        text = f"{seconds:.0f} s" if seconds < 600 else f"{seconds:.0f} s ({seconds / 60:.0f} min)"
        ax.text(seconds + max(values) * 0.012, y, text, va="center", fontsize=10.5, color=INK)
    ax.set_yticks(positions, [label for label, _seconds in bars], fontsize=10.5, color=INK2)
    ax.set_xticks([])
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.spines["left"].set_color(AXIS)
    save(fig, "6_method_times.png")


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    cmap = LinearSegmentedColormap.from_list("near_to_far", BLUE_RAMP)
    OUT.mkdir(parents=True, exist_ok=True)

    plot_tree(plt, LineCollection)
    plot_distance_vs_truth(plt, FuncFormatter)
    plot_synthetic_heatmaps(plt, FuncFormatter, cmap)
    plot_real_heatmap(plt, FuncFormatter, LineCollection, cmap)
    plot_run_growth(plt)
    plot_method_times(plt)
    for source, name in ((SCALING, "7_scaling.png"), (PARALLEL / "parallel.png", "8_parallel.png")):
        if source.exists():
            shutil.copyfile(source, OUT / name)
            log(f"copied {source} to {OUT / name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.midpoint_figures", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "plot"])
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
