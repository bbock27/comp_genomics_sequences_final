"""Simulate a collection of near-identical genomes with a ground-truth variant log.

Model
-----
Every copy is mutated independently from the same reference (a star
phylogeny). Walking along the reference, each site starts a variant with
probability ``divergence``. A variant is a SNP, a short insertion, or a short
deletion (default mix 80/10/10). Indel lengths are geometric with mean 2,
capped at ``max_indel``. At least one unchanged reference base separates
consecutive variants of the same copy, so the log is unambiguous.

Two copies at divergence d therefore differ from each other at about 2d of
their sites.

Tree mode
---------
In the star model every pair of genomes is equally distant, so "which genomes
are most similar" has no answer. :func:`simulate_tree_collection` instead
places the genomes at the leaves of a balanced binary tree of a fixed height
H. Each branch carries variants at rate divergence / H, and a genome has the
variants of every branch between the root (the reference) and its leaf.
Genomes i and j share the branches above their lowest common ancestor, so
siblings (g000, g001) differ at about 2d/H of their sites, cousins at 4d/H,
and the most distant pairs at 2d. Every genome is still at divergence d from
the reference.

With ``variation`` v > 0 the branches have unequal lengths: each branch's rate
is divergence / H times a factor drawn uniformly from [1 - v, 1 + v]. Genomes
then sit at different distances from the reference, and a genome's nearest
neighbour need not be its sibling. The average is unchanged.

Coordinates are 0-based throughout.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .seq import ALPHABET, random_sequence, read_fasta, rng_for

SNP, INS, DEL = "SNP", "INS", "DEL"


@dataclass(frozen=True)
class Variant:
    """One difference between a copy and the reference.

    ref_pos   reference position of the first affected base. For an insertion,
              the inserted bases go immediately *before* reference base ref_pos.
    ref, alt  reference allele and copy allele ("" for the empty side of an indel)
    copy_pos  position in the copy where ``alt`` starts. For a deletion, the
              position of the first base after the deleted stretch.
    """

    genome: int
    ref_pos: int
    kind: str
    ref: str
    alt: str
    copy_pos: int

    @property
    def ref_end(self) -> int:
        return self.ref_pos + len(self.ref)

    @property
    def copy_end(self) -> int:
        return self.copy_pos + len(self.alt)


@dataclass(frozen=True)
class MutationModel:
    snp_frac: float = 0.8
    ins_frac: float = 0.1
    del_frac: float = 0.1
    max_indel: int = 10

    def __post_init__(self) -> None:
        total = self.snp_frac + self.ins_frac + self.del_frac
        if min(self.snp_frac, self.ins_frac, self.del_frac) < 0 or abs(total - 1.0) > 1e-9:
            raise ValueError("snp/ins/del fractions must be non-negative and sum to 1")
        if self.max_indel < 1:
            raise ValueError("max_indel must be at least 1")


def _by_position(v: Variant) -> tuple[int, int]:
    """Sort key: variants in reference order."""
    return v.ref_pos, v.ref_end


def _indel_length(rng: random.Random, max_indel: int) -> int:
    n = 1
    while n < max_indel and rng.random() < 0.5:
        n += 1
    return n


def mutate(
    reference: str,
    rng: random.Random,
    divergence: float,
    model: MutationModel = MutationModel(),
    genome: int = 0,
) -> tuple[str, list[Variant]]:
    """Return a mutated copy of ``reference`` and the list of variants applied."""
    if not 0.0 <= divergence < 1.0:
        raise ValueError("divergence must be in [0, 1)")
    if divergence == 0.0 or not reference:
        return reference, []

    length = len(reference)
    log_keep = math.log1p(-divergence)
    pieces: list[str] = []
    variants: list[Variant] = []
    last = 0  # reference bases before `last` are already emitted or deleted
    shift = 0  # copy coordinate minus reference coordinate, so far
    pos = 0

    while True:
        # Number of unchanged sites before the next variant is geometric.
        pos += int(math.log1p(-rng.random()) / log_keep)
        if pos >= length:
            break
        x = rng.random()
        if x < model.snp_frac:
            ref = reference[pos]
            alt = rng.choice([b for b in ALPHABET if b != ref])
            pieces += [reference[last:pos], alt]
            variants.append(Variant(genome, pos, SNP, ref, alt, pos + shift))
            last = pos + 1
        elif x < model.snp_frac + model.ins_frac:
            alt = random_sequence(rng, _indel_length(rng, model.max_indel))
            pieces += [reference[last:pos], alt]
            variants.append(Variant(genome, pos, INS, "", alt, pos + shift))
            shift += len(alt)
            last = pos
        else:
            n = min(_indel_length(rng, model.max_indel), length - pos)
            pieces.append(reference[last:pos])
            variants.append(Variant(genome, pos, DEL, reference[pos : pos + n], "", pos + shift))
            shift -= n
            last = pos + n
        # Keep one unchanged reference base between consecutive variants.
        pos = variants[-1].ref_end + 1

    pieces.append(reference[last:])
    return "".join(pieces), variants


def apply_variants(reference: str, variants: list[Variant]) -> str:
    """Rebuild a copy from the reference and its variant log.

    Deliberately independent of :func:`mutate`, so tests can replay a log and
    check that it reproduces the simulated genome exactly.
    """
    out: list[str] = []
    last = 0
    for v in sorted(variants, key=_by_position):
        if v.ref_pos < last:
            raise ValueError(f"overlapping variants at reference position {v.ref_pos}")
        if reference[v.ref_pos : v.ref_end] != v.ref:
            raise ValueError(f"reference allele mismatch at position {v.ref_pos}")
        out += [reference[last : v.ref_pos], v.alt]
        last = v.ref_end
    out.append(reference[last:])
    return "".join(out)


class CoordMap:
    """Convert between reference and copy coordinates for one genome."""

    def __init__(self, variants: list[Variant]):
        self.variants = sorted(variants, key=_by_position)
        self._ref_ends = [v.ref_end for v in self.variants]
        self._copy_ends = [v.copy_end for v in self.variants]
        self._shifts: list[int] = []
        shift = 0
        for v in self.variants:
            shift += len(v.alt) - len(v.ref)
            self._shifts.append(shift)

    def ref_to_copy(self, p: int) -> int:
        """Copy position of reference base p.

        A deleted base maps to the first copy base after the deletion.
        """
        i = bisect.bisect_right(self._ref_ends, p)
        if i < len(self.variants) and self.variants[i].ref_pos <= p:
            return self.variants[i].copy_pos
        return p + (self._shifts[i - 1] if i else 0)

    def copy_to_ref(self, c: int) -> int:
        """Reference position of copy base c.

        An inserted base maps to the reference base that follows the insertion.
        """
        i = bisect.bisect_right(self._copy_ends, c)
        if i < len(self.variants) and self.variants[i].copy_pos <= c:
            return self.variants[i].ref_pos
        return c - (self._shifts[i - 1] if i else 0)


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------

def genome_name(i: int) -> str:
    return f"g{i:03d}"


def simulate_collection(
    reference: str,
    copies: int,
    divergence: float,
    seed: int,
    model: MutationModel = MutationModel(),
) -> tuple[list[str], list[list[Variant]]]:
    """Simulate ``copies`` mutated copies of the reference.

    Genome i depends only on (seed, divergence, i), so the first n genomes of a
    larger collection are exactly the n-genome collection.
    """
    genomes, logs = [], []
    for i in range(copies):
        copy, variants = mutate(reference, rng_for(seed, "genome", divergence, i), divergence, model, i)
        genomes.append(copy)
        logs.append(variants)
    return genomes, logs


def _conflicts(v: Variant, starts: list[int], accepted: list[Variant]) -> bool:
    """True if v overlaps or touches a variant in the sorted list ``accepted``."""
    i = bisect.bisect_left(starts, v.ref_pos)
    if i > 0 and v.ref_pos < accepted[i - 1].ref_end + 1:
        return True
    return i < len(accepted) and accepted[i].ref_pos < v.ref_end + 1


def tree_branches(i: int, height: int) -> list[tuple[int, int]]:
    """Branches (level, node) from the root down to leaf i.

    Level 0 is the leaf's own branch; the branch at level h is shared by the
    2**h genomes with the same value of i >> h.
    """
    return [(h, i >> h) for h in range(height - 1, -1, -1)]


def branch_rate(seed: int, divergence: float, height: int, level: int, node: int, variation: float = 0.0) -> float:
    """Per-site variant rate on one branch of the tree (its length)."""
    rate = divergence / height
    if variation:
        rate *= rng_for(seed, "branch", divergence, height, level, node).uniform(1 - variation, 1 + variation)
    return rate


def simulate_tree_collection(
    reference: str,
    copies: int,
    divergence: float,
    seed: int,
    model: MutationModel = MutationModel(),
    height: int = 4,
    variation: float = 0.0,
) -> tuple[list[str], list[list[Variant]]]:
    """Simulate genomes related by a balanced binary tree (see module docstring).

    Genome i depends only on (seed, divergence, height, variation, i), so collections are
    nested exactly as in the star model. Genomes 0 .. 2**height - 1 form one
    tree; later genomes start further trees that share nothing with the first.

    All variants are placed in reference coordinates. When a variant on a lower
    branch would overlap or touch one inherited from above, the lower one is
    dropped, so every log stays unambiguous and shared variants stay identical
    in every genome that inherits them.
    """
    if height < 1:
        raise ValueError("tree height must be at least 1")
    if not 0.0 <= variation < 1.0:
        raise ValueError("branch variation must be in [0, 1)")
    node_cache: dict[tuple[int, int], list[Variant]] = {}

    def node_variants(level: int, node: int) -> list[Variant]:
        """Variants accumulated from the root down to this node, sorted."""
        if level == height:
            return []
        key = (level, node)
        if key not in node_cache:
            inherited = node_variants(level + 1, node >> 1)
            rng = rng_for(seed, "tree", divergence, height, level, node)
            _, own = mutate(reference, rng, branch_rate(seed, divergence, height, level, node, variation), model)
            starts = [v.ref_pos for v in inherited]
            kept = [v for v in own if not _conflicts(v, starts, inherited)]
            node_cache[key] = sorted(inherited + kept, key=_by_position)
        return node_cache[key]

    genomes, logs = [], []
    for i in range(copies):
        shift, log = 0, []
        for v in node_variants(0, i):
            log.append(Variant(i, v.ref_pos, v.kind, v.ref, v.alt, v.ref_pos + shift))
            shift += len(v.alt) - len(v.ref)
        genomes.append(apply_variants(reference, log))
        logs.append(log)
        if i % 2 == 1:  # both leaves of this pair are done; free their lists
            node_cache.pop((0, i), None)
            node_cache.pop((0, i - 1), None)
    return genomes, logs


def newick(copies: int, divergence: float, height: int, seed: int = 0, variation: float = 0.0) -> str:
    """The tree relating the first ``copies`` genomes, with branch lengths."""

    def above(level: int, node: int) -> float:
        """Length of the branch above a node (the top of a tree has none)."""
        return branch_rate(seed, divergence, height, level, node, variation) if level < height else 0.0

    def subtree(level: int, node: int) -> tuple[str, float] | None:
        """(newick text, length of the branch above it), or None if empty."""
        if level == 0:
            return (genome_name(node), above(0, node)) if node < copies else None
        kids = [k for k in (subtree(level - 1, 2 * node), subtree(level - 1, 2 * node + 1)) if k]
        if not kids:
            return None
        if len(kids) == 1:  # a lone child absorbs its parent's branch
            return kids[0][0], kids[0][1] + above(level, node)
        text = "(" + ",".join(f"{t}:{length:.6g}" for t, length in kids) + ")"
        return text, above(level, node)

    tops = [t for t in (subtree(height, r) for r in range((copies - 1 >> height) + 1)) if t]
    if len(tops) == 1 and tops[0][0].startswith("("):
        return tops[0][0] + ";"
    return "(" + ",".join(f"{t}:{length:.6g}" for t, length in tops) + ");"


def pairwise_truth(logs: list[list[Variant]], reference_length: int) -> list[dict]:
    """Ground truth for genome comparison, from the variant logs.

    For each pair: the number of variants both genomes carry, the number only
    one carries, and distance = (only_a + only_b) / reference length.
    """
    keys = [{(v.ref_pos, v.ref, v.alt) for v in log} for log in logs]
    rows = []
    for a in range(len(logs)):
        for b in range(a + 1, len(logs)):
            shared = len(keys[a] & keys[b])
            only_a, only_b = len(keys[a]) - shared, len(keys[b]) - shared
            rows.append({"genome_a": a, "genome_b": b, "shared_variants": shared, "only_a": only_a,
                         "only_b": only_b, "distance": (only_a + only_b) / reference_length})
    return rows


def write_pairwise(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        fh.write("genome_a\tgenome_b\tshared_variants\tonly_a\tonly_b\tdistance\n")
        for r in rows:
            fh.write(f"{genome_name(r['genome_a'])}\t{genome_name(r['genome_b'])}\t{r['shared_variants']}\t"
                     f"{r['only_a']}\t{r['only_b']}\t{r['distance']:.6g}\n")


def simulate_control(length: int, copies: int, seed: int) -> list[str]:
    """Unrelated uniformly random genomes: the non-repetitive control."""
    return [random_sequence(rng_for(seed, "control", i), length) for i in range(copies)]


VARIANT_COLUMNS = ["genome", "ref_pos", "type", "ref", "alt", "copy_pos"]


def write_variants(path: str | Path, logs: list[list[Variant]]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(VARIANT_COLUMNS) + "\n")
        for variants in logs:
            for v in variants:
                fh.write(
                    f"{genome_name(v.genome)}\t{v.ref_pos}\t{v.kind}\t{v.ref or '-'}\t{v.alt or '-'}\t{v.copy_pos}\n"
                )


def read_variants(path: str | Path, copies: int) -> list[list[Variant]]:
    logs: list[list[Variant]] = [[] for _ in range(copies)]
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != VARIANT_COLUMNS:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in fh:
            g, ref_pos, kind, ref, alt, copy_pos = line.rstrip("\n").split("\t")
            gi = int(g[1:])
            if gi < copies:
                logs[gi].append(
                    Variant(gi, int(ref_pos), kind, "" if ref == "-" else ref, "" if alt == "-" else alt, int(copy_pos))
                )
    return logs


@dataclass
class Collection:
    genomes: list[str]
    logs: list[list[Variant]]  # empty lists for the control (no variant log)
    manifest: dict

    @property
    def has_variants(self) -> bool:
        return any(self.logs)


def load_collection(directory: str | Path, num_genomes: int | None = None) -> Collection:
    """Load the first ``num_genomes`` genomes (default: all) of a collection."""
    directory = Path(directory)
    records = read_fasta(directory / "genomes.fasta")
    if num_genomes is not None:
        if num_genomes > len(records):
            raise ValueError(f"{directory} has only {len(records)} genomes, asked for {num_genomes}")
        records = records[:num_genomes]
    genomes = [seq for _, seq in records]
    vpath = directory / "variants.tsv"
    logs = read_variants(vpath, len(genomes)) if vpath.exists() else [[] for _ in genomes]
    mpath = directory / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    return Collection(genomes, logs, manifest)
