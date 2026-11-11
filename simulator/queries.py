"""Query patterns in four classes.

present      an exact substring of one genome
mutated      a substring with a fixed number of substituted bases
random       uniformly random sequence
recombinant  the left part from genome A and the right part from genome B,
             joined between a variant of A and a variant of B. The pattern
             carries A's allele on the left and B's allele on the right, so
             normally no single genome contains it, but a de Bruijn graph
             accepts it whenever the two variants are far enough apart.

The class says how a pattern was *built*. Whether it actually occurs in some
genome (a mutated pattern can land on a repeat, for example) is recorded
separately in the ``present_in`` column computed by :mod:`simulator.truth`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path

from .genomes import Variant, genome_name
from .reads import _present, pick_window
from .seq import ALPHABET, random_sequence, write_fasta

CLASSES = ("present", "mutated", "random", "recombinant")


@dataclass
class Query:
    """A query pattern and how it was built.

    genome, offset    source window (present, mutated) or the left part's
                      genome and start (recombinant); -1 for random
    genome2, offset2  recombinant only: the right part's genome and its start
    breakpoint        recombinant only: length of the left part, so
                      seq[:breakpoint] is from genome and the rest from genome2
    variant_gap       recombinant only: number of reference bases between the
                      two switched variants (the bases both genomes share)
    edits             mutated only: "pos:old>new" for each substitution
    """

    id: str
    cls: str
    seq: str
    genome: int = -1
    offset: int = -1
    genome2: int = -1
    offset2: int = -1
    breakpoint: int = -1
    variant_gap: int = -1
    edits: str = "-"
    present_in: list[int] | None = field(default=None)


def recombinant_candidates(
    log_a: list[Variant], log_b: list[Variant], length: int, min_gap: int = 1, max_gap: int | None = None
) -> list[tuple[Variant, Variant]]:
    """Pairs (vA, vB): a variant of A followed by a variant of B, with no other
    variant of either genome in between, that fit in a pattern of ``length``."""
    # each entry is (ref_pos, ref_end, source genome, variant); sort on the first three
    merged = sorted(
        [(v.ref_pos, v.ref_end, 0, v) for v in log_a] + [(v.ref_pos, v.ref_end, 1, v) for v in log_b],
        key=itemgetter(0, 1, 2),
    )
    out = []
    b_end = 0  # furthest reference position touched by a variant of B so far
    for (_, _, src1, v1), (_, _, src2, v2) in zip(merged, merged[1:]):
        if src1 == 1:
            b_end = max(b_end, v1.ref_end)
            continue
        if src2 != 1:
            continue
        gap = v2.ref_pos - v1.ref_end
        if gap < min_gap or (max_gap is not None and gap > max_gap) or b_end > v1.ref_end:
            continue
        # one flanking base on each side, both alleles, and the shared gap
        if len(v1.alt) + 1 + gap + len(v2.alt) + 1 <= length:
            out.append((v1, v2))
    return out


def build_recombinant(
    genome_a: str, genome_b: str, va: Variant, vb: Variant, length: int, rng: random.Random
) -> tuple[str, int, int, int] | None:
    """Return (pattern, offset in A, offset in B of the right part, breakpoint)."""
    gap = vb.ref_pos - va.ref_end
    a_end = va.copy_end  # left part is A[s : a_end]
    b_start = vb.copy_pos - gap  # right part is B[b_start : e]
    min_left = len(va.alt) + 1
    min_right = gap + len(vb.alt) + 1
    # left length must also respect the ends of both genomes
    lo = max(min_left, length - (len(genome_b) - b_start))
    hi = min(length - min_right, a_end)
    if lo > hi:
        return None
    left = rng.randint(lo, hi)
    s = a_end - left
    pattern = genome_a[s:a_end] + genome_b[b_start : b_start + length - left]
    return pattern, s, b_start, left


def make_queries(
    genomes: list[str],
    logs: list[list[Variant]],
    rng: random.Random,
    num_per_class: int,
    length: int,
    mutations: int = 1,
    min_gap: int = 1,
    max_gap: int | None = None,
    classes: tuple[str, ...] = CLASSES,
) -> list[Query]:
    """Build ``num_per_class`` queries of each class.

    The recombinant class is skipped when it cannot be built: fewer than two
    genomes, no variant log (the control), or no variant pair close enough.
    """
    queries: list[Query] = []

    def qid() -> str:
        return f"q{len(queries):06d}"

    if "present" in classes:
        for _ in range(num_per_class):
            g, off = pick_window(genomes, rng, length)
            queries.append(Query(qid(), "present", genomes[g][off : off + length], g, off))

    if "mutated" in classes:
        for _ in range(num_per_class):
            g, off = pick_window(genomes, rng, length)
            chars = list(genomes[g][off : off + length])
            edits = []
            for p in sorted(rng.sample(range(length), min(mutations, length))):
                new = rng.choice([b for b in ALPHABET if b != chars[p]])
                edits.append(f"{p}:{chars[p]}>{new}")
                chars[p] = new
            queries.append(Query(qid(), "mutated", "".join(chars), g, off, edits=";".join(edits) or "-"))

    if "random" in classes:
        for _ in range(num_per_class):
            queries.append(Query(qid(), "random", random_sequence(rng, length)))

    if "recombinant" in classes and len(genomes) >= 2 and any(logs):
        cache: dict[tuple[int, int], list[tuple[Variant, Variant]]] = {}
        made = attempts = 0
        while made < num_per_class and attempts < 50 * num_per_class + 1000:
            attempts += 1
            a, b = rng.sample(range(len(genomes)), 2)
            if (a, b) not in cache:
                cache[(a, b)] = recombinant_candidates(logs[a], logs[b], length, min_gap, max_gap)
            if not cache[(a, b)]:
                continue
            va, vb = rng.choice(cache[(a, b)])
            built = build_recombinant(genomes[a], genomes[b], va, vb, length, rng)
            if built is None:
                continue
            seq, off_a, off_b, bp = built
            queries.append(
                Query(qid(), "recombinant", seq, a, off_a, b, off_b, bp, vb.ref_pos - va.ref_end)
            )
            made += 1
    return queries


def write_queries(prefix: str | Path, queries: list[Query]) -> None:
    """Write <prefix>.fasta (sequences) and <prefix>.tsv (ground truth)."""
    prefix = str(prefix)
    write_fasta(prefix + ".fasta", ((f"{q.id} class={q.cls}", q.seq) for q in queries), width=0)
    with_truth = any(q.present_in is not None for q in queries)

    def name(g: int) -> str:
        return genome_name(g) if g >= 0 else "-"

    def num(x: int) -> str:
        return str(x) if x >= 0 else "-"

    with open(prefix + ".tsv", "w") as fh:
        cols = ["query_id", "class", "length", "genome", "offset", "genome2", "offset2", "breakpoint", "variant_gap", "edits"]
        fh.write("\t".join(cols + (["present_in"] if with_truth else [])) + "\n")
        for q in queries:
            row = [q.id, q.cls, str(len(q.seq)), name(q.genome), num(q.offset), name(q.genome2), num(q.offset2),
                   num(q.breakpoint), num(q.variant_gap), q.edits]
            if with_truth:
                row.append(_present(q.present_in))
            fh.write("\t".join(row) + "\n")
