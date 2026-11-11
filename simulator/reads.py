"""Sample reads from a genome collection, recording where each came from."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .genomes import CoordMap, Variant, genome_name
from .seq import ALPHABET, revcomp, write_fasta


@dataclass
class Read:
    """A read and its ground truth.

    genome, offset  source genome index and 0-based start in that genome
                    (always on the forward strand of the genome)
    strand          '+' or '-'; a '-' read is the reverse complement of
                    genome[offset : offset + length]
    errors          number of substituted bases
    ref_offset      reference coordinate of the read's start (-1 if the
                    collection has no variant log). Reads from different
                    genomes with the same ref_offset come from the same locus.
    present_in      genomes containing the read exactly (filled in by truth)
    source_in       genomes containing the read's error-free original. Every
                    one of them is an equally correct answer to "which genome
                    is this read from". Equal to present_in when errors == 0.
    source_seq      the error-free window on the genome's forward strand (kept
                    in memory only when the read has errors; not written)
    """

    id: str
    seq: str
    genome: int
    offset: int
    strand: str = "+"
    errors: int = 0
    ref_offset: int = -1
    present_in: list[int] | None = field(default=None)
    source_in: list[int] | None = field(default=None)
    source_seq: str | None = field(default=None, repr=False)


def pick_window(genomes: list[str], rng: random.Random, length: int) -> tuple[int, int]:
    """Pick (genome, offset) uniformly over all windows of the given length."""
    weights = [max(len(g) - length + 1, 0) for g in genomes]
    if not any(weights):
        raise ValueError(f"no genome is at least {length} bases long")
    g = rng.choices(range(len(genomes)), weights=weights)[0]
    return g, rng.randrange(weights[g])


def substitute(seq: str, rng: random.Random, rate: float) -> tuple[str, list[tuple[int, str, str]]]:
    """Substitute each base independently with probability ``rate``."""
    edits: list[tuple[int, str, str]] = []
    if rate <= 0:
        return seq, edits
    chars = list(seq)
    for i, c in enumerate(chars):
        if rng.random() < rate:
            new = rng.choice([b for b in ALPHABET if b != c])
            edits.append((i, c, new))
            chars[i] = new
    return "".join(chars), edits


def sample_reads(
    genomes: list[str],
    rng: random.Random,
    num: int,
    length: int,
    error_rate: float = 0.0,
    both_strands: bool = False,
    logs: list[list[Variant]] | None = None,
) -> list[Read]:
    maps = [CoordMap(v) for v in logs] if logs and any(logs) else None
    width = max(6, len(str(num)))
    reads = []
    for i in range(num):
        g, offset = pick_window(genomes, rng, length)
        seq = window = genomes[g][offset : offset + length]
        strand = "+"
        if both_strands and rng.random() < 0.5:
            seq, strand = revcomp(seq), "-"
        seq, edits = substitute(seq, rng, error_rate)
        ref_offset = maps[g].copy_to_ref(offset) if maps else -1
        reads.append(Read(f"r{i:0{width}d}", seq, g, offset, strand, len(edits), ref_offset,
                          source_seq=window if edits else None))
    return reads


def _present(present_in: list[int] | None) -> str:
    return ",".join(genome_name(g) for g in present_in) if present_in else "-"


def write_reads(prefix: str | Path, reads: list[Read]) -> None:
    """Write <prefix>.fasta (sequences) and <prefix>.tsv (ground truth)."""
    prefix = str(prefix)
    write_fasta(
        prefix + ".fasta",
        ((f"{r.id} genome={genome_name(r.genome)} offset={r.offset} strand={r.strand} errors={r.errors}", r.seq) for r in reads),
        width=0,
    )
    with_truth = any(r.present_in is not None for r in reads)
    with open(prefix + ".tsv", "w") as fh:
        cols = ["read_id", "genome", "offset", "strand", "length", "errors", "ref_offset"]
        fh.write("\t".join(cols + (["present_in", "source_in"] if with_truth else [])) + "\n")
        for r in reads:
            row = [r.id, genome_name(r.genome), r.offset, r.strand, len(r.seq), r.errors, r.ref_offset]
            if with_truth:
                row += [_present(r.present_in), _present(r.source_in)]
            fh.write("\t".join(map(str, row)) + "\n")
