"""Turn a downloaded assembly into one clean, consistently oriented chromosome.

Three steps, each recorded in the output table:

1. Chromosome only. A complete assembly holds the chromosome plus any
   plasmids as separate FASTA records; we keep the longest record.
2. No ambiguous bases. Characters outside ACGT are removed.
3. Same strand and start. Circular chromosomes are deposited starting anywhere
   and on either strand. We locate the dnaA gene (which sits next to the
   origin of replication) using k-mers of the E. coli K-12 dnaA sequence,
   reverse-complement the chromosome if dnaA is on the reverse strand, and
   rotate it to begin at the first base of dnaA.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulator.seq import clean, read_fasta, revcomp

ANCHOR_FILE = Path(__file__).with_name("dnaA_k12.fasta")
K = 31

_CODE = np.zeros(256, dtype=np.uint64)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i


def load_anchor() -> str:
    return read_fasta(ANCHOR_FILE)[0][1]


def pack_kmers(seq: str, k: int = K) -> np.ndarray:
    """2-bit packed k-mer starting at every position of an ACGT string."""
    n = len(seq) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    arr = _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    out = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        out = (out << np.uint64(2)) | arr[j : j + n]
    return out


def _anchor_start(genome_kmers: np.ndarray, anchor: str, genome_length: int) -> tuple[int, int]:
    """Where a copy of ``anchor`` starts in the genome, by k-mer voting.

    Each anchor k-mer found in the genome votes for (its position - its offset
    in the anchor). Indels inside the gene shift the votes of everything after
    them, so the final answer comes from the agreeing k-mer nearest the start
    of the anchor. Returns (start modulo genome length, votes for the winner).
    """
    akm = pack_kmers(anchor)
    values, first = np.unique(akm, return_index=True)  # ignore k-mers repeated within the anchor
    hits = np.nonzero(np.isin(genome_kmers, values))[0]
    if len(hits) == 0:
        return -1, 0
    offsets = first[np.searchsorted(values, genome_kmers[hits])]
    starts = (hits - offsets) % genome_length
    mode, count = Counter(starts.tolist()).most_common(1)[0]
    # hits within a few bases of the consensus belong to the same gene copy
    delta = (starts - mode + genome_length // 2) % genome_length - genome_length // 2
    near = np.abs(delta) <= 60
    best = np.argmin(np.where(near, offsets, len(anchor)))
    return int(starts[best]), int(near.sum())


@dataclass
class Normalized:
    seq: str
    strand: str  # '+' kept as deposited, '-' reverse-complemented
    rotation: int  # bases moved from the front to the back (after any flip)
    anchor_votes: int  # anchor k-mers agreeing on the start (0 = anchor not found)


def normalize(seq: str, anchor: str) -> Normalized:
    """Reverse-complement and rotate a circular chromosome to start at the anchor."""
    n = len(seq)
    kmers = pack_kmers(seq)
    fwd_start, fwd_votes = _anchor_start(kmers, anchor, n)
    rc_start, rc_votes = _anchor_start(kmers, revcomp(anchor), n)
    if max(fwd_votes, rc_votes) == 0:
        return Normalized(seq, "+", 0, 0)
    if fwd_votes >= rc_votes:
        strand, start, votes = "+", fwd_start, fwd_votes
    else:
        # flip the genome, then find the start again on the forward strand
        seq, strand = revcomp(seq), "-"
        start, votes = _anchor_start(pack_kmers(seq), anchor, n)
    return Normalized(seq[start:] + seq[:start], strand, start, votes)


@dataclass
class Processed:
    seq: str
    record: str  # header of the record kept
    records: int  # records in the assembly (chromosome + plasmids)
    other_bases: int  # bases in the records dropped
    ambiguous: int  # non-ACGT characters removed
    strand: str
    rotation: int
    anchor_votes: int


def process_assembly(fasta: str | Path, anchor: str) -> Processed:
    records = read_fasta(fasta)
    if not records:
        raise ValueError(f"{fasta}: no FASTA records")
    header, raw = max(records, key=lambda r: len(r[1]))
    seq, dropped = clean(raw)
    norm = normalize(seq, anchor)
    return Processed(norm.seq, header, len(records), sum(len(s) for _, s in records) - len(raw), dropped,
                     norm.strand, norm.rotation, norm.anchor_votes)
