"""FASTA I/O, deterministic seeding, and small sequence helpers."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable

ALPHABET = "ACGT"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def derive_seed(*parts: object) -> int:
    """Derive a 64-bit seed from a base seed and any number of labels.

    Every random stream in the simulator gets its own seed this way, so that
    (for example) genome 3 at divergence 0.001 is the same sequence no matter
    how many other genomes are generated alongside it.
    """
    text = "/".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def rng_for(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))


def random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(ALPHABET, k=length))


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def clean(seq: str) -> tuple[str, int]:
    """Uppercase and drop every character outside ACGT.

    Returns the cleaned sequence and the number of characters dropped.
    """
    upper = seq.upper()
    if set(upper) <= set(ALPHABET):
        return upper, 0
    kept = "".join(c for c in upper if c in ALPHABET)
    return kept, len(upper) - len(kept)


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA file into a list of (header without '>', sequence)."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            elif line:
                if header is None:
                    raise ValueError(f"{path}: sequence data before first header")
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta(path: str | Path, records: Iterable[tuple[str, str]], width: int = 80) -> None:
    """Write (header, sequence) records. width=0 puts each sequence on one line."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            if width <= 0:
                fh.write(seq + "\n")
            else:
                for i in range(0, len(seq), width):
                    fh.write(seq[i : i + width] + "\n")


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
