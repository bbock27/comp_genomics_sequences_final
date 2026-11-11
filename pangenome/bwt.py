"""BWT and document array of a collection of genomes.

The collection is the concatenation  G0 $0 G1 $1 ... , where every genome has
its own terminator and terminators sort before all bases, in genome order.
Because each terminator is unique, comparing two suffixes never reads past the
first terminator, so a suffix is ordered by the rest of its own genome only.

That gives the property the similarity analysis relies on: the suffix order of
any sub-collection is the full order with the other genomes' rows removed.

Outputs (the suffix array itself is only an intermediate and is discarded):

bwt   uint8 per row: 0 for '$', 1-4 for A, C, G, T. The character before
      each suffix; a suffix at the start of a genome gets '$'.
doc   the genome each row's suffix starts in (the document array)
"""

from __future__ import annotations

import numpy as np

SYMBOLS = "$ACGT"
_SYM = np.zeros(256, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _SYM[ord(_c)] = _i + 1


def encode(genomes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate genomes with terminators.

    Returns (sym, doc): the symbol (0 = terminator, 1-4 = ACGT) and the genome
    index of every text position.
    """
    d = len(genomes)
    n = sum(len(g) for g in genomes) + d
    sym = np.zeros(n, dtype=np.uint8)
    doc = np.zeros(n, dtype=np.uint8 if d <= 256 else np.uint16)
    pos = 0
    for i, g in enumerate(genomes):
        raw = np.frombuffer(g.encode("ascii"), dtype=np.uint8)
        codes = _SYM[raw]
        if not codes.all():
            raise ValueError(f"genome {i} contains characters outside ACGT")
        sym[pos : pos + len(g)] = codes
        doc[pos : pos + len(g) + 1] = i
        pos += len(g) + 1
    return sym, doc


def suffix_ranks(sym: np.ndarray, doc: np.ndarray) -> np.ndarray:
    """Rank of every suffix (the inverse suffix array), by prefix doubling.

    Round 0 ranks suffixes by their first w characters at once (as many as fit
    in 64 bits), then each round doubles the compared length with one sort.
    """
    n = len(sym)
    d = int(doc.max()) + 1
    # sort codes: 0 = past the end, 1..d = terminator of genome 0..d-1, then A, C, G, T
    code = np.where(sym == 0, doc.astype(np.int64) + 1, sym.astype(np.int64) + d).astype(np.uint64)
    base = d + 5
    w = 1
    while base ** (w + 1) < 2**63:
        w += 1
    key = np.zeros(n, dtype=np.uint64)
    for j in range(w):
        key *= np.uint64(base)
        if j < n:
            key[: n - j] += code[j:]
    del code
    rank = np.unique(key, return_inverse=True)[1].astype(np.int64)
    del key

    k = w
    while int(rank.max()) < n - 1:  # until every suffix has its own rank
        combined = rank * (n + 1)
        if k < n:
            combined[: n - k] += rank[k:] + 1  # second half of the 2k-prefix; 0 if past the end
        order = np.argsort(combined)
        combined = combined[order]
        rank[order[0]] = 0
        rank[order[1:]] = np.cumsum(combined[1:] != combined[:-1])
        k *= 2
    return rank


def build_bwt(genomes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (bwt, doc) for the collection; see the module docstring."""
    sym, doc = encode(genomes)
    rank = suffix_ranks(sym, doc)
    sa = np.empty(len(sym), dtype=np.int64)
    sa[rank] = np.arange(len(sym), dtype=np.int64)
    del rank
    # The character before position 0 wraps to the last terminator, and the
    # character before any other genome start is a terminator too: both are '$'.
    return sym[sa - 1], doc[sa]


def count_runs(bwt: np.ndarray) -> int:
    """Number of maximal runs of equal characters."""
    return int(len(bwt) > 0) + int(np.count_nonzero(bwt[1:] != bwt[:-1]))


def bwt_string(bwt: np.ndarray) -> str:
    return "".join(SYMBOLS[c] for c in bwt.tolist())
