"""Exact-match ground truth: which genomes contain each pattern.

This is the simulator's own brute-force check. It answers only "does this
exact string occur on the forward strand of genome g", which is what is needed
to score Q2 in exact mode and to detect false matches from the graph.

With numpy it scans each genome once for all patterns together (a 2-bit packed
anchor k-mer lookup followed by a full string comparison). Without numpy it
falls back to one ``str.find`` per pattern and genome, which gives the same
answer more slowly.
"""

from __future__ import annotations

try:  # numpy is optional
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def exact_presence_naive(patterns: list[str], genomes: list[str]) -> list[list[int]]:
    return [[gi for gi, g in enumerate(genomes) if p in g] for p in patterns]


def exact_presence(patterns: list[str], genomes: list[str]) -> list[list[int]]:
    """For each pattern, the sorted list of genome indices that contain it."""
    if np is None or not patterns:
        return exact_presence_naive(patterns, genomes)
    anchor = min(32, min(len(p) for p in patterns))
    if anchor < 12 or any(set(p) - set("ACGT") for p in patterns):
        return exact_presence_naive(patterns, genomes)

    code = np.zeros(256, dtype=np.uint64)
    for i, c in enumerate("ACGT"):
        code[ord(c)] = i

    def pack(text: str, n: int):
        arr = code[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
        out = np.zeros(n, dtype=np.uint64)
        for j in range(anchor):
            out = (out << np.uint64(2)) | arr[j : j + n]
        return out

    by_anchor: dict[int, list[int]] = {}
    for pi, p in enumerate(patterns):
        by_anchor.setdefault(int(pack(p[:anchor], 1)[0]), []).append(pi)
    keys = np.fromiter(by_anchor.keys(), dtype=np.uint64, count=len(by_anchor))

    found: list[set[int]] = [set() for _ in patterns]
    for gi, g in enumerate(genomes):
        n = len(g) - anchor + 1
        if n <= 0:
            continue
        kmers = pack(g, n)
        hits = np.nonzero(np.isin(kmers, keys))[0]
        for pos, key in zip(hits.tolist(), kmers[hits].tolist()):
            for pi in by_anchor[key]:
                if gi not in found[pi] and g.startswith(patterns[pi], pos):
                    found[pi].add(gi)
    return [sorted(s) for s in found]
