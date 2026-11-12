"""BWT and document array of a collection of genomes, built with SA-IS.

Same output as pangenome.bwt.build.build_bwt, by a different construction.
pangenome.bwt.build sorts suffixes by prefix doubling, which re-sorts the whole
array once per round. SA-IS (Nong, Zhang and Chan 2011) builds the suffix
array in time linear in the length of the text, with no sorting at all.

How SA-IS works, in brief:

1. Classify every suffix as S-type (smaller than the suffix after it) or
   L-type (larger). An S-type position whose left neighbour is L-type is an
   LMS position ("leftmost S"). At most half of all positions are LMS.
2. The text is cut into LMS substrings, each running from one LMS position to
   the next. Sort them with one "induced sort": drop the LMS positions at the
   ends of their first-character buckets, then one left-to-right pass places
   the L-type suffixes and one right-to-left pass places the S-type ones.
3. Give each distinct LMS substring a number, in sorted order. Replacing every
   LMS substring by its number yields a shorter text. If the numbers are all
   different the order of the LMS suffixes is known; otherwise solve the
   shorter text recursively.
4. Put the LMS suffixes into their buckets in their true order and repeat the
   two induced-sort passes. This time the result is the full suffix array.

The collection text follows pangenome.bwt.build: every genome has its own
terminator, and terminators sort before all bases in genome order. SA-IS also
needs a final character smaller than everything else, so one extra sentinel is
appended; its row is dropped at the end. Because every terminator is unique,
the sentinel does not change the order of any other suffix.

As in pangenome.bwt.build, the suffix array is only an intermediate and is
discarded.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .build import encode


@njit(cache=True)
def _bucket_heads(counts, heads):
    """First row of each character's bucket."""
    total = 0
    for c in range(len(counts)):
        heads[c] = total
        total += counts[c]


@njit(cache=True)
def _bucket_tails(counts, tails):
    """Last row of each character's bucket."""
    total = 0
    for c in range(len(counts)):
        total += counts[c]
        tails[c] = total - 1


@njit(cache=True)
def _induce(text, s_type, counts, pointer, sa):
    """Given some suffixes already in sa (the rest -1), place every L-type
    suffix and then every S-type suffix.

    A suffix is placed when the suffix one position to its right is visited:
    L-type suffixes fill their buckets from the front in a left-to-right
    pass, S-type suffixes fill theirs from the back in a right-to-left pass.
    """
    n = len(text)
    _bucket_heads(counts, pointer)
    for i in range(n):
        j = sa[i] - 1
        if j >= 0 and not s_type[j]:
            c = text[j]
            sa[pointer[c]] = j
            pointer[c] += 1
    _bucket_tails(counts, pointer)
    for i in range(n - 1, -1, -1):
        j = sa[i] - 1
        if j >= 0 and s_type[j]:
            c = text[j]
            sa[pointer[c]] = j
            pointer[c] -= 1


@njit(cache=True)
def _same_lms_substring(text, s_type, a, b):
    """Whether the LMS substrings starting at a and b are equal (same
    characters and same types, up to and including the next LMS position)."""
    n = len(text)
    if a == n - 1 or b == n - 1:  # the sentinel's substring equals no other
        return False
    i = 0
    while True:
        a_lms = s_type[a + i] and not s_type[a + i - 1]
        b_lms = s_type[b + i] and not s_type[b + i - 1]
        if i > 0 and a_lms and b_lms:
            return True
        if a_lms != b_lms or text[a + i] != text[b + i]:
            return False
        i += 1


@njit(cache=True)
def _sais(text, alphabet, sa):
    """Fill sa with the suffix array of text.

    text holds integers in [0, alphabet); its last character must be 0 and
    appear nowhere else.
    """
    n = len(text)
    if n == 1:
        sa[0] = 0
        return

    # 1. suffix types and bucket sizes
    s_type = np.zeros(n, dtype=np.bool_)
    s_type[n - 1] = True
    for i in range(n - 2, -1, -1):
        s_type[i] = text[i] < text[i + 1] or (text[i] == text[i + 1] and s_type[i + 1])
    counts = np.zeros(alphabet, dtype=np.int64)
    for i in range(n):
        counts[text[i]] += 1
    pointer = np.empty(alphabet, dtype=np.int64)

    # 2. sort the LMS substrings
    sa[:] = -1
    _bucket_tails(counts, pointer)
    lms_count = 0
    for i in range(1, n):
        if s_type[i] and not s_type[i - 1]:
            c = text[i]
            sa[pointer[c]] = i
            pointer[c] -= 1
            lms_count += 1
    _induce(text, s_type, counts, pointer, sa)

    # 3. number the LMS substrings in sorted order; name[p] for LMS position p
    name = np.full(n, -1, dtype=np.int64)
    names = 0
    previous = -1
    for i in range(n):
        p = sa[i]
        if p > 0 and s_type[p] and not s_type[p - 1]:
            if previous >= 0 and not _same_lms_substring(text, s_type, previous, p):
                names += 1
            name[p] = names
            previous = p
    names += 1

    lms_position = np.empty(lms_count, dtype=np.int64)
    reduced = np.empty(lms_count, dtype=np.int64)
    j = 0
    for i in range(1, n):
        if name[i] >= 0:
            lms_position[j] = i
            reduced[j] = name[i]
            j += 1
    name = np.empty(0, dtype=np.int64)  # no longer needed; free it before recursing

    reduced_sa = np.empty(lms_count, dtype=np.int64)
    if names == lms_count:  # every substring is distinct, so the order is already known
        for i in range(lms_count):
            reduced_sa[reduced[i]] = i
    else:
        _sais(reduced, names, reduced_sa)

    # 4. LMS suffixes in their true order, then induce everything else
    sa[:] = -1
    _bucket_tails(counts, pointer)
    for i in range(lms_count - 1, -1, -1):
        p = lms_position[reduced_sa[i]]
        c = text[p]
        sa[pointer[c]] = p
        pointer[c] -= 1
    _induce(text, s_type, counts, pointer, sa)


def suffix_array_sais(sym: np.ndarray, doc: np.ndarray) -> np.ndarray:
    """Suffix array of the collection text (see pangenome.bwt.build.encode)."""
    n = len(sym)
    d = int(doc.max()) + 1
    # 0 = final sentinel, 1..d = terminator of genome 0..d-1, then A, C, G, T
    text = np.empty(n + 1, dtype=np.int64)
    text[:n] = np.where(sym == 0, doc.astype(np.int64) + 1, sym.astype(np.int64) + d)
    text[n] = 0
    sa = np.empty(n + 1, dtype=np.int64)
    _sais(text, d + 5, sa)
    return sa[1:]  # row 0 is the sentinel's own suffix


def build_bwt_sais(genomes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (bwt, doc) for the collection; see pangenome.bwt.build."""
    sym, doc = encode(genomes)
    sa = suffix_array_sais(sym, doc)
    # The character before position 0 wraps to the last terminator, and the
    # character before any other genome start is a terminator too: both are '$'.
    return sym[sa - 1], doc[sa]
