"""Colored de Bruijn graph of a collection of genomes.

Slide a window of length k along every genome. Each distinct window (a k-mer)
is a node, and its color is the set of genomes it was seen in, numbered by
their position in the input list (0 for the first genome).

Slide a window of length k+1 the same way. Each one is an edge from its first
k bases to its last k bases. So two k-mers are joined only if they really sit
next to each other in some genome; overlapping in k-1 bases is not enough.
With k = 2 and the genomes "AAC" and "CG", the nodes AC and CG overlap in C,
but ACG is in no genome, so there is no edge from AC to CG.

Only the forward strand is read: no reverse complements.

Example:

    colors, edges = build_graph(["ACGT", "ACGA"], 3)

    colors == {"ACG": {0, 1}, "CGT": {0}, "CGA": {1}}
    edges  == {"ACG": {"CGT", "CGA"}, "CGT": set(), "CGA": set()}

Using the result:

    kmer in colors    is this k-mer a node?
    colors[kmer]      the genomes that contain it
    edges[kmer]       the k-mers its edges lead to (an empty set if none)

Both dictionaries have the same keys, the nodes. Looking up a k-mer that is
not a node raises KeyError, so check "kmer in colors" first. The sets belong
to the graph: to change one, work on a copy such as set(colors[kmer]).
"""


def build_graph(genomes: list[str], k: int):
    """Return (colors, edges). colors[kmer] is the set of genomes that contain
    the k-mer; edges[kmer] is the set of k-mers it has an edge to."""
    if k < 1:
        raise ValueError("k must be at least 1")

    colors = {}
    edges = {}

    for genome_id, genome in enumerate(genomes):
        # Nodes. A genome shorter than k has no window, so the loop does not run.
        for start in range(len(genome) - k + 1):
            kmer = genome[start : start + k]
            if kmer not in colors:
                colors[kmer] = set()
                edges[kmer] = set()  # every node gets an entry, even with no edge out
            colors[kmer].add(genome_id)

        # Edges. Both ends of a (k+1)-mer are k-mers of this genome, so the
        # loop above has already made them nodes.
        for start in range(len(genome) - k):
            edge = genome[start : start + k + 1]  # a (k+1)-mer
            source = edge[:k]  # its first k bases
            target = edge[1:]  # its last k bases
            edges[source].add(target)

    return colors, edges
