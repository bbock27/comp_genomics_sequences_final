"""Synthetic data for the pangenome indexing project.

Modules
-------
seq      FASTA I/O, seeding, small sequence helpers
genomes  mutate a reference into near-identical copies, with a variant log
reads    sample reads with their source genome and offset
queries  query patterns: present, mutated, random, recombinant
truth    exact-match ground truth (which genomes contain a pattern)
cli      command-line entry point (python -m simulator ...)
"""

__version__ = "0.1.0"
