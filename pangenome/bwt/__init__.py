"""BWT-based index and analyses.

build             BWT and document array of a genome collection
similarity        pairwise genome similarity from BWT run counts (one BWT, one sweep)
similarity_naive  the same measures from a separate BWT per pair of genomes

similarity_parallel        similarity, with the sweep spread over several cores
similarity_naive_parallel  similarity_naive, with the pairs spread over several processes
"""
