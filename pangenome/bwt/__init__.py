"""BWT-based index and analyses.

build             BWT and document array of a genome collection
build_sais        the same BWT and document array, built with SA-IS (linear time)
similarity        pairwise genome similarity from BWT run counts (one BWT, one sweep)
similarity_naive  the same measures from a separate BWT per pair of genomes
similarity_sais   similarity, with the BWT built by SA-IS instead of prefix doubling

similarity_parallel        similarity, with the sweep spread over several cores
similarity_naive_parallel  similarity_naive, with the pairs spread over several processes
"""
