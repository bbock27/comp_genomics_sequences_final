# Synthetic data simulator

Generates every synthetic input the project needs, with ground truth, from fixed seeds. Python 3.9+, standard library only (numpy, if installed, makes the ground-truth pass faster).

## Quick start
Run from the repository root:

```sh
conda env create -f environment.yml     # once; then: conda activate pangenome
python -m simulator fetch-reference     # E. coli K-12 MG1655 from NCBI -> data/reference/
python -m simulator suite --small       # Python-scale dataset -> data/synthetic_small/ (5 s, 24 MB)
python -m simulator suite               # full E. coli -> data/synthetic/ (about 1 minute, 320 MB)
python -m unittest discover tests       # tests
```

`data/` is git-ignored. Anyone who runs the same commands gets byte-identical files.

**Which one to use.** `--small` uses the first 100 kb of the reference and 2,000 reads per set, so the largest collection is 1.6 million bases, which is quick to index while developing. The full suite uses the whole 4.6 Mb genome (74 million bases at 16 copies) and is meant for the final runs with numpy and numba. The two have the same layout and file formats. The 100 kb slice has almost no internal repeats (136 positions lie in a repeated 31-mer), so repeat-induced false matches only show up at full scale.

## What the suite writes

```
data/synthetic/
  uneven_d0.001/           16 copies related by a tree, about 0.1% divergence from the reference
    genomes.fasta          records g000 ... g015
    variants.tsv           every variant of every copy
    pairwise.tsv           shared and differing variants for every pair of genomes
    tree.nwk               the tree, with the length of every branch
    manifest.json          parameters, seed, reference checksum, variant counts
    n001/ n002/ n004/ n008/ n016/
      reads_err0.fasta/.tsv      10,000 error-free 150 bp reads
      reads_err0.01.fasta/.tsv   the same with 1% substitution errors
      queries.fasta/.tsv         250 patterns per class, 100 bp
  uneven_d0.01/            same, at 1% divergence
  control/                 16 unrelated random genomes of the same length
```

**Collections are nested.** Genome `i` depends only on the seed, the divergence, and `i`. So the first `n` records of `genomes.fasta` are the `n`-genome collection, and an index for 4 genomes should read the first 4 records. The reads and queries in `n004/` were drawn from those 4 genomes only. `n001/` is the single-copy case, for separating repeat-induced errors from recombination-induced ones.

## How the genomes are related
The default is the **uneven tree**. Two other models are available with `--relations` (for `suite`) or `--relation` (for `genomes`); each writes its own directory.

- **Tree** (`tree_*`): the copies are the leaves of a balanced binary tree of height 4, and each branch carries variants at rate `d/4`. Genomes inherit the variants of every branch above them, so close relatives share most of their variants.
- **Star** (`star_*`): every copy is mutated independently from the reference, so all pairs are about equally distant (`2d`) and "which genomes are most similar" has no answer.

In an equal-branch tree collection, the pair distance depends on how far up the tree two genomes meet:

| Pair | Example | Expected distance |
|---|---|---|
| siblings | g000, g001 | `0.5 d` |
| same group of 4 | g000, g002 | `1.0 d` |
| same group of 8 | g000, g004 | `1.5 d` |
| opposite halves | g000, g008 | `2.0 d` |

- **Uneven** (`uneven_*`, the default): the same tree shape, but each branch's rate is `d/4` times a factor drawn uniformly from 0.2 to 1.8. Genomes sit at different distances from the reference (about `0.65 d` to `1.4 d` in the default data), and pair distances no longer fall into four neat levels. This is the closest of the three to real data. The actual branch lengths are in `tree.nwk`, and each genome's divergence from the reference is in `manifest.json` under `genome_divergence`.

In an equal-branch tree, every genome is still at divergence `d` from the reference. All collections are nested as before. `tree.nwk` holds the tree in Newick format. Genomes g016 and up would start a second, unrelated tree (`--tree-height` changes the size).

## Mutation model
Each reference site starts a variant with probability `divergence`: 80% SNPs, 10% insertions, 10% deletions. Indel lengths are geometric with mean 2, capped at 10. At least one unchanged base separates consecutive variants in a copy. In tree mode the same process runs on each branch at that branch's rate (`d/4` when branches are equal); a variant that would overlap one inherited from higher in the tree is dropped.

## File formats
All coordinates are 0-based. `-` means empty or not applicable.

**`variants.tsv`**: `genome  ref_pos  type  ref  alt  copy_pos`
- `type` is `SNP`, `INS`, or `DEL`.
- An insertion goes immediately before reference base `ref_pos`.
- `copy_pos` is where `alt` starts in the copy (for a deletion, the first base after it).

**`pairwise.tsv`**: `genome_a  genome_b  shared_variants  only_a  only_b  distance`
- One row per pair. Two variants are the same when position, `ref`, and `alt` all match.
- `distance` is `(only_a + only_b) / reference length`. This is the answer key for Q1.

**`reads_*.tsv`**: `read_id  genome  offset  strand  length  errors  ref_offset  present_in  source_in`
- `genome`, `offset`: where the read was sampled from.
- `ref_offset`: the same position in reference coordinates. Reads from different genomes with equal `ref_offset` come from the same locus.
- `present_in`: every genome that contains the read exactly, as written (with its errors).
- `source_in`: every genome that contains the read's error-free original. Each of these is an equally correct answer to "which genome is this read from", so score voting mode against this column. It equals `present_in` for error-free reads.

**`queries.tsv`**: `query_id  class  length  genome  offset  genome2  offset2  breakpoint  variant_gap  edits  present_in`

| Class | How it is built | Columns used |
|---|---|---|
| `present` | exact substring of a genome | `genome`, `offset` |
| `mutated` | substring with substituted bases | `genome`, `offset`, `edits` (`pos:old>new`) |
| `random` | uniformly random | none |
| `recombinant` | left part from one genome, right part from another | all but `edits` |

For a recombinant, `seq[:breakpoint]` is `genome[offset:]` and the rest is `genome2[offset2:]`. The left part contains a variant of `genome` and the right part a variant of `genome2`. `variant_gap` is the number of reference bases between those two variants, which is the quantity to compare against `k`.

**`class` is how a pattern was built; `present_in` is the truth.** A graph match is false exactly when `present_in` is `-`. A few recombinant or mutated patterns do occur in some genome (repeats, or an indel in a repeated motif), and `present_in` records that.

## Individual commands

```sh
python -m simulator genomes --copies 8 --divergence 0.005 --out data/custom
python -m simulator genomes --copies 8 --divergence 0.005 --relation tree --out data/custom_tree
python -m simulator genomes --copies 8 --divergence 0.005 --relation star --out data/custom_star
python -m simulator control --copies 8 --out data/custom_control
python -m simulator reads   --collection data/custom --num-genomes 4 --num 5000 --length 150 \
                            --error-rate 0.01 --out data/custom/reads
python -m simulator queries --collection data/custom --min-gap 20 --max-gap 40 --out data/custom/queries
```

`--min-gap` and `--max-gap` restrict recombinants to a range of variant distances. `--both-strands` samples reads from either strand (off by default). `--ref-start` and `--ref-length` use a slice of the reference; for a tiny dataset to check an index against brute force, try `suite --ref-length 2000 --copies 1 2 4 --out data/tiny`. `suite --relations uneven tree star` generates all three kinds. Run any command with `-h` for all options.

## Using it from Python

```python
from simulator.genomes import load_collection, CoordMap
col = load_collection("data/synthetic_small/uneven_d0.01", num_genomes=4)
col.genomes[0]                         # sequence of g000
CoordMap(col.logs[0]).copy_to_ref(12345)
```

## Not included
- Random tree shapes. The tree is always balanced, though branch lengths can vary.
- Repeated mutation at the same site on different branches.
- Indel sequencing errors in reads. Errors are substitutions only.
- Edit-distance truth for reads with errors. `source_in` says which genomes the read could have come from; it does not check whether some other genome happens to be equally close after the errors.
