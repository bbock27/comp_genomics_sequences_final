# Real genome collections

Downloads a pinned random sample of complete E. coli genomes from NCBI and turns each into one clean chromosome with a consistent strand and start. Needs the `pangenome` environment (numpy) and NCBI's `datasets` and `dataformat` binaries in `ncbi_tools/` or on `PATH`.

## Quick start
Run from the repository root:

```sh
python -m realdata all        # download (250 MB) + preprocess + reads and queries, a few minutes
```

This writes `data/real/ecoli/`:

| File | Contents |
|---|---|
| `genomes.fasta` | 50 chromosomes, records `g000` to `g049`, 248 million bases |
| `genomes.tsv` | per genome: accession, strain, length, and what preprocessing did |
| `manifest.json` | summary |
| `n001/` ... `n050/` | reads and queries drawn from the first 1, 2, 4, 8, 16, 32, and all 50 genomes |
| `raw/` | the assemblies as downloaded |

## Same format as the synthetic data
`data/real/ecoli/` is laid out like a synthetic collection such as `data/synthetic/uneven_d0.01/`, so code written against one reads the other:

- `genomes.fasta` has records `g000`, `g001`, ... and the first `n` records are the `n`-genome collection. Genomes are in random order, so that is a random subset.
- Each `nXXX/` holds `reads_err0`, `reads_err0.01`, and `queries` as `.fasta` and `.tsv`, with the same columns and the same defaults (10,000 reads of 150 bp, 250 queries of 100 bp per class). See `simulator/README.md` for the columns.
- `present_in` and `source_in` are computed the same way, by exact search of every genome.

What real data cannot have:

| Synthetic | Real |
|---|---|
| `variants.tsv`, `pairwise.tsv`, `tree.nwk` | absent: the true variants and tree are unknown |
| `ref_offset` column in reads | always `-1`: there is no shared reference coordinate |
| `recombinant` query class | absent: building one needs the variant log |

`python -m realdata samples` regenerates just the reads and queries; it takes the same options as the simulator's suite (`--copies`, `--num-reads`, `--read-length`, `--read-error-rates`, `--num-queries`, `--query-length`, `--seed`).

## Which genomes
`realdata/ecoli_accessions.tsv` lists the 50 assemblies. It is tracked in git and is what `download` uses, so everyone gets the same genomes. They were drawn at random (seed 447) from the 5,511 complete RefSeq E. coli assemblies available on 2026-11-11, excluding ones NCBI flags as atypical.

`python -m realdata sample` redraws the sample from NCBI's current catalogue. The catalogue grows, so this will usually pick different genomes; only run it if you want a new dataset. `--num` and `--seed` change the size and the draw. For another species, use `--name salmonella --taxon "Salmonella enterica"`.

## Preprocessing
1. **Chromosome only.** An assembly's FASTA holds the chromosome and any plasmids as separate records. The longest record is kept.
2. **No ambiguous bases.** Characters outside ACGT are removed. Genomes with more than 100 are skipped (`--max-ambiguous`).
3. **Same strand and start.** Every chromosome is reverse-complemented if needed and rotated to begin at the first base of the dnaA gene, reading in dnaA's direction. dnaA is found by matching 31-mers of the K-12 dnaA sequence (`dnaA_k12.fasta`). Genomes where it cannot be found are skipped.

`genomes.tsv` records the result for each genome:

| Column | Meaning |
|---|---|
| `records`, `other_bases_dropped` | records in the assembly, and bases in the ones dropped (plasmids) |
| `ambiguous_removed` | non-ACGT characters removed |
| `strand` | `-` if the chromosome was reverse-complemented |
| `rotation` | bases moved from the front to the back |
| `anchor_votes` | dnaA 31-mers that agreed on the start |

## What this does not do
- **It does not make the genomes collinear.** Real genomes differ by inversions, insertions, and gene gain and loss. A shared start and strand only removes the arbitrary differences in how each was deposited.
- **There is no variant log or pairwise truth.** Ground truth for read placement comes from sampling reads (the `present_in` and `source_in` columns). For genome comparison, the exact string-side answer has to serve as the reference.
- **Plasmids are discarded,** so plasmid-borne genes are not in the collection.
