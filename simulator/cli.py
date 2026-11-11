"""Command-line interface: python -m simulator <command> ...

Commands
--------
fetch-reference  download E. coli K-12 MG1655 (NC_000913.3) from NCBI
genomes          mutate a reference into a collection with a variant log
control          unrelated random genomes (non-repetitive control)
reads            sample reads from a collection
queries          build query patterns from a collection
suite            everything above for a grid of sizes and divergences
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .genomes import (
    Collection, MutationModel, genome_name, load_collection, newick, pairwise_truth, simulate_collection,
    simulate_control, simulate_tree_collection, write_pairwise, write_variants,
)
from .queries import make_queries, write_queries
from .reads import sample_reads, write_reads
from .seq import clean, read_fasta, revcomp, rng_for, sha256_of, write_fasta
from .truth import exact_presence

ECOLI_ACCESSION = "NC_000913.3"
ECOLI_LENGTH = 4_641_652
DEFAULT_REFERENCE = "data/reference/ecoli_k12_mg1655.fasta"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_reference(path: str, start: int = 0, length: int | None = None) -> tuple[str, str]:
    """Load the first FASTA record, optionally only [start, start + length)."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found; run: python -m simulator fetch-reference")
    records = read_fasta(path)
    if not records:
        raise SystemExit(f"{path}: no FASTA records")
    if len(records) > 1:
        log(f"warning: {path} has {len(records)} records; using only the first ({records[0][0]})")
    header, raw = records[0]
    seq, dropped = clean(raw)
    if dropped:
        log(f"warning: dropped {dropped} non-ACGT characters from the reference")
    if start or length is not None:
        end = len(seq) if length is None else start + length
        if not 0 <= start < end <= len(seq):
            raise SystemExit(f"region {start}..{end} is outside the reference (length {len(seq)})")
        seq, header = seq[start:end], f"{header} [region {start}..{end}]"
    return header, seq


def model_from(args: argparse.Namespace) -> MutationModel:
    return MutationModel(args.snp_frac, args.ins_frac, args.del_frac, args.max_indel)


def write_collection(out: Path, genomes: list[str], logs, manifest: dict, label: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_fasta(out / "genomes.fasta", ((f"{genome_name(i)} {label}", g) for i, g in enumerate(genomes)))
    if logs is not None:
        write_variants(out / "variants.tsv", logs)
        counts = Counter(v.kind for vs in logs for v in vs)
        manifest["variants"] = {"total": sum(counts.values()), **dict(sorted(counts.items()))}
    manifest["simulator_version"] = __version__
    manifest["genome_lengths"] = [len(g) for g in genomes]
    manifest["note"] = "Genome i depends only on (seed, divergence, i): the first n records are the n-genome collection."
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def make_genomes(header: str, ref: str, copies: int, divergence: float, seed: int, model: MutationModel,
                 out: Path, tree_height: int | None = None, variation: float = 0.0) -> None:
    """Write a mutated collection: star model, or tree model if tree_height is set."""
    if tree_height:
        genomes, logs = simulate_tree_collection(ref, copies, divergence, seed, model, tree_height, variation)
        relation = {"relation": "tree", "tree_height": tree_height, "branch_variation": variation}
    else:
        genomes, logs = simulate_collection(ref, copies, divergence, seed, model)
        relation = {"relation": "star"}
    manifest = {
        "kind": "mutated", **relation, "seed": seed, "copies": copies, "divergence": divergence,
        "model": asdict(model), "reference": {"header": header, "length": len(ref), "sha256": sha256_of(ref)},
        "genome_divergence": [round(len(v) / len(ref), 6) for v in logs],
    }
    write_collection(out, genomes, logs, manifest, f"{relation['relation']} divergence={divergence} seed={seed}")
    write_pairwise(out / "pairwise.tsv", pairwise_truth(logs, len(ref)))
    if tree_height:
        (out / "tree.nwk").write_text(newick(copies, divergence, tree_height, seed, variation) + "\n")
    log(f"  {out}: {copies} genomes ({relation['relation']}), {sum(map(len, logs))} variants")


def make_control(length: int, copies: int, seed: int, out: Path) -> None:
    genomes = simulate_control(length, copies, seed)
    manifest = {"kind": "control", "seed": seed, "copies": copies, "length": length}
    write_collection(out, genomes, None, manifest, f"random seed={seed}")
    log(f"  {out}: {copies} random genomes of length {length}")


def add_truth(items, genomes: list[str]) -> None:
    """Fill in present_in for reads or queries (reads on '-' are flipped back),
    and source_in for reads (genomes containing the error-free original)."""
    patterns = [revcomp(x.seq) if getattr(x, "strand", "+") == "-" else x.seq for x in items]
    with_errors = [x for x in items if getattr(x, "source_seq", None)]
    found = exact_presence(patterns + [x.source_seq for x in with_errors], genomes)
    for x, f in zip(items, found):
        x.present_in = f
        if hasattr(x, "source_in"):
            x.source_in = f
    for x, f in zip(with_errors, found[len(items):]):
        x.source_in = f


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        log(f"{out} already exists (use --force to download again)")
        return 0
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={args.accession}&rettype=fasta&retmode=text"
    )
    log(f"downloading {args.accession} from NCBI ...")
    req = urllib.request.Request(url, headers={"User-Agent": "pangenome-simulator"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode()
    if not text.startswith(">"):
        raise SystemExit("NCBI did not return FASTA; try again later")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    _, seq = load_reference(str(out))
    if args.accession == ECOLI_ACCESSION and len(seq) != ECOLI_LENGTH:
        raise SystemExit(f"unexpected length {len(seq)} (expected {ECOLI_LENGTH}); file kept at {out}")
    log(f"wrote {out} ({len(seq)} bp, sha256 {sha256_of(seq)[:16]}...)")
    return 0


def cmd_genomes(args: argparse.Namespace) -> int:
    header, ref = load_reference(args.reference, args.ref_start, args.ref_length)
    make_genomes(header, ref, args.copies, args.divergence, args.seed, model_from(args), Path(args.out),
                 args.tree_height if args.relation != "star" else None,
                 args.branch_variation if args.relation == "uneven" else 0.0)
    return 0


def cmd_control(args: argparse.Namespace) -> int:
    length = args.length if args.length else len(load_reference(args.reference, args.ref_start, args.ref_length)[1])
    make_control(length, args.copies, args.seed, Path(args.out))
    return 0


def cmd_reads(args: argparse.Namespace) -> int:
    col = load_collection(args.collection, args.num_genomes)
    rng = rng_for(args.seed, "reads", len(col.genomes), args.length, args.error_rate)
    reads = sample_reads(col.genomes, rng, args.num, args.length, args.error_rate, args.both_strands, col.logs)
    if not args.no_truth:
        add_truth(reads, col.genomes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_reads(args.out, reads)
    log(f"wrote {args.out}.fasta and {args.out}.tsv ({len(reads)} reads)")
    return 0


def cmd_queries(args: argparse.Namespace) -> int:
    col = load_collection(args.collection, args.num_genomes)
    rng = rng_for(args.seed, "queries", len(col.genomes), args.length)
    queries = make_queries(col.genomes, col.logs, rng, args.num_per_class, args.length, args.mutations, args.min_gap, args.max_gap)
    if not args.no_truth:
        add_truth(queries, col.genomes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_queries(args.out, queries)
    counts = Counter(q.cls for q in queries)
    log(f"wrote {args.out}.fasta and {args.out}.tsv ({dict(counts)})")
    return 0


def _samples_for(col: Collection, label: str, sizes: list[int], args: argparse.Namespace):
    """Reads and queries for every collection size, with truth from one pass."""
    batches = []  # (n, file prefix, items, writer)
    for n in sizes:
        genomes, logs = col.genomes[:n], col.logs[:n]
        for err in args.read_error_rates:
            rng = rng_for(args.seed, "reads", label, n, args.read_length, err)
            reads = sample_reads(genomes, rng, args.num_reads, args.read_length, err, args.both_strands, logs)
            batches.append((n, f"reads_err{err:g}", reads, write_reads))
        rng = rng_for(args.seed, "queries", label, n, args.query_length)
        queries = make_queries(genomes, logs, rng, args.num_queries, args.query_length, args.mutations)
        batches.append((n, "queries", queries, write_queries))
    if not args.no_truth:
        everything = [x for _, _, items, _ in batches for x in items]
        add_truth(everything, col.genomes)  # one scan of every genome
        for n, _, items, _ in batches:
            for x in items:
                x.present_in = [g for g in x.present_in if g < n]
                if getattr(x, "source_in", None) is not None:
                    x.source_in = [g for g in x.source_in if g < n]
    return batches


def cmd_suite(args: argparse.Namespace) -> int:
    if args.small:  # a scale that pure-Python indexes can handle
        args.ref_length = args.ref_length or 100_000
        args.num_reads = args.num_reads or 2000
        args.out = args.out or "data/synthetic_small"
    args.num_reads = args.num_reads or 10000
    args.out = args.out or "data/synthetic"
    out = Path(args.out)
    sizes = sorted(set(args.copies))
    top = max(sizes)
    model = model_from(args)
    start = time.time()
    header, ref = load_reference(args.reference, args.ref_start, args.ref_length)

    collections = []
    for relation in args.relations:
        for d in args.divergence:
            label = f"{relation}_d{d:g}"
            log(f"[{time.time() - start:5.0f}s] {relation} genomes at divergence {d:g}")
            make_genomes(header, ref, top, d, args.seed, model, out / label,
                         args.tree_height if relation != "star" else None,
                         args.branch_variation if relation == "uneven" else 0.0)
            collections.append(label)
    if not args.no_control:
        log(f"[{time.time() - start:5.0f}s] random control")
        make_control(len(ref), top, args.seed, out / "control")
        collections.append("control")

    for label in collections:
        log(f"[{time.time() - start:5.0f}s] reads and queries for {label}")
        col = load_collection(out / label)
        for n, name, items, writer in _samples_for(col, label, sizes, args):
            sub = out / label / f"n{n:03d}"
            sub.mkdir(parents=True, exist_ok=True)
            writer(sub / name, items)

    config = {k: v for k, v in vars(args).items() if k not in ("func", "command")}
    (out / "suite.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"[{time.time() - start:5.0f}s] done: {out}")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def _add_region(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ref-start", type=int, default=0, help="use the reference from this 0-based position")
    p.add_argument("--ref-length", type=int, help="use only this many reference bases")


def _add_model(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("mutation model")
    g.add_argument("--snp-frac", type=float, default=0.8, help="fraction of variants that are SNPs (default 0.8)")
    g.add_argument("--ins-frac", type=float, default=0.1, help="fraction that are insertions (default 0.1)")
    g.add_argument("--del-frac", type=float, default=0.1, help="fraction that are deletions (default 0.1)")
    g.add_argument("--max-indel", type=int, default=10, help="longest indel (default 10)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m simulator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-reference", help="download the E. coli K-12 MG1655 reference")
    p.add_argument("--out", default=DEFAULT_REFERENCE)
    p.add_argument("--accession", default=ECOLI_ACCESSION)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("genomes", help="simulate mutated copies of a reference")
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--copies", type=int, required=True)
    p.add_argument("--divergence", type=float, required=True, help="per-site variant rate against the reference")
    p.add_argument("--seed", type=int, default=447)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--relation", choices=["uneven", "tree", "star"], default="uneven",
                   help="how genomes are related: a tree with unequal branches (default), a tree with equal "
                        "branches, or independent copies")
    p.add_argument("--tree-height", type=int, default=4, help="levels in the tree; 2**height genomes per tree (default 4)")
    p.add_argument("--branch-variation", type=float, default=0.8,
                   help="for 'uneven': branch lengths are scaled by a factor in [1-V, 1+V] (default 0.8)")
    _add_region(p)
    _add_model(p)
    p.set_defaults(func=cmd_genomes)

    p = sub.add_parser("control", help="simulate unrelated random genomes")
    p.add_argument("--length", type=int, help="genome length (default: length of --reference)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--copies", type=int, required=True)
    p.add_argument("--seed", type=int, default=447)
    p.add_argument("--out", required=True, help="output directory")
    _add_region(p)
    p.set_defaults(func=cmd_control)

    p = sub.add_parser("reads", help="sample reads from a collection")
    p.add_argument("--collection", required=True, help="directory written by 'genomes' or 'control'")
    p.add_argument("--num-genomes", type=int, help="use only the first N genomes")
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--length", type=int, default=150)
    p.add_argument("--error-rate", type=float, default=0.0, help="per-base substitution rate")
    p.add_argument("--both-strands", action="store_true")
    p.add_argument("--seed", type=int, default=447)
    p.add_argument("--no-truth", action="store_true", help="skip the present_in and source_in columns")
    p.add_argument("--out", required=True, help="output prefix (writes PREFIX.fasta and PREFIX.tsv)")
    p.set_defaults(func=cmd_reads)

    p = sub.add_parser("queries", help="build query patterns from a collection")
    p.add_argument("--collection", required=True)
    p.add_argument("--num-genomes", type=int)
    p.add_argument("--num-per-class", type=int, default=250)
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--mutations", type=int, default=1, help="substitutions per mutated query")
    p.add_argument("--min-gap", type=int, default=1, help="recombinants: min distance between switched variants")
    p.add_argument("--max-gap", type=int, help="recombinants: max distance between switched variants")
    p.add_argument("--seed", type=int, default=447)
    p.add_argument("--no-truth", action="store_true")
    p.add_argument("--out", required=True, help="output prefix")
    p.set_defaults(func=cmd_queries)

    p = sub.add_parser("suite", help="regenerate every synthetic dataset")
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--small", action="store_true",
                   help="small preset for pure-Python indexes: first 100 kb of the reference, 2000 reads, "
                        "written to data/synthetic_small")
    p.add_argument("--out", help="output directory (default data/synthetic)")
    p.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="collection sizes")
    p.add_argument("--divergence", type=float, nargs="+", default=[0.001, 0.01])
    p.add_argument("--relations", nargs="+", choices=["uneven", "tree", "star"], default=["uneven"],
                   help="how genomes are related: a tree with unequal branches (default), a tree with equal "
                        "branches, or independent copies; one collection per relation and divergence")
    p.add_argument("--tree-height", type=int, default=4)
    p.add_argument("--branch-variation", type=float, default=0.8,
                   help="for 'uneven' collections: branch lengths are scaled by a factor in [1-V, 1+V] (default 0.8)")
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--num-reads", type=int, help="reads per set (default 10000)")
    p.add_argument("--read-length", type=int, default=150)
    p.add_argument("--read-error-rates", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--both-strands", action="store_true")
    p.add_argument("--num-queries", type=int, default=250, help="queries per class")
    p.add_argument("--query-length", type=int, default=100)
    p.add_argument("--mutations", type=int, default=1)
    p.add_argument("--no-truth", action="store_true")
    p.add_argument("--seed", type=int, default=447)
    _add_region(p)
    _add_model(p)
    p.set_defaults(func=cmd_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
