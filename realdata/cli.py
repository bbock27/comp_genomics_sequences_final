"""Command-line interface: python -m realdata <command>

Commands
--------
sample      list every matching assembly at NCBI and pick a seeded random
            sample; writes the accession list that pins the dataset
download    download the assemblies in the accession list
preprocess  chromosome only, no ambiguous bases, same strand and start
samples     reads and query patterns with ground truth for each collection
            size, in the same files and columns as the synthetic data
all         download + preprocess + samples (uses the existing accession list)

The accession list (realdata/<name>_accessions.tsv) is tracked in git. NCBI's
catalogue changes over time, so re-running ``sample`` may select different
genomes; ``download`` and ``preprocess`` always use the tracked list.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from simulator import __version__
from simulator.cli import _samples_for
from simulator.genomes import load_collection
from simulator.seq import rng_for, sha256_of, write_fasta

from .preprocess import load_anchor, process_assembly

PACKAGE = Path(__file__).parent
PRESETS = {
    "ecoli": {"taxon": "Escherichia coli", "num": 50},
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tool(name: str, tools_dir: str) -> str:
    local = Path(tools_dir) / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"'{name}' not found in {tools_dir}/ or on PATH (NCBI datasets command-line tools)")
    return found


def accession_file(name: str) -> Path:
    return PACKAGE / f"{name}_accessions.tsv"


def read_accessions(name: str) -> list[dict]:
    path = accession_file(name)
    if not path.exists():
        raise SystemExit(f"{path} not found; run: python -m realdata sample --name {name}")
    lines = [l.rstrip("\n").split("\t") for l in path.read_text().splitlines() if l and not l.startswith("#")]
    return [dict(zip(lines[0], row)) for row in lines[1:]]


# --------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    out = Path(args.data) / args.name
    out.mkdir(parents=True, exist_ok=True)
    listing = out / "all_assemblies.tsv"
    log(f"listing complete RefSeq assemblies of {args.taxon!r} ...")
    summary = subprocess.Popen(
        [tool("datasets", args.tools_dir), "summary", "genome", "taxon", args.taxon, "--assembly-level", "complete",
         "--assembly-source", "RefSeq", "--exclude-atypical", "--as-json-lines"],
        stdout=subprocess.PIPE)
    with open(listing, "w") as fh:
        subprocess.run(
            [tool("dataformat", args.tools_dir), "tsv", "genome", "--fields",
             "accession,organism-infraspecific-strain,assmstats-total-sequence-len"],
            stdin=summary.stdout, stdout=fh, check=True)
    if summary.wait() != 0:
        raise SystemExit("datasets summary failed")

    rows = [l.split("\t") for l in listing.read_text().splitlines()[1:] if l]
    rows = sorted({r[0]: r for r in rows}.values())  # unique, in a fixed order
    if len(rows) < args.num:
        raise SystemExit(f"only {len(rows)} assemblies match; cannot sample {args.num}")
    picked = rng_for(args.seed, "real", args.name).sample(rows, args.num)
    with open(accession_file(args.name), "w") as fh:
        fh.write(f"# {args.num} of {len(rows)} complete RefSeq assemblies of {args.taxon}, random sample, seed {args.seed}\n")
        fh.write("accession\tstrain\ttotal_length\n")
        for r in picked:
            fh.write("\t".join(r) + "\n")
    log(f"sampled {args.num} of {len(rows)} -> {accession_file(args.name)}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    out = Path(args.data) / args.name
    out.mkdir(parents=True, exist_ok=True)
    accessions = [r["accession"] for r in read_accessions(args.name)]
    raw = out / "raw"
    have = {p.parent.name for p in raw.glob("ncbi_dataset/data/*/*.fna")}
    missing = [a for a in accessions if a not in have]
    if not missing:
        log(f"all {len(accessions)} assemblies already in {raw}")
        return 0
    wanted = out / "to_download.txt"
    wanted.write_text("\n".join(missing) + "\n")
    archive = out / "download.zip"
    log(f"downloading {len(missing)} assemblies ...")
    subprocess.run([tool("datasets", args.tools_dir), "download", "genome", "accession", "--inputfile", str(wanted),
                    "--include", "genome", "--filename", str(archive), "--no-progressbar"], check=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(raw)
    archive.unlink()
    wanted.unlink()
    have = {p.parent.name for p in raw.glob("ncbi_dataset/data/*/*.fna")}
    still = [a for a in accessions if a not in have]
    if still:
        raise SystemExit(f"{len(still)} assemblies were not delivered: {', '.join(still[:5])} ...")
    log(f"downloaded to {raw}")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    out = Path(args.data) / args.name
    rows = read_accessions(args.name)
    anchor = load_anchor()
    records, table, skipped = [], [], []
    for row in rows:
        acc = row["accession"]
        files = sorted((out / "raw" / "ncbi_dataset" / "data" / acc).glob("*.fna"))
        if not files:
            raise SystemExit(f"{acc} is not downloaded; run: python -m realdata download --name {args.name}")
        p = process_assembly(files[0], anchor)
        status = "ok"
        if p.anchor_votes < args.min_votes:
            status = "skipped: dnaA not found"
        elif p.ambiguous > args.max_ambiguous:
            status = f"skipped: {p.ambiguous} ambiguous bases"
        if status == "ok":
            name = f"g{len(records):03d}"
            records.append((f"{name} {acc} strain={row['strain'] or '-'}", p.seq))
        else:
            name = "-"
            skipped.append(acc)
        table.append([name, acc, row["strain"] or "-", len(p.seq), p.records, p.other_bases, p.ambiguous, p.strand,
                      p.rotation, p.anchor_votes, status])
        log(f"  {name:5s}{acc}  {len(p.seq):>8} bp  strand {p.strand}  rotated {p.rotation:>8}  "
            f"plasmid records {p.records - 1}  ambiguous {p.ambiguous}  {'' if status == 'ok' else status}")

    write_fasta(out / "genomes.fasta", records)
    cols = ["genome", "accession", "strain", "length", "records", "other_bases_dropped", "ambiguous_removed", "strand",
            "rotation", "anchor_votes", "status"]
    with open(out / "genomes.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in table:
            fh.write("\t".join(map(str, r)) + "\n")
    manifest = {
        "kind": "real", "name": args.name, "copies": len(records), "skipped": skipped,
        "simulator_version": __version__,
        "genome_lengths": [len(s) for _, s in records],
        "preprocessing": "longest record only; non-ACGT removed; reverse-complemented and rotated to start at dnaA",
        "anchor_sha256": sha256_of(anchor),
        "note": "Genomes are in sampled (random) order, so the first n records are a random n-genome subset.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"wrote {out / 'genomes.fasta'}: {len(records)} genomes, {sum(len(s) for _, s in records)} bases"
        + (f"; skipped {len(skipped)}" if skipped else ""))
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    """Reads and queries for each collection size, exactly as the simulator's
    suite writes them for a synthetic collection."""
    out = Path(args.data) / args.name
    if not (out / "genomes.fasta").exists():
        raise SystemExit(f"{out / 'genomes.fasta'} not found; run: python -m realdata preprocess --name {args.name}")
    col = load_collection(out)
    total = len(col.genomes)
    sizes = sorted({n for n in args.copies if n < total} | {total})
    log(f"sampling reads and queries from {total} genomes for sizes {sizes} ...")
    for n, name, items, writer in _samples_for(col, args.name, sizes, args):
        sub = out / f"n{n:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        writer(sub / name, items)
    manifest = col.manifest
    manifest["samples"] = {
        "sizes": sizes, "seed": args.seed, "num_reads": args.num_reads, "read_length": args.read_length,
        "read_error_rates": args.read_error_rates, "both_strands": args.both_strands,
        "num_queries_per_class": args.num_queries, "query_length": args.query_length, "mutations": args.mutations,
        "query_classes": ["present", "mutated", "random"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"wrote {', '.join(f'n{n:03d}' for n in sizes)} under {out}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    return cmd_download(args) or cmd_preprocess(args) or cmd_samples(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m realdata", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", default="ecoli", help="dataset name (default ecoli)")
        p.add_argument("--data", default="data/real", help="where downloads and output go")
        p.add_argument("--tools-dir", default="ncbi_tools", help="directory holding datasets and dataformat")

    p = sub.add_parser("sample", help="pick a seeded random sample of assemblies")
    common(p)
    p.add_argument("--taxon", help="NCBI taxon name (default: from the dataset name)")
    p.add_argument("--num", type=int, help="number of genomes (default 50)")
    p.add_argument("--seed", type=int, default=447)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("download", help="download the sampled assemblies")
    common(p)
    p.set_defaults(func=cmd_download)

    def preprocess_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-ambiguous", type=int, default=100, help="skip genomes with more non-ACGT characters")
        p.add_argument("--min-votes", type=int, default=20, help="skip genomes with fewer agreeing dnaA k-mers")

    def sample_options(p: argparse.ArgumentParser) -> None:
        # same defaults as 'python -m simulator suite'
        p.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                       help="collection sizes; the full collection is always included")
        p.add_argument("--num-reads", type=int, default=10000)
        p.add_argument("--read-length", type=int, default=150)
        p.add_argument("--read-error-rates", type=float, nargs="+", default=[0.0, 0.01])
        p.add_argument("--both-strands", action="store_true")
        p.add_argument("--num-queries", type=int, default=250, help="queries per class")
        p.add_argument("--query-length", type=int, default=100)
        p.add_argument("--mutations", type=int, default=1)
        p.add_argument("--no-truth", action="store_true")
        p.add_argument("--seed", type=int, default=447)

    p = sub.add_parser("preprocess", help="clean and orient the downloaded assemblies")
    common(p)
    preprocess_options(p)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("samples", help="reads and queries with ground truth, as for synthetic data")
    common(p)
    sample_options(p)
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("all", help="download + preprocess + samples")
    common(p)
    preprocess_options(p)
    sample_options(p)
    p.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample":
        preset = PRESETS.get(args.name, {})
        args.taxon = args.taxon or preset.get("taxon")
        args.num = args.num or preset.get("num", 50)
        if not args.taxon:
            raise SystemExit("--taxon is required for a dataset name without a preset")
    return args.func(args)
