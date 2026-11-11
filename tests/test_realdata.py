"""Tests for real-data preprocessing. Run: python -m unittest discover tests"""

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from realdata.cli import main  # noqa: E402
from realdata.preprocess import load_anchor, normalize, process_assembly  # noqa: E402
from simulator.cli import main as simulator_main  # noqa: E402
from simulator.genomes import load_collection, mutate  # noqa: E402
from simulator.seq import random_sequence, revcomp, write_fasta  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.anchor = load_anchor()
        rng = random.Random(1)
        self.rest = random_sequence(rng, 60_000)
        self.canonical = self.anchor + self.rest  # starts at the anchor, forward strand

    def test_anchor_is_dnaA(self):
        self.assertEqual(len(self.anchor), 1404)
        self.assertEqual(self.anchor[:3], "GTG")

    def test_recovers_rotation_and_strand(self):
        n = len(self.canonical)
        for cut in (0, 1, 700, 12_345, n - 5):
            rotated = self.canonical[cut:] + self.canonical[:cut]
            for seq, strand in ((rotated, "+"), (revcomp(rotated), "-")):
                got = normalize(seq, self.anchor)
                self.assertEqual(got.seq, self.canonical, (cut, strand))
                self.assertEqual(got.strand, strand)
                self.assertGreater(got.anchor_votes, 1000)

    def test_diverged_anchor(self):
        # a dnaA copy 3% diverged, with indels, still pins the exact start
        tail, log = mutate(self.anchor[40:], random.Random(5), 0.03)
        self.assertTrue(any(v.kind != "SNP" for v in log))
        variant = self.anchor[:40] + tail
        genome = self.rest[:20_000] + revcomp(variant) + self.rest[20_000:]
        got = normalize(genome, self.anchor)
        self.assertEqual(got.strand, "-")
        self.assertEqual(got.seq[:40], self.anchor[:40])
        self.assertEqual(len(got.seq), len(genome))
        self.assertEqual(sorted(got.seq), sorted(revcomp(genome)))

    def test_missing_anchor(self):
        got = normalize(self.rest, self.anchor)
        self.assertEqual((got.seq, got.anchor_votes, got.rotation), (self.rest, 0, 0))

    def test_process_assembly(self):
        rotated = self.canonical[30_000:] + self.canonical[:30_000]
        messy = rotated[:100] + "NNNnn" + rotated[100:].lower()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.fna"
            write_fasta(path, [("plasmid pX", "ACGT" * 500), ("chromosome", messy), ("plasmid pY", "GG" * 50)])
            p = process_assembly(path, self.anchor)
        self.assertEqual(p.seq, self.canonical)
        self.assertEqual((p.record, p.records, p.other_bases, p.ambiguous), ("chromosome", 3, 2100, 5))
        self.assertEqual(set(p.seq), set("ACGT"))


class SamplesTests(unittest.TestCase):
    def test_same_files_and_columns_as_synthetic(self):
        rng = random.Random(3)
        base = random_sequence(rng, 20_000)
        genomes = [mutate(base, random.Random(i), 0.02)[0] for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            real = tmp / "real" / "toy"
            real.mkdir(parents=True)
            write_fasta(real / "genomes.fasta", [(f"g{i:03d} ACC{i} strain=x", g) for i, g in enumerate(genomes)])
            (real / "manifest.json").write_text('{"kind": "real", "copies": 5}')
            opts = ["--num-reads", "60", "--num-queries", "10"]
            self.assertEqual(main(["samples", "--name", "toy", "--data", str(tmp / "real"), *opts]), 0)

            write_fasta(tmp / "ref.fa", [("ref", base)])
            simulator_main(["suite", "--reference", str(tmp / "ref.fa"), "--out", str(tmp / "syn"), "--copies", "1", "4",
                            "--divergence", "0.02", "--no-control", *opts])
            syn = tmp / "syn" / "uneven_d0.02"

            self.assertEqual(sorted(p.name for p in real.iterdir() if p.is_dir()), ["n001", "n002", "n004", "n005"])
            files = sorted(p.name for p in (syn / "n004").iterdir())
            self.assertEqual(sorted(p.name for p in (real / "n004").iterdir()), files)
            for f in files:
                if f.endswith(".tsv"):  # identical columns
                    self.assertEqual((real / "n004" / f).read_text().splitlines()[0],
                                     (syn / "n004" / f).read_text().splitlines()[0], f)

            # truth is correct and restricted to the sub-collection
            for line in (real / "n002" / "reads_err0.tsv").read_text().splitlines()[1:]:
                row = line.split("\t")
                g, off = int(row[1][1:]), int(row[2])
                window = genomes[g][off:off + 150]
                self.assertEqual(row[7], ",".join(f"g{i:03d}" for i in range(2) if window in genomes[i]))
                self.assertEqual(row[6], "-1")  # no reference coordinate for real genomes
            classes = {l.split("\t")[1] for l in (real / "n005" / "queries.tsv").read_text().splitlines()[1:]}
            self.assertEqual(classes, {"present", "mutated", "random"})
            self.assertEqual(load_collection(real).manifest["samples"]["sizes"], [1, 2, 4, 5])


if __name__ == "__main__":
    unittest.main()
