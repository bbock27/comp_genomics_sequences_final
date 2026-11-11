"""Tests for the synthetic data simulator. Run: python -m unittest discover tests"""

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.genomes import (  # noqa: E402
    DEL, INS, SNP, CoordMap, MutationModel, apply_variants, load_collection, mutate, read_variants,
    branch_rate, newick, pairwise_truth, simulate_collection, simulate_control, simulate_tree_collection, write_variants,
)
from simulator.queries import make_queries  # noqa: E402
from simulator.reads import sample_reads  # noqa: E402
from simulator.seq import random_sequence, read_fasta, revcomp, write_fasta  # noqa: E402
from simulator.truth import exact_presence, exact_presence_naive  # noqa: E402


def make_ref(n=50_000, seed=1):
    return random_sequence(random.Random(seed), n)


class MutateTests(unittest.TestCase):
    def test_log_replays_to_copy(self):
        ref = make_ref()
        for d in (0.0, 0.001, 0.01, 0.1, 0.4):
            copy, log = mutate(ref, random.Random(7), d)
            self.assertEqual(apply_variants(ref, log), copy, d)

    def test_alleles_and_positions(self):
        ref = make_ref()
        copy, log = mutate(ref, random.Random(3), 0.05)
        self.assertEqual({v.kind for v in log}, {SNP, INS, DEL})
        prev_end = -1
        for v in log:
            self.assertEqual(ref[v.ref_pos:v.ref_end], v.ref)
            self.assertEqual(copy[v.copy_pos:v.copy_end], v.alt)
            self.assertGreater(v.ref_pos, prev_end)  # >= 1 unchanged base between variants
            prev_end = v.ref_end
            if v.kind == SNP:
                self.assertNotEqual(v.ref, v.alt)
                self.assertEqual((len(v.ref), len(v.alt)), (1, 1))
            elif v.kind == INS:
                self.assertEqual(v.ref, "")
                self.assertTrue(1 <= len(v.alt) <= 10)
            else:
                self.assertEqual(v.alt, "")
                self.assertTrue(1 <= len(v.ref) <= 10)

    def test_rate_and_mix(self):
        ref = make_ref(400_000)
        _, log = mutate(ref, random.Random(5), 0.01)
        self.assertAlmostEqual(len(log) / len(ref), 0.01, delta=0.001)
        snps = sum(v.kind == SNP for v in log) / len(log)
        self.assertAlmostEqual(snps, 0.8, delta=0.03)

    def test_snp_only_model_keeps_length(self):
        ref = make_ref()
        copy, log = mutate(ref, random.Random(2), 0.02, MutationModel(1.0, 0.0, 0.0))
        self.assertEqual(len(copy), len(ref))
        self.assertEqual(sum(a != b for a, b in zip(ref, copy)), len(log))

    def test_coord_map(self):
        ref = make_ref(20_000)
        copy, log = mutate(ref, random.Random(11), 0.05)
        cm = CoordMap(log)
        deleted = {p for v in log if v.kind == DEL for p in range(v.ref_pos, v.ref_end)}
        snp_pos = {v.ref_pos for v in log if v.kind == SNP}
        for p in range(len(ref)):
            c = cm.ref_to_copy(p)
            if p in deleted:
                continue
            self.assertEqual(cm.copy_to_ref(c), p)
            if p not in snp_pos:
                self.assertEqual(copy[c], ref[p])
        self.assertEqual(cm.ref_to_copy(len(ref)), len(copy))

    def test_bad_model(self):
        with self.assertRaises(ValueError):
            MutationModel(0.5, 0.1, 0.1)


class CollectionTests(unittest.TestCase):
    def test_deterministic_and_nested(self):
        ref = make_ref()
        g8, l8 = simulate_collection(ref, 8, 0.01, seed=42)
        g3, l3 = simulate_collection(ref, 3, 0.01, seed=42)
        self.assertEqual(g8[:3], g3)
        self.assertEqual(l8[:3], l3)
        self.assertEqual(len(set(g8)), 8)
        self.assertNotEqual(simulate_collection(ref, 1, 0.01, seed=43)[0], g3[:1])
        for g, log in zip(g8, l8):
            self.assertEqual(apply_variants(ref, log), g)

    def test_control(self):
        c = simulate_control(5000, 4, seed=1)
        self.assertEqual([len(x) for x in c], [5000] * 4)
        self.assertEqual(len(set(c)), 4)
        self.assertEqual(c[:2], simulate_control(5000, 2, seed=1))

    def test_variant_and_fasta_roundtrip(self):
        ref = make_ref(5000)
        genomes, logs = simulate_collection(ref, 3, 0.05, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            write_variants(Path(tmp) / "v.tsv", logs)
            self.assertEqual(read_variants(Path(tmp) / "v.tsv", 3), logs)
            self.assertEqual(read_variants(Path(tmp) / "v.tsv", 2), logs[:2])
            write_fasta(Path(tmp) / "g.fa", [(f"g{i}", g) for i, g in enumerate(genomes)])
            self.assertEqual([s for _, s in read_fasta(Path(tmp) / "g.fa")], genomes)


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.ref = make_ref(200_000)
        self.d = 0.02
        self.genomes, self.logs = simulate_tree_collection(self.ref, 16, self.d, seed=42, height=4)

    def test_logs_replay_and_are_clean(self):
        for i, (g, log) in enumerate(zip(self.genomes, self.logs)):
            self.assertEqual(apply_variants(self.ref, log), g)
            prev_end = -1
            for v in log:
                self.assertEqual(v.genome, i)
                self.assertGreater(v.ref_pos, prev_end)
                self.assertEqual(g[v.copy_pos:v.copy_end], v.alt)
                prev_end = v.ref_end
            self.assertAlmostEqual(len(log) / len(self.ref), self.d, delta=0.002)

    def test_nested_and_deterministic(self):
        g5, l5 = simulate_tree_collection(self.ref, 5, self.d, seed=42, height=4)
        self.assertEqual((g5, l5), (self.genomes[:5], self.logs[:5]))

    def test_distances_follow_the_tree(self):
        dist = {(r["genome_a"], r["genome_b"]): r for r in pairwise_truth(self.logs, len(self.ref))}
        self.assertEqual(len(dist), 120)
        for (a, b), r in dist.items():
            level = (a ^ b).bit_length()  # 1 = siblings ... 4 = joined only at the root
            self.assertAlmostEqual(r["distance"], 2 * self.d * level / 4, delta=0.0025, msg=(a, b))
            self.assertAlmostEqual(r["shared_variants"] / len(self.ref), self.d * (4 - level) / 4, delta=0.0015)
        # each genome's nearest neighbour is its sibling
        for a in range(16):
            nearest = min((b for b in range(16) if b != a), key=lambda b: dist[min(a, b), max(a, b)]["distance"])
            self.assertEqual(nearest, a ^ 1)

    def test_second_tree_is_unrelated(self):
        g, logs = simulate_tree_collection(self.ref, 6, self.d, seed=42, height=2)
        rows = {(r["genome_a"], r["genome_b"]): r for r in pairwise_truth(logs, len(self.ref))}
        self.assertGreater(rows[0, 1]["shared_variants"], 500)
        self.assertLess(rows[0, 4]["shared_variants"], 30)  # only chance coincidences

    def test_newick(self):
        self.assertEqual(newick(4, 0.02, 2), "((g000:0.01,g001:0.01):0.01,(g002:0.01,g003:0.01):0.01);")
        self.assertEqual(newick(3, 0.02, 2), "((g000:0.01,g001:0.01):0.01,g002:0.02);")
        self.assertEqual(newick(1, 0.02, 2), "(g000:0.02);")
        self.assertEqual(newick(5, 0.02, 2).count("g00"), 5)
        text = newick(16, 0.02, 4)
        self.assertEqual((text.count("("), text.count(")"), text.count(",")), (15, 15, 15))

    def test_unequal_branches(self):
        d, h, v = 0.02, 4, 0.8
        genomes, logs = simulate_tree_collection(self.ref, 16, d, seed=42, height=h, variation=v)
        self.assertEqual(simulate_tree_collection(self.ref, 5, d, seed=42, height=h, variation=v),
                         (genomes[:5], logs[:5]))
        self.assertNotEqual(genomes, self.genomes)

        def length(level, node):
            return branch_rate(42, d, h, level, node, v)

        rates = [length(lv, nd) for lv in range(h) for nd in range(16 >> lv)]
        self.assertTrue(all(d / h * (1 - v) <= r <= d / h * (1 + v) for r in rates))
        self.assertGreater(max(rates) / min(rates), 3)

        # each genome's divergence is the sum of the branches above it
        depth = []
        for i, (g, log) in enumerate(zip(genomes, logs)):
            self.assertEqual(apply_variants(self.ref, log), g)
            expect = sum(length(lv, i >> lv) for lv in range(h))
            self.assertAlmostEqual(len(log) / len(self.ref), expect, delta=0.002)
            depth.append(len(log) / len(self.ref))
        self.assertGreater(max(depth) - min(depth), 0.005)  # genomes at different depths
        self.assertAlmostEqual(sum(depth) / 16, d, delta=0.004)

        # each pair's distance is the path between them through their common ancestor
        for r in pairwise_truth(logs, len(self.ref)):
            a, b = r["genome_a"], r["genome_b"]
            top = (a ^ b).bit_length()
            path = sum(length(lv, a >> lv) + length(lv, b >> lv) for lv in range(top))
            self.assertAlmostEqual(r["distance"], path, delta=0.0025, msg=(a, b))

        # the Newick file carries the same lengths
        text = newick(16, d, h, seed=42, variation=v)
        self.assertIn(f"g000:{length(0, 0):.6g}", text)
        self.assertIn(f"g015:{length(0, 15):.6g}", text)
        with self.assertRaises(ValueError):
            simulate_tree_collection(self.ref, 2, d, seed=1, variation=1.0)

    def test_recombinants_in_tree_mode(self):
        queries = make_queries(self.genomes, self.logs, random.Random(3), 300, 60, classes=("recombinant",))
        self.assertEqual(len(queries), 300)
        maps = [CoordMap(log) for log in self.logs]
        for q in queries:
            a, b, bp = self.genomes[q.genome], self.genomes[q.genome2], q.breakpoint
            self.assertEqual(a[q.offset:q.offset + bp] + b[q.offset2:q.offset2 + 60 - bp], q.seq)
            self.assertEqual(maps[q.genome].copy_to_ref(q.offset + bp), maps[q.genome2].copy_to_ref(q.offset2))
        truth = exact_presence_naive([q.seq for q in queries], self.genomes)
        self.assertEqual(sum(bool(t) for t in truth), 0)


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.ref = make_ref()
        self.genomes, self.logs = simulate_collection(self.ref, 4, 0.01, seed=5)

    def test_exact_reads(self):
        reads = sample_reads(self.genomes, random.Random(1), 500, 100, logs=self.logs)
        self.assertEqual(len({r.id for r in reads}), 500)
        self.assertEqual({r.genome for r in reads}, {0, 1, 2, 3})
        for r in reads:
            self.assertEqual(self.genomes[r.genome][r.offset:r.offset + 100], r.seq)
            self.assertEqual((r.strand, r.errors), ("+", 0))
            self.assertEqual(CoordMap(self.logs[r.genome]).copy_to_ref(r.offset), r.ref_offset)

    def test_errors_and_strands(self):
        reads = sample_reads(self.genomes, random.Random(2), 2000, 100, error_rate=0.02, both_strands=True)
        total = 0
        for r in reads:
            src = self.genomes[r.genome][r.offset:r.offset + 100]
            if r.strand == "-":
                src = revcomp(src)
            self.assertEqual(sum(a != b for a, b in zip(src, r.seq)), r.errors)
            total += r.errors
        self.assertAlmostEqual(total / (2000 * 100), 0.02, delta=0.003)
        self.assertEqual({r.strand for r in reads}, {"+", "-"})
        self.assertEqual(reads[0].ref_offset, -1)


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.ref = make_ref(100_000)
        self.genomes, self.logs = simulate_collection(self.ref, 4, 0.01, seed=5)
        self.queries = make_queries(self.genomes, self.logs, random.Random(3), 200, 60, mutations=2)

    def by_class(self, cls):
        return [q for q in self.queries if q.cls == cls]

    def test_counts_and_lengths(self):
        for cls in ("present", "mutated", "random", "recombinant"):
            self.assertEqual(len(self.by_class(cls)), 200, cls)
        self.assertTrue(all(len(q.seq) == 60 for q in self.queries))
        self.assertEqual(len({q.id for q in self.queries}), 800)

    def test_present_and_mutated(self):
        for q in self.by_class("present"):
            self.assertEqual(self.genomes[q.genome][q.offset:q.offset + 60], q.seq)
        for q in self.by_class("mutated"):
            src = self.genomes[q.genome][q.offset:q.offset + 60]
            diffs = [f"{i}:{a}>{b}" for i, (a, b) in enumerate(zip(src, q.seq)) if a != b]
            self.assertEqual(";".join(diffs), q.edits)
            self.assertEqual(len(diffs), 2)

    def test_recombinants(self):
        maps = [CoordMap(log) for log in self.logs]
        for q in self.by_class("recombinant"):
            a, b, bp = self.genomes[q.genome], self.genomes[q.genome2], q.breakpoint
            self.assertNotEqual(q.genome, q.genome2)
            self.assertEqual(a[q.offset:q.offset + bp], q.seq[:bp])
            self.assertEqual(b[q.offset2:q.offset2 + 60 - bp], q.seq[bp:])
            # the two halves meet at the same reference position
            self.assertEqual(maps[q.genome].copy_to_ref(q.offset + bp), maps[q.genome2].copy_to_ref(q.offset2))
            # each side carries a variant of its own genome, variant_gap apart
            ra = maps[q.genome].copy_to_ref(q.offset + bp)
            va = [v for v in self.logs[q.genome] if v.ref_end == ra]
            vb = [v for v in self.logs[q.genome2] if v.ref_pos == ra + q.variant_gap]
            self.assertEqual((len(va), len(vb)), (1, 1))
            self.assertGreaterEqual(q.variant_gap, 1)
            self.assertGreater(va[0].copy_pos, q.offset)
            self.assertLess(vb[0].copy_end, q.offset2 + 60 - bp)
        # with random genomes of this size, a recombinant is in no genome
        truth = exact_presence_naive([q.seq for q in self.by_class("recombinant")], self.genomes)
        self.assertEqual(sum(bool(t) for t in truth), 0)

    def test_gap_filter(self):
        qs = make_queries(self.genomes, self.logs, random.Random(4), 50, 60, min_gap=20, max_gap=30,
                          classes=("recombinant",))
        self.assertEqual(len(qs), 50)
        self.assertTrue(all(20 <= q.variant_gap <= 30 for q in qs))

    def test_recombinant_skipped_when_impossible(self):
        one = make_queries(self.genomes[:1], self.logs[:1], random.Random(1), 10, 60)
        self.assertEqual(sum(q.cls == "recombinant" for q in one), 0)
        ctrl = simulate_control(5000, 3, seed=1)
        self.assertEqual(len(make_queries(ctrl, [[], [], []], random.Random(1), 10, 60)), 30)


class TruthTests(unittest.TestCase):
    def test_matches_naive(self):
        ref = make_ref(30_000)
        # high-copy repeat plus near-identical genomes: many anchor collisions
        ref = ref[:10_000] + ref[2000:2600] * 3 + ref[10_000:]
        genomes, logs = simulate_collection(ref, 5, 0.01, seed=8)
        queries = make_queries(genomes, logs, random.Random(6), 150, 50)
        patterns = [q.seq for q in queries] + [genomes[0][:40], genomes[4][-45:], ref[2000:2050]]
        fast = exact_presence(patterns, genomes)
        self.assertEqual(fast, exact_presence_naive(patterns, genomes))
        for q, found in zip(queries, fast):
            if q.cls == "present":
                self.assertIn(q.genome, found)
            if q.cls == "random":
                self.assertEqual(found, [])


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "simulator", *map(str, args)], cwd=ROOT, check=True,
                              capture_output=True, text=True)

    def test_suite_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_fasta(tmp / "ref.fa", [("toy reference", make_ref(30_000))])
            common = ["--reference", tmp / "ref.fa", "--copies", 1, 2, 4, "--divergence", 0.01,
                      "--num-reads", 200, "--num-queries", 40, "--relations", "star", "tree", "uneven"]
            self.run_cli("suite", *common, "--out", tmp / "a")
            self.run_cli("suite", *common, "--out", tmp / "b")
            files = sorted(p.relative_to(tmp / "a") for p in (tmp / "a").rglob("*") if p.is_file())
            self.assertIn(Path("star_d0.01/n004/queries.tsv"), files)
            self.assertIn(Path("tree_d0.01/n004/reads_err0.tsv"), files)
            self.assertIn(Path("tree_d0.01/tree.nwk"), files)
            self.assertIn(Path("uneven_d0.01/n004/queries.tsv"), files)
            self.assertIn(Path("uneven_d0.01/tree.nwk"), files)
            self.assertIn(Path("star_d0.01/pairwise.tsv"), files)
            self.assertIn(Path("control/n002/reads_err0.01.fasta"), files)
            for f in files:
                if f.name != "suite.json":
                    self.assertEqual((tmp / "a" / f).read_bytes(), (tmp / "b" / f).read_bytes(), f)

            ref = read_fasta(tmp / "ref.fa")[0][1]
            for label in ("star_d0.01", "tree_d0.01", "uneven_d0.01"):
                col = load_collection(tmp / "a" / label)
                self.assertEqual(len(col.genomes), 4)
                for g, log in zip(col.genomes, col.logs):
                    self.assertEqual(apply_variants(ref, log), g)
            col = load_collection(tmp / "a" / "star_d0.01")

            # truth columns are restricted to the genomes of each sub-collection
            rows = [line.split("\t") for line in (tmp / "a/star_d0.01/n002/reads_err0.tsv").read_text().splitlines()]
            self.assertEqual(rows[0][-2:], ["present_in", "source_in"])
            for row in rows[1:]:
                self.assertIn(row[1], row[-1].split(","))
                self.assertEqual(row[-1], row[-2])
                self.assertTrue(set(row[-1].split(",")) <= {"g000", "g001"})

            # reads with errors: source_in is the truth for the error-free original
            genomes = load_collection(tmp / "a" / "tree_d0.01").genomes
            rows = [line.split("\t") for line in (tmp / "a/tree_d0.01/n004/reads_err0.01.tsv").read_text().splitlines()]
            differing = 0
            for row in rows[1:]:
                g, off, n = int(row[1][1:]), int(row[2]), int(row[4])
                window = genomes[g][off:off + n]
                expect = ",".join(f"g{i:03d}" for i, x in enumerate(genomes) if window in x)
                self.assertEqual(row[-1], expect)
                self.assertIn(row[1], row[-1].split(","))
                differing += row[-1] != row[-2]
                if row[5] == "0":
                    self.assertEqual(row[-1], row[-2])
            self.assertGreater(differing, 50)

            # --ref-start / --ref-length take a slice of the reference
            self.run_cli("genomes", "--reference", tmp / "ref.fa", "--copies", 1, "--divergence", 0, "--ref-start", 100,
                         "--ref-length", 500, "--out", tmp / "slice")
            self.assertEqual(load_collection(tmp / "slice").genomes, [ref[100:600]])

            # standalone commands agree with the library
            self.run_cli("genomes", "--reference", tmp / "ref.fa", "--copies", 2, "--divergence", 0.01,
                         "--relation", "star", "--out", tmp / "c")
            self.assertEqual(load_collection(tmp / "c").genomes, col.genomes[:2])

            # the default is the uneven tree, for both 'genomes' and 'suite'
            self.run_cli("genomes", "--reference", tmp / "ref.fa", "--copies", 4, "--divergence", 0.01, "--out", tmp / "u")
            uneven = load_collection(tmp / "a" / "uneven_d0.01")
            self.assertEqual(load_collection(tmp / "u").genomes, uneven.genomes)
            self.assertEqual(load_collection(tmp / "u").manifest["branch_variation"], 0.8)
            self.run_cli("suite", "--reference", tmp / "ref.fa", "--copies", 1, 2, "--divergence", 0.01,
                         "--num-reads", 20, "--num-queries", 5, "--out", tmp / "dflt")
            self.assertEqual(sorted(p.name for p in (tmp / "dflt").iterdir() if p.is_dir()), ["control", "uneven_d0.01"])
            self.run_cli("reads", "--collection", tmp / "c", "--num", 50, "--out", tmp / "c" / "reads")
            self.run_cli("queries", "--collection", tmp / "c", "--num-per-class", 20, "--out", tmp / "c" / "queries")
            self.assertEqual(len(read_fasta(tmp / "c" / "queries.fasta")), 80)


if __name__ == "__main__":
    unittest.main()
