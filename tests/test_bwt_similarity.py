"""Tests for the BWT and run-count similarity. Run: python -m unittest discover tests"""

import random
import sys
import unittest
from operator import itemgetter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pangenome.bwt.build import build_bwt, bwt_string, count_runs  # noqa: E402
from pangenome.bwt.similarity import pairwise_runs, similarity, spearman  # noqa: E402
from pangenome.bwt.similarity_naive import similarity_naive  # noqa: E402
from simulator.genomes import pairwise_truth, simulate_tree_collection  # noqa: E402
from simulator.seq import random_sequence  # noqa: E402


def brute_bwt(genomes):
    """Reference implementation: sort every suffix of every genome directly.

    A suffix is its remaining bases followed by its genome's terminator;
    terminators sort before bases and in genome order.
    """
    rows = []
    for i, g in enumerate(genomes):
        for p in range(len(g) + 1):
            key = [("ACGT".index(c) + 1, 0) for c in g[p:]] + [(0, i)]
            rows.append((key, g[p - 1] if p > 0 else "$", i))
    rows.sort(key=itemgetter(0))
    return "".join(r[1] for r in rows), [r[2] for r in rows]


def runs(s):
    return sum(1 for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


class BwtTests(unittest.TestCase):
    def test_single_string(self):
        bwt, doc = build_bwt(["ACGACG"])
        self.assertEqual((bwt_string(bwt), doc.tolist()), brute_bwt(["ACGACG"]))

    def test_matches_brute_force(self):
        rng = random.Random(1)
        for trial in range(150):
            d = rng.randint(1, 5)
            base = random_sequence(rng, rng.randint(1, 40))
            genomes = []
            for _ in range(d):  # a mix of related, identical, repetitive, and unrelated genomes
                kind = rng.random()
                if kind < 0.3:
                    g = base
                elif kind < 0.6:
                    g = "".join(c if rng.random() > 0.1 else rng.choice("ACGT") for c in base)
                elif kind < 0.8:
                    g = rng.choice("ACGT") * rng.randint(1, 30)
                else:
                    g = random_sequence(rng, rng.randint(1, 40))
                genomes.append(g)
            bwt, doc = build_bwt(genomes)
            want_bwt, want_doc = brute_bwt(genomes)
            self.assertEqual(bwt_string(bwt), want_bwt, genomes)
            self.assertEqual(doc.tolist(), want_doc, genomes)
            self.assertEqual(count_runs(bwt), runs(want_bwt))

    def test_rejects_non_acgt(self):
        with self.assertRaises(ValueError):
            build_bwt(["ACGN"])


class SimilarityTests(unittest.TestCase):
    def test_pair_runs_equal_separately_built_bwts(self):
        rng = random.Random(2)
        base = random_sequence(rng, 60)
        genomes = [base, base, base[:30] + random_sequence(rng, 30), random_sequence(rng, 50), "A" * 20,
                   "".join(c if rng.random() > 0.05 else rng.choice("ACGT") for c in base)]
        bwt, doc = build_bwt(genomes)
        got = pairwise_runs(bwt, doc, len(genomes))
        for a in range(len(genomes)):
            self.assertEqual(got[a, a], runs(brute_bwt([genomes[a]])[0]))
            for b in range(a + 1, len(genomes)):
                self.assertEqual(got[a, b], runs(brute_bwt([genomes[a], genomes[b]])[0]), (a, b))
                self.assertEqual(got[b, a], got[a, b])

    def test_identical_and_unrelated(self):
        rng = random.Random(3)
        a, b = random_sequence(rng, 20_000), random_sequence(rng, 20_000)
        res = similarity([a, a, b])
        self.assertLess(res["distance"][0, 1], 0.001)   # identical
        self.assertGreater(res["distance"][0, 2], 0.95)  # unrelated
        self.assertLess(res["r_over_n"][0, 1], 0.6 * res["r_over_n"][0, 2])
        self.assertEqual(res["n"][0, 2], 40_002)

    def test_tracks_true_distance_on_a_tree(self):
        ref = random_sequence(random.Random(4), 60_000)
        genomes, logs = simulate_tree_collection(ref, 8, 0.02, seed=7, height=3, variation=0.8)
        res = similarity(genomes)
        truth = {(r["genome_a"], r["genome_b"]): r["distance"] for r in pairwise_truth(logs, len(ref))}
        pairs = sorted(truth)
        true = np.array([truth[p] for p in pairs])
        for key in ("distance", "r_over_n"):
            est = np.array([res[key][p] for p in pairs])
            self.assertGreater(spearman(est, true), 0.95, key)
        for a in range(8):  # nearest neighbour by run count is the true nearest
            others = [b for b in range(8) if b != a]
            estimated = [res["distance"][a, b] for b in others]
            actual = [truth[min(a, b), max(a, b)] for b in others]
            self.assertEqual(others[estimated.index(min(estimated))], others[actual.index(min(actual))])

    def test_naive_version_agrees(self):
        ref = random_sequence(random.Random(9), 8_000)
        genomes, _ = simulate_tree_collection(ref, 6, 0.03, seed=3, height=3, variation=0.8)
        genomes += [random_sequence(random.Random(10), 5_000), genomes[0]]
        fast, naive = similarity(genomes), similarity_naive(genomes)
        for key in ("runs", "n"):
            self.assertTrue(np.array_equal(fast[key], naive[key]), key)
        for key in ("r_over_n", "distance"):
            self.assertTrue(np.allclose(fast[key], naive[key]), key)
        self.assertNotIn("collection_runs", naive)

    def test_spearman(self):
        self.assertAlmostEqual(spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])), 1.0)
        self.assertAlmostEqual(spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])), -1.0)


if __name__ == "__main__":
    unittest.main()
