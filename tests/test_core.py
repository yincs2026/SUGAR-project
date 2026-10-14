import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sugar.data import demo, prepare, proxy_graph
from sugar.features import FeatureStore
from sugar.offline import Detector, build_snapshots, select_snapshot, cosine, verify_manifest
from sugar.semantics import validate, ITEM_SCHEMA, PROFILE_SCHEMA, SemanticBackend, topic_clusters
from sugar.utils import load_config, load_json, save_json, digest
from sugar.model import SUGAR, SASBlock
from sugar.train import Collator, Queries, sample_negatives, evaluate, rank_metrics
from sugar.experiments import js, drift_analysis


def config():
    c = load_config(Path(__file__).resolve().parents[1] / "configs/paper.yaml")
    c.update({"semantic_backend": "mock", "allow_nonpaper": True, "dim": 16, "max_len": 80,
              "dropout": 0, "neighbors": 3, "checkpoint": 5, "lag": 2, "k": 5,
              "edit_size": 5, "cooldown": 10, "alignment_negatives": 8, "batch_size": 16})
    return c


class DetectorTests(unittest.TestCase):
    def test_two_strikes(self):
        d = Detector(threshold="global", tau=0.7)
        self.assertFalse(d.check(0.6, 20)[0])
        self.assertEqual(d.state, "Suspect")
        self.assertTrue(d.check(0.5, 40)[0])
        self.assertEqual(d.state, "Revising")

    def test_recovery_resets_strikes(self):
        d = Detector(threshold="global")
        d.check(0.1, 20)
        d.check(0.9, 40)
        self.assertFalse(d.check(0.1, 60)[0])

    def test_cooldown_begins_at_commit(self):
        d = Detector(threshold="global", cooldown=50)
        d.check(0.1, 20)
        d.check(0.1, 40)
        self.assertFalse(d.check(0.1, 60)[0])
        d.committed(70)
        self.assertFalse(d.check(0.1, 119)[0])
        self.assertFalse(d.check(0.1, 120)[0])
        self.assertTrue(d.check(0.1, 140)[0])

    def test_threshold_uses_past_only(self):
        d = Detector(warmup=2, xi=1)
        d.check(0.8, 20)
        d.check(1.0, 40)
        self.assertAlmostEqual(d.check(-1.0, 60)[1], 0.8)

    def test_single_strike(self):
        self.assertTrue(Detector(threshold="global", single=True).check(0, 20)[0])

    def test_equality_is_not_violation(self):
        self.assertFalse(Detector(threshold="global", tau=0.7).check(0.7, 20)[0])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.c = config()
        demo(cls.root / "raw")
        cls.data = prepare(cls.root / "raw", cls.root / "prepared", cls.c)
        build_snapshots(cls.data, cls.root / "artifacts", cls.c, cls.root / "cache")
        cls.store = FeatureStore(cls.data, cls.root / "artifacts", cls.c)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_split(self):
        u = next(iter(self.data["sequences"]))
        seq = self.data["sequences"][u]
        self.assertEqual(dict(Queries(self.data, "validation").entries)[u], len(seq) - 2)
        self.assertEqual(dict(Queries(self.data, "test").entries)[u], len(seq) - 1)
        self.assertEqual(len(Queries(self.data, "train")), sum(len(s) - 3 for s in self.data["sequences"].values()))

    def test_core_counts(self):
        self.assertEqual(self.data["counts"]["items"], 18)
        for seq in self.data["sequences"].values():
            self.assertGreaterEqual(len(seq), self.c["core"])

    def test_snapshot_source_is_training_only(self):
        for u, snapshots in self.store.snapshots.items():
            for snap in snapshots:
                self.assertLessEqual(snap["source_pos"], len(self.data["sequences"][u]) - 2)
                self.assertEqual(snap["commit_pos"], snap["source_pos"] + self.c["lag"])

    def test_no_profile_before_commit(self):
        u = next(iter(self.data["sequences"]))
        self.assertFalse(self.store.query(u, 6)["present"])
        self.assertTrue(self.store.query(u, 7)["present"])

    def test_temporal_audit(self):
        report = self.store.audit()
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_prediction_positions"], 12 * 64)

    def test_neighbor_source_times(self):
        for u in self.data["sequences"]:
            f = self.store.query(u, 42)
            self.assertTrue(all(t < f["target_time"] for t in f["neighbor_source_times"]))

    def test_no_heldout_recent_behavior(self):
        u = next(iter(self.data["sequences"]))
        seq = self.data["sequences"][u]
        a, b = self.store.query(u, len(seq) - 2), self.store.query(u, len(seq) - 1)
        np.testing.assert_array_equal(a["recent"], b["recent"])

    def test_evidence_uses_event_indices(self):
        u = next(iter(self.data["sequences"]))
        for snap in self.store.snapshots[u]:
            self.assertEqual(snap["n"], len(set(snap["evidence_indices"])))
            self.assertTrue(all(i < snap["source_pos"] for i in snap["evidence_indices"]))
        self.assertGreater(self.store.snapshots[u][-1]["n"], len({r[0] for r in self.data["sequences"][u][:-2]}))

    def test_commit_at_target_excluded(self):
        u = next(iter(self.data["sequences"]))
        snap = self.store.snapshots[u][0]
        self.assertIsNone(select_snapshot([snap], snap["commit_time"]))
        self.assertIsNotNone(select_snapshot([snap], snap["commit_time"] + 1))

    def test_uncommittable_snapshot_excluded(self):
        u = next(iter(self.data["sequences"]))
        snap = {**self.store.snapshots[u][-1], "commit_time": None}
        self.assertIsNone(select_snapshot([snap], 1e30))

    def test_corrupt_source_rejected(self):
        u = next(iter(self.data["sequences"]))
        snap = {**self.store.snapshots[u][0], "source_time": 1e29}
        with self.assertRaises(AssertionError):
            select_snapshot([snap], 1e20)

    def test_wrong_dataset_rejected(self):
        data = copy.deepcopy(self.data)
        data["fingerprint"] = "bad"
        with self.assertRaises(ValueError):
            FeatureStore(data, self.root / "artifacts", self.c)

    def test_tampered_data_rejected(self):
        data = copy.deepcopy(self.data)
        data["sequences"]["1"][0][0] = 999
        with self.assertRaises(ValueError):
            FeatureStore(data, self.root / "artifacts", self.c)

    def test_leakage_ablation_is_detected(self):
        store = FeatureStore(self.data, self.root / "artifacts", {**self.c, "leaky_snapshots": True})
        report = store.audit()
        self.assertFalse(report["passed"])
        self.assertTrue(report["intentional_leakage_ablation"])
        self.assertGreater(report["temporal_violations"], 0)

    def test_wrong_offline_config_rejected(self):
        c = {**self.c, "lag": 9}
        with self.assertRaises(ValueError):
            FeatureStore(self.data, self.root / "artifacts", c)

    def test_frozen_artifacts_have_no_gradient(self):
        batch = Collator(self.data, self.store, self.c)(Queries(self.data, "train").entries[80:84])
        model = SUGAR(18, self.store.dim, self.c, self.store.embeddings)
        h, _ = model(batch[0], batch[1])
        h.sum().backward()
        self.assertFalse(model.semantics.requires_grad)
        self.assertIsNone(model.semantics.grad)

    def test_negative_exclusion(self):
        queries = Queries(self.data, "train").entries[:5]
        negatives = sample_negatives(self.data, queries, 100, np.random.default_rng(2))
        for (u, _), row in zip(queries, negatives):
            seen = {r[0] for r in self.data["sequences"][u][:-2]}
            self.assertFalse(seen & set(row.tolist()))

    def test_stress_available(self):
        for mode in ["stale", "corrupted"]:
            store = FeatureStore(self.data, self.root / "artifacts", self.c, mode)
            self.assertTrue(store.audit()["passed"])

    def test_proxy_graph_ignores_heldout(self):
        a = proxy_graph(copy.deepcopy(self.data))["graph"]
        altered = copy.deepcopy(self.data)
        for seq in altered["sequences"].values():
            seq[-1][0], seq[-2][0] = 999, 998
        self.assertEqual(a, proxy_graph(altered)["graph"])

    def test_delayed_job_does_not_revert_profile(self):
        c = {**self.c, "lag": 12}
        root = self.root / "longlag"
        build_snapshots(self.data, root, c, self.root / "cache")
        store = FeatureStore(self.data, root, c)
        u = next(iter(self.data["sequences"]))
        for pos in range(17, len(self.data["sequences"][u])):
            self.assertTrue(store.query(u, pos)["present"])

    def test_weighted_profile_embedding(self):
        b = SemanticBackend(self.c, self.root / "cache")
        p = {"dominant_interests": [{"topic": "sport", "weight": 2}, {"topic": "books", "weight": 1}],
             "emerging_interests": [], "content_preferences": [], "excluded_topics": []}
        v = (b.encode(["sport", "books"]) * np.array([2, 1])[:, None]).sum(0)
        np.testing.assert_allclose(b.profile_vector(p), v / np.linalg.norm(v), atol=1e-6)

    def test_cache_reused(self):
        b = SemanticBackend(self.c, self.root / "cache")
        payload = self.data["catalog"]["1"]
        x = b.generate("item", payload)
        self.assertTrue(b.usage[-1]["cache_hit"])
        self.assertEqual(x, b.generate("item", payload))

    def test_full_catalog_metrics(self):
        c = {**self.c, "model": "sasrec", "alignment_weight": 0}
        model = SUGAR(18, self.store.dim, c, self.store.embeddings)
        result = evaluate(model, self.data, self.store, c, chunk=3)
        self.assertEqual(len(result["users"]), 12)
        self.assertTrue(all(1 <= r["rank"] <= 18 for r in result["users"]))
        self.assertEqual(result["metrics"]["HR@20"], 1)

    def test_chunked_ranking_consistent(self):
        c = {**self.c, "model": "sasrec", "alignment_weight": 0}
        model = SUGAR(18, self.store.dim, c, self.store.embeddings)
        a = evaluate(model, self.data, self.store, c, chunk=3)
        b = evaluate(model, self.data, self.store, c, chunk=100)
        self.assertEqual([r["rank"] for r in a["users"]], [r["rank"] for r in b["users"]])


class EquationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = config()
        self.model = SUGAR(20, 32, self.c, np.ones((21, 32), dtype=np.float32))
        self.sequence = torch.tensor([[1, 2, 3], [2, 3, 0]])
        self.features = {"profile": torch.randn(2, 32), "present": torch.tensor([True, False]),
                         "n": torch.tensor([50.0, 0]), "sim": torch.tensor([0.8, 0]), "recent": torch.randn(2, 32),
                         "neighbors": torch.randn(2, 3, 32), "neighbor_mask": torch.tensor([[True, True, False], [False, False, False]])}

    def test_reliability_equation(self):
        _, gates = self.model(self.sequence, self.features)
        expected = torch.sigmoid(torch.tensor([0.8 + 0.5, 0]))
        torch.testing.assert_close(gates["rho"], expected)

    def test_no_neighbors_gate_zero(self):
        _, gates = self.model(self.sequence, self.features)
        self.assertEqual(gates["gamma"][1].item(), 0)

    def test_forced_closed_is_sasrec(self):
        self.model.c = {**self.c, "force_gate": 0}
        h, _ = self.model(self.sequence, self.features)
        torch.testing.assert_close(h, self.model.encode_sequence(self.sequence))

    def test_trainable_social_temperature(self):
        h, _ = self.model(self.sequence, self.features)
        h.square().sum().backward()
        self.assertIsNotNone(self.model.raw_beta.grad)
        self.assertGreater(abs(self.model.raw_beta.grad.item()), 1e-9)

    def test_default_profile_learns(self):
        h, _ = self.model(self.sequence, self.features)
        h.sum().backward()
        self.assertGreater(self.model.default_profile.grad.abs().sum().item(), 0)

    def test_causal_attention(self):
        block = SASBlock(16, 2, 0).eval()
        x = torch.randn(1, 5, 16)
        altered = x.clone()
        altered[:, 3:] += 100
        mask = torch.zeros(1, 5, dtype=torch.bool)
        torch.testing.assert_close(block(x, mask)[:, :3], block(altered, mask)[:, :3])

    def test_no_padding_nans(self):
        h, _ = self.model(self.sequence, self.features)
        self.assertTrue(torch.isfinite(h).all())

    def test_bce_loss_formula(self):
        self.model.c = {**self.c, "alignment_weight": 0}
        h = torch.randn(2, 16)
        positive = torch.tensor([1, 2])
        negatives = torch.tensor([[4, 5], [6, 7]])
        loss, _, _ = self.model.loss(h, positive, negatives, negatives)
        p = (h * self.model.items(positive)).sum(-1)
        n = self.model.scores(h, negatives)
        expected = -(torch.nn.functional.logsigmoid(p) + torch.nn.functional.logsigmoid(-n).sum(1)).mean()
        torch.testing.assert_close(loss, expected)

    def test_alignment_updates_shared_projection(self):
        h, _ = self.model(self.sequence, self.features)
        neg = torch.tensor([[5, 6], [7, 8]])
        # Distinct semantic vectors avoid a constant-loss alignment fixture.
        self.model.semantics.copy_(torch.randn_like(self.model.semantics))
        loss, _, _ = self.model.loss(h, torch.tensor([3, 4]), neg, neg)
        loss.backward()
        self.assertGreater(self.model.projection.weight.grad.abs().sum().item(), 0)

    def test_js_bounds_and_symmetry(self):
        self.assertAlmostEqual(js([1, 0], [0, 1]), np.log(2))
        self.assertAlmostEqual(js([1, 0], [1, 0]), 0)
        self.assertAlmostEqual(js([1, 3], [2, 1]), js([2, 1], [1, 3]))

    def test_metrics(self):
        self.assertEqual(rank_metrics(1)["NDCG@10"], 1)
        self.assertEqual(rank_metrics(11)["NDCG@10"], 0)
        self.assertAlmostEqual(rank_metrics(2)["NDCG@10"], 1 / np.log2(3))

    def test_invalid_semantics(self):
        for d in [{}, {"topic": ""}, {"topic": "test", "extra": 1}]:
            with self.assertRaises(ValueError):
                validate(d, ITEM_SCHEMA)

    def test_real_models_require_commits(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                SemanticBackend({**self.c, "semantic_backend": "real"}, t)

    def test_topic_clustering_counts_repeats(self):
        d = {"topic": "sport", "style": "tutorial", "summary": "sport"}
        groups = topic_clusters([d, d])
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(groups[0]["frequency"], 1)


if __name__ == "__main__":
    unittest.main()
