import tempfile
import unittest
from pathlib import Path
from sugar.experiments import import_baseline, aggregate, paired_test
from sugar.utils import save_json, load_json, digest


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = {"sequences": {"1": [[1, 1], [2, 2], [3, 3]]}, "items": {"a": 1, "b": 2, "c": 3}}
        data["fingerprint"] = digest(data)
        self.data = data
        save_json(self.root / "protocol/data.json", data)
        save_json(self.root / "protocol/protocol.json", {"mask_seen": True, "mock": True})
        self.result = {"model": "external-test-fixture", "source_repository": "https://example.com/test-only",
                       "source_commit": "a" * 40, "seed": 42, "data_fingerprint": data["fingerprint"],
                       "validation_config": {"dim": 16}, "users": [{"user": 1, "target": 3, "rank": 2}],
                       "full_catalog": True, "mask_seen": True}

    def tearDown(self):
        self.temp.cleanup()

    def test_import_preserves_mock_and_computes_metrics(self):
        save_json(self.root / "result.json", self.result)
        report = import_baseline(self.root / "protocol", self.root / "result.json", self.root / "run/test.json")
        self.assertTrue(report["mock"])
        self.assertEqual(report["metrics"]["HR@10"], 1)
        self.assertTrue((self.root / "run/config.json").exists())

    def test_wrong_target_rejected(self):
        self.result["users"][0]["target"] = 1
        save_json(self.root / "result.json", self.result)
        with self.assertRaises(ValueError):
            import_baseline(self.root / "protocol", self.root / "result.json", self.root / "run/test.json")

    def test_incomplete_commit_rejected(self):
        self.result["source_commit"] = "main"
        save_json(self.root / "result.json", self.result)
        with self.assertRaises(ValueError):
            import_baseline(self.root / "protocol", self.root / "result.json", self.root / "run/test.json")

    def test_duplicate_seeds_rejected(self):
        save_json(self.root / "result.json", self.result)
        import_baseline(self.root / "protocol", self.root / "result.json", self.root / "run/test.json")
        with self.assertRaises(ValueError):
            aggregate([self.root / "run/test.json"] * 2, self.root / "summary.json")
