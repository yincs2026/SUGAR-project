import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from test_core import config
from sugar.data import demo, prepare
from sugar.offline import build_snapshots
from sugar.train import train, evaluate
from sugar.utils import load_json


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = {**config(), "epochs": 2, "batch_size": 32}
        demo(self.root / "raw", events=40)
        data = prepare(self.root / "raw", self.root / "prepared", self.c)
        build_snapshots(data, self.root / "artifacts", self.c, self.root / "cache")
        self.data_path = self.root / "prepared/data.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_resume_matches_uninterrupted_training(self):
        full = self.root / "full"
        interrupted = self.root / "interrupted"
        train(self.data_path, self.root / "artifacts", full, self.c)
        calls = 0
        def failing_evaluate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Simulated process interruption")
            return evaluate(*args, **kwargs)
        with patch("sugar.train.evaluate", failing_evaluate):
            with self.assertRaises(RuntimeError):
                train(self.data_path, self.root / "artifacts", interrupted, self.c)
        train(self.data_path, self.root / "artifacts", interrupted, self.c, resume=interrupted / "last.pt")
        a = torch.load(full / "last.pt", weights_only=False)
        b = torch.load(interrupted / "last.pt", weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual(load_json(full / "test.json")["metrics"], load_json(interrupted / "test.json")["metrics"])

    def test_mock_guard(self):
        with self.assertRaises(ValueError):
            train(self.data_path, self.root / "artifacts", self.root / "guard", {**self.c, "allow_nonpaper": False})

    def test_bpr_and_baselines_train(self):
        for model, loss in [("sugar", "bpr"), ("sasrec", "bce"), ("lm-feature", "bce")]:
            c = {**self.c, "epochs": 1, "model": model, "loss": loss}
            result = train(self.data_path, self.root / "artifacts", self.root / f"{model}-{loss}", c)
            self.assertTrue(result["mock"])
            self.assertTrue((self.root / f"{model}-{loss}/best.pt").exists())


if __name__ == "__main__":
    unittest.main()
