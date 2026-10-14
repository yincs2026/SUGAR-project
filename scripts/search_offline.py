#!/usr/bin/env python3
"""Joint offline random search; rejects configurations above the paper's revision budget."""
import argparse
from pathlib import Path
import numpy as np
from sugar.utils import load_config, load_json, save_json
from sugar.offline import build_snapshots
from sugar.train import train
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--cache", default="cache/semantic")
p.add_argument("--device", default="cuda")
p.add_argument("--trials", type=int, default=20)
a = p.parse_args()
c = load_config(a.config)
data = load_json(a.data)
space = {"k": [10, 20, 30, 50], "checkpoint": [10, 20, 50], "edit_size": [5, 10, 20],
         "min_profile": [5, 10], "cooldown": [20, 50], "lag": [0, 5, 10, 20]}
rng = np.random.default_rng(c["seed"])
best, trials = None, []
for trial in range(a.trials):
    cc = {**c, **{key: values[int(rng.integers(len(values)))] for key, values in space.items()}}
    cc["threshold"] = "personalized" if rng.integers(2) else "global"
    if cc["threshold"] == "personalized":
        cc["xi"] = float(rng.choice([0.5, 1, 1.5, 2]))
    else:
        cc["tau"] = float(rng.choice(np.arange(0.55, 0.851, 0.05)))
    root = Path(a.out) / f"trial-{trial:03}"
    build_snapshots(data, root / "artifacts", cc, a.cache)
    logs = load_json(root / "artifacts/offline_log.json")
    frequency = 1000 * sum(x["trigger"] for x in logs) / len(logs) if logs else 0.0
    row = {"trial": trial, "config": cc, "revision_calls_per_1000": frequency,
           "within_budget": frequency <= 40}
    if row["within_budget"]:
        train(a.data, root / "artifacts", root / "run", cc, a.device, evaluate_test=False)
        row["validation_ndcg"] = load_json(root / "run/validation.json")["metrics"]["NDCG@10"]
        if best is None or row["validation_ndcg"] > best["validation_ndcg"]:
            best = row
    trials.append(row)
    save_json(Path(a.out) / "search.json", {"trials": trials, "best": best, "selection": "validation under budget"})
if best is None:
    raise SystemExit("No feasible configuration; no best model claimed")
