#!/usr/bin/env python3
"""Independent trained variants with separate snapshots when their offline policy changes."""
import argparse
from pathlib import Path
from sugar.utils import load_config, load_json, save_json
from sugar.offline import build_snapshots
from sugar.train import train
from sugar.experiments import aggregate
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--cache", default="cache/semantic")
p.add_argument("--device", default="cuda")
a = p.parse_args()
c = load_config(a.config)
data = load_json(a.data)
variants = {"full": {}, "sasrec": {"model": "sasrec", "alignment_weight": 0},
            "lm-feature": {"model": "lm-feature", "alignment_weight": 0}, "no-drift": {"no_drift": True},
            "leaky-snapshots": {"leaky_snapshots": True}, "no-social": {"no_social": True}, "no-alignment": {"alignment_weight": 0}}
for name, patch in variants.items():
    cc = {**c, **patch}
    root = Path(a.out) / name
    build_snapshots(data, root / "artifacts", cc, a.cache)
    results = []
    for seed in cc["seeds"]:
        directory = root / f"seed-{seed}"
        train(a.data, root / "artifacts", directory, {**cc, "seed": seed}, a.device)
        results.append(str(directory / "test.json"))
    aggregate(results, root / "summary.json")
# Fixed fusion requires validation tuning, not silently using an arbitrary best value.
fixed_root = Path(a.out) / "fixed-fusion"
best = None
for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
    cc = {**c, "fixed_fusion": weight}
    build_snapshots(data, fixed_root / "artifacts", cc, a.cache)
    directory = fixed_root / f"tuning-{weight:g}"
    train(a.data, fixed_root / "artifacts", directory, cc, a.device, evaluate_test=False)
    metric = load_json(directory / "validation.json")["metrics"]["NDCG@10"]
    if best is None or metric > best["metric"]:
        best = {"weight": weight, "metric": metric}
save_json(fixed_root / "selected_weight.json", best)
cc = {**c, "fixed_fusion": best["weight"]}
results = []
for seed in cc["seeds"]:
    directory = fixed_root / f"seed-{seed}"
    train(a.data, fixed_root / "artifacts", directory, {**cc, "seed": seed}, a.device)
    results.append(str(directory / "test.json"))
aggregate(results, fixed_root / "summary.json")
