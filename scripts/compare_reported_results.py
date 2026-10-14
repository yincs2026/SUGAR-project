#!/usr/bin/env python3
"""Compare actual multi-seed summary against reported numbers without fabricating matches."""
import argparse
from pathlib import Path
from sugar.utils import load_json, save_json
p = argparse.ArgumentParser()
p.add_argument("--summary", required=True)
p.add_argument("--dataset", choices=["yelp", "douban-book", "epinions"], required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
result = load_json(a.summary)
if result.get("mock"):
    raise SystemExit("Synthetic/mock runs cannot be compared as a paper reproduction")
reference = load_json(Path(__file__).resolve().parents[1] / "docs/paper_reference.json")
comparison = {}
for name, reported in reference["metrics"][a.dataset].items():
    observed = result["metrics"][name]["mean"]
    comparison[name] = {"reported": reported, "observed": observed, "absolute_difference": observed - reported,
                        "relative_difference_percent": 100 * (observed / reported - 1)}
save_json(a.out, {"dataset": a.dataset, "reported_reference_is_not_computed": True, "comparison": comparison})
