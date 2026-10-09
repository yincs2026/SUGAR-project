"""E2: run the ablation grid on one dataset.

A1/A3/A4 reuse the A0 snapshot store (no LLM calls). A2 requires the
_nodrift snapshot store, built first if missing.

Usage:
  python scripts/run_ablation.py --dataset beauty --variants a1 a3 a4
  python scripts/run_ablation.py --dataset beauty --variants a2   # builds store if needed
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config

VARIANTS = {
    "a1": {"model": "a1_no_profile", "needs_nodrift_store": False},
    "a2": {"model": "a2_no_drift", "needs_nodrift_store": True},
    "a3": {"model": "a3_fixed_gate", "needs_nodrift_store": False},
    "a4": {"model": "a4_no_social", "needs_nodrift_store": False,
           "no_social": 1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants", nargs="+", default=["a1", "a2", "a3", "a4"])
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    snap_dir = Path(cfg["offline"].get("snapshot_dir", "outputs/snapshots"))

    for v in args.variants:
        spec = VARIANTS[v]
        if spec["needs_nodrift_store"]:
            store = snap_dir / f"{args.dataset}_nodrift.pkl"
            if not store.exists():
                print(f"[E2] building no-drift snapshot store for {v} ...")
                subprocess.run(
                    [sys.executable, "scripts/run_offline.py",
                     "--phase", "m2m3", "--no_drift"], check=True)
        cmd = [sys.executable, "scripts/train.py",
               "--model", spec["model"], "--dataset", args.dataset,
               "--tag", f"{v}_{spec['model']}_{args.dataset}"]
        if spec.get("no_social"):
            cmd += ["--no_social", "1"]
        print("[E2] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # summary
    results = {}
    for v in args.variants:
        f = Path("outputs/results") / \
            f"{v}_{VARIANTS[v]['model']}_{args.dataset}.json"
        if f.exists():
            with open(f) as fh:
                results[v] = json.load(fh)["test"]
    print(json.dumps(results, indent=2))
    with open(Path("outputs/results") /
              f"e2_summary_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()