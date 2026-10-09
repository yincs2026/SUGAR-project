"""E7: sensitivity to the window size k and the alignment weight lambda.

Each grid point trains a SHORT run (fewer epochs are acceptable for a
sensitivity shape) on the lightest dataset (Beauty), one seed.
Everything else comes from the base config; overrides use the dotted-CLI
mechanism of load_config.

Usage:
  python scripts/sensitivity.py --dataset beauty --param model.lambda_align \
      --values 0 0.01 0.1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--param", required=True,
                    help="dotted config key, e.g. offline.window_size_k "
                         "or model.lambda_align")
    ap.add_argument("--values", nargs="+", required=True, type=float)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    rows = []
    for v in args.values:
        tag = f"e7_{args.param.replace('.', '_')}_{v}_{args.dataset}"
        cmd = [sys.executable, "scripts/train.py", "--model", "sugar",
               "--dataset", args.dataset, "--tag", tag,
               "--config", args.config,
               f"{args.param}={v}",
               "training.epochs=60", "training.patience=6"]
        print("[E7] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        f = Path("outputs/results") / f"{tag}.json"
        with open(f) as fh:
            rows.append({"param": args.param, "value": v,
                         "test": json.load(fh)["test"]})

    with open(Path("outputs/results") /
              f"e7_{args.param.replace('.', '_')}_{args.dataset}.json",
              "w") as f:
        json.dump({"experiment": "E7", "rows": rows}, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()