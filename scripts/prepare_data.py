import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from data.prepare import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--processed_root", default="outputs/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)   
    prepare(cfg, args.processed_root)


if __name__ == "__main__":
    main()