"""Train + test one baseline: python scripts/train_baseline.py --model sasrec --dataset beauty"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.utils.seed import set_seed
from sugar.data.dataset import load_processed
from sugar.baselines.models import GRU4Rec, SASRecCE, BERT4Rec, UniSRecV
from sugar.baselines.run_baseline import train_baseline
from sugar.training.evaluator import FullRankingEvaluator


def build_baseline(name: str, cfg: dict, n_items: int):
    m = cfg["model"]
    if name == "gru4rec":
        model = GRU4Rec(n_items + 1, m["d"], num_layers=1, max_len=m["max_len"],
                        dropout=m["dropout"])
        family = "bce"
    elif name == "sasrec":
        model = SASRecCE(n_items + 1, m["d"], m["n_blocks"], m["n_heads"],
                         m["max_len"], m["dropout"])
        family = "ce"
    elif name == "bert4rec":
        model = BERT4Rec(n_items + 1, m["d"], m["n_blocks"], m["n_heads"],
                         m["max_len"], m["dropout"])
        family = "mlm"
    elif name == "unisrec":
        model = UniSRecV(n_items + 1, m["d"], m["n_blocks"], m["n_heads"],
                         m["max_len"], m["dropout"], d_e=m["de"])
        family = "ce"
    else:
        raise ValueError(name)
    cfg["model"]["family"] = family
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["gru4rec", "sasrec", "bert4rec", "unisrec"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.model}_{args.dataset}"

    sequences, items, _ = load_processed(args.dataset,
                                         cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"])  # catalog; +1 padding inside models

    train_seqs, valid_targets, test_targets = {}, {}, {}
    for u, s in sequences.items():
        if len(s) < 3:
            continue
        train_seqs[u] = s[:-2]
        valid_targets[u] = s[-2]
        test_targets[u] = s[-1]

    frozen = None
    if args.model == "unisrec":
        emb_path = (Path(cfg["data"]["processed_dir"]) / args.dataset
                    / "item_embeddings.npy")
        frozen = np.load(emb_path) if emb_path.exists() else None
        if frozen is None:
            print("WARNING: no frozen item embeddings; UniSRecV degrades "
                  "to plain SASRec (run offline --phase m1 first)")

    model = build_baseline(args.model, cfg, n_items)
    result = train_baseline(model, cfg, train_seqs, valid_targets, device, tag,
                            frozen_emb=frozen)

    evaluator = FullRankingEvaluator(model.n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)
    test_metrics = evaluator.evaluate(model, train_seqs,
                                      cfg["model"]["max_len"], test_targets)
    out = {"tag": tag, "model": args.model, "dataset": args.dataset,
           "train": result, "test": test_metrics, "seed": cfg["seed"]}
    with open(f"outputs/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("TEST:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()