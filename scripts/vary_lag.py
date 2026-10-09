"""E4.2: staleness sweep -- rebuild snapshot visibility under different
commit lags Lambda without re-running the LLM (commit time is metadata).

For each Lambda in the grid, shift every snapshot's t_commit to t_chk + Lambda,
re-evaluate the SAME trained checkpoint, and save one result row.

Usage:
  python scripts/vary_lag.py --dataset beauty --ckpt_tag sugar_beauty \
      --lags 0 1 5 20 50
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.utils.seed import set_seed
from sugar.data.dataset import load_processed
from sugar.offline.snapshot_store import SnapshotStore
from sugar.online.model import SUGARModel
from sugar.training.evaluator import FullRankingEvaluator
from sugar.training.trainer import build_window_means


def rebuild_store(store: SnapshotStore, lag: int) -> SnapshotStore:
    out = SnapshotStore()
    out._snapshots = pickle.loads(pickle.dumps(store._snapshots))
    for snaps in out._snapshots.values():
        for s in snaps:
            s["t_commit"] = s["t_chk"] + lag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt_tag", required=True)
    ap.add_argument("--lags", nargs="+", type=int,
                    default=[0, 1, 5, 20, 50])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    name = args.dataset

    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"]) + 1
    emb = np.load(Path(cfg["data"]["processed_dir"]) / name / "item_embeddings.npy")
    base_store = SnapshotStore.load(str(Path(cfg["offline"].get(
        "snapshot_dir", "outputs/snapshots")) / f"{name}.pkl"))

    train_seqs, test_targets = {}, {}
    for u, s in sequences.items():
        if len(s) < 3:
            continue
        train_seqs[u] = s[:-2]
        test_targets[u] = s[-1]

    model = SUGARModel(n_items=n_items, d=cfg["model"]["d"],
                       d_e=cfg["model"]["de"],
                       n_blocks=cfg["model"]["n_blocks"],
                       n_heads=cfg["model"]["n_heads"],
                       max_len=cfg["model"]["max_len"],
                       kappa=cfg["model"]["kappa"], mode="full")
    ckpt = Path(cfg["training"]["checkpoint_dir"]) / f"{args.ckpt_tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    window_means = build_window_means(train_seqs, emb,
                                      cfg["offline"]["window_size_k"])
    evaluator = FullRankingEvaluator(n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)

    rows = []
    for lag in args.lags:
        store = rebuild_store(base_store, lag)
        metrics = evaluator.evaluate(model, train_seqs,
                                     cfg["model"]["max_len"], test_targets,
                                     snapshot_store=store,
                                     window_means=window_means)
        rows.append({"lag": lag, "test": metrics})
        print(f"[E4.2] lag={lag}: NDCG@10={metrics['NDCG@10']:.4f} "
              f"HR@10={metrics['HR@10']:.4f}")

    with open(Path("outputs/results") / f"e42_{name}.json", "w") as f:
        json.dump({"experiment": "E4.2", "dataset": name, "rows": rows},
                  f, indent=2)


if __name__ == "__main__":
    main()