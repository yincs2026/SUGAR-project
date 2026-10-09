"""Train one model with one config.

Usage:
  python scripts/train.py --model sugar --dataset beauty
  python scripts/train.py --model a3_fixed_gate --dataset beauty
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.utils.seed import set_seed
from sugar.data.dataset import load_processed, ValidTestDataset
from sugar.data.social_graph import get_graph
from sugar.online.model import SUGARModel
from sugar.training.trainer import Trainer
from sugar.training.evaluator import FullRankingEvaluator
from sugar.offline.snapshot_store import SnapshotStore


def build_model(model_name: str, cfg: dict, n_items: int) -> SUGARModel:
    m = cfg["model"]
    mode = {"sugar": "full",
            "a1_no_profile": "no_profile",
            "a3_fixed_gate": "fixed_gate",
            "a4_no_social": "full"}.get(model_name, "full")
    return SUGARModel(
        n_items=n_items, d=m["d"], d_e=m["de"], n_blocks=m["n_blocks"],
        n_heads=m["n_heads"], max_len=m["max_len"], dropout=m["dropout"],
        kappa=m["kappa"], max_neighbors=m["n_social_neighbors_L"], mode=mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["sugar", "a1_no_profile", "a3_fixed_gate",
                             "a4_no_social", "a2_no_drift"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--strict_snapshots", type=int, default=1)
    ap.add_argument("--no_social", type=int, default=0,
                    help="A4: disable neighbor map injection")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.model}_{args.dataset}"

    sequences, items, social_edges = load_processed(
        args.dataset, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"]) + 1  # + padding

    # split: train prefix (drop last 2), valid target (-2), test target (-1)
    train_seqs, valid_targets, test_targets = {}, {}, {}
    for u, s in sequences.items():
        if len(s) < 3:
            continue
        train_seqs[u] = s[:-2]
        valid_targets[u] = s[-2]
        test_targets[u] = s[-1]

    snapshot_store, item_emb = None, None
    snap_path = Path(cfg["offline"].get("snapshot_dir",
                                        "outputs/snapshots")) / f"{args.dataset}.pkl"
    emb_path = Path(cfg["data"]["processed_dir"]) / args.dataset / "item_embeddings.npy"
    if snap_path.exists() and emb_path.exists():
        snapshot_store = SnapshotStore.load(str(snap_path))
        item_emb = np.load(emb_path)
        print(f"loaded snapshots: {snapshot_store.n_snapshots} "
              f"({snapshot_store.n_users} users)")
    elif args.model != "a1_no_profile":
        print("WARNING: no snapshot store found; "
              "run scripts/run_offline.py --phase all first "
              "(model will run in degraded cold-start mode)")

    graph = None
    if args.model != "a4_no_social" and args.no_social == 0 \
            and len(social_edges) > 0:
        graph = get_graph("real", n_users=len(sequences),
                          real_edges_path=None)
        # populate from processed edges
        for a, b in social_edges:
            graph.add_edge(a, b)

    model = build_model(args.model, cfg, n_items)
    trainer = Trainer(model, cfg, train_seqs, snapshot_store=snapshot_store,
                      social_graph=graph, item_embeddings=item_emb,
                      device=device, strict_snapshots=bool(args.strict_snapshots),
                      tag=tag)
    if item_emb is not None:
        trainer.attach_frozen_embeddings(item_emb)

    result = trainer.train(valid_targets)

    evaluator = FullRankingEvaluator(n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)
    test_metrics = evaluator.evaluate(
        model, train_seqs, cfg["model"]["max_len"], test_targets,
        snapshot_store=snapshot_store, neighbor_map=trainer.neighbor_map,
        window_means=trainer.window_means,
        strict=bool(args.strict_snapshots))

    out = {"tag": tag, "model": args.model, "dataset": args.dataset,
           "train": result, "test": test_metrics, "seed": cfg["seed"]}
    results_dir = Path("outputs/results"); results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("TEST:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()