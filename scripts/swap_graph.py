"""E5: social graph substitution (Section 3.5 protocol).

Compares three graph conditions on the SAME trained SUGAR checkpoint:
  real           : the processed social edges
  random         : random graph with the same edge count (control)
  co_interaction : proxy graph from TRAIN interactions only (Section 3.5)
The E5 expectation: random ~ A4 (no social gain), co_interaction ~ real on
Beauty. Cold-start rows (users with < n0 interactions) are reported via
sequence-length segmentation in the analysis stage.

Usage:
  python scripts/swap_graph.py --dataset yelp_filtered --ckpt_tag sugar_yelp_filtered \
      --modes real random co_interaction
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
from sugar.data.dataset import load_processed
from sugar.data.social_graph import get_graph
from sugar.offline.snapshot_store import SnapshotStore
from sugar.online.model import SUGARModel
from sugar.training.evaluator import FullRankingEvaluator
from sugar.training.trainer import build_window_means, build_neighbor_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt_tag", required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["real", "random", "co_interaction"])
    ap.add_argument("--threshold", type=int, default=3,
                    help="co-interaction link threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    name = args.dataset

    sequences, items, social_edges = load_processed(
        name, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"]) + 1
    emb = np.load(Path(cfg["data"]["processed_dir"]) / name
                  / "item_embeddings.npy")
    store = SnapshotStore.load(str(Path(cfg["offline"].get(
        "snapshot_dir", "outputs/snapshots")) / f"{name}.pkl"))

    train_seqs, test_targets = {}, {}
    n0 = cfg["offline"]["min_profile_n0"]
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
                       kappa=cfg["model"]["kappa"],
                       max_neighbors=cfg["model"]["n_social_neighbors_L"],
                       mode="full")
    ckpt = Path(cfg["training"]["checkpoint_dir"]) / f"{args.ckpt_tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    window_means = build_window_means(train_seqs, emb,
                                      cfg["offline"]["window_size_k"])
    evaluator = FullRankingEvaluator(n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)

    real_path = str(Path(cfg["data"]["processed_dir"]) / name
                    / "social_edges.pkl")
    rows = []
    for mode in args.modes:
        graph = get_graph(mode, n_users=len(sequences),
                          real_edges_path=real_path,
                          train_sequences=train_seqs,
                          threshold=args.threshold)
        # populate the real graph from processed edges when loading by path
        if mode == "real" and graph.n_edges == 0 and social_edges:
            for a, b in social_edges:
                graph.add_edge(a, b)
        neighbor_map = build_neighbor_map(
            graph, store, window_means,
            cfg["model"]["de"], cfg["model"]["n_social_neighbors_L"])
        metrics = evaluator.evaluate(model, train_seqs,
                                     cfg["model"]["max_len"], test_targets,
                                     snapshot_store=store,
                                     neighbor_map=neighbor_map,
                                     window_means=window_means)
        rows.append({"mode": mode, "n_edges": graph.n_edges,
                     "n_users_with_neighbors": len(neighbor_map),
                     "test": metrics})
        print(f"[E5] mode={mode}: edges={graph.n_edges} "
              f"NDCG@10={metrics['NDCG@10']:.4f} HR@10={metrics['HR@10']:.4f}")

    with open(Path("outputs/results") / f"e5_{name}.json", "w") as f:
        json.dump({"experiment": "E5", "dataset": name, "rows": rows,
                   "seed": cfg["seed"]}, f, indent=2)


if __name__ == "__main__":
    main()