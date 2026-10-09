"""E3 part 1: synthetic drift injection (Section 4 / E3 protocol).

For a fraction of users, from a random cut point in their sequence, replace
`drift_ratio` of subsequent items with items from an OFF-TOPIC cluster
(items whose frozen descriptor embedding is farthest from the user's own
window mean). Saves the ground-truth drift labels for detector evaluation.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.utils.seed import set_seed
from sugar.data.dataset import load_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--drift_ratio", type=float, default=0.8,
                    help="share of post-cut interactions replaced")
    ap.add_argument("--out_suffix", default="_drift")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    name = args.dataset

    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    emb = np.load(Path(cfg["data"]["processed_dir"]) / name
                  / "item_embeddings.npy")

    rng = np.random.default_rng(cfg["seed"])
    users = sorted(sequences.keys())
    n_drift = int(len(users) * args.fraction)
    drift_users = set(rng.choice(users, size=n_drift, replace=False).tolist())

    # global item popularity used only to pick the off-topic pool
    new_sequences, labels = {}, {}
    for u, seq in sequences.items():
        if u not in drift_users or len(seq) < 10:
            new_sequences[u] = seq
            labels[u] = 0
            continue
        cut = rng.integers(len(seq) // 3, len(seq) - 3)
        prefix = seq[:cut]
        post = seq[cut:]
        wm = emb[prefix[-cfg["offline"]["window_size_k"]:]].mean(axis=0)
        wm_norm = wm / (np.linalg.norm(wm) + 1e-9)
        # off-topic pool: 5% of items most dissimilar to the user's window
        sims = (emb @ wm_norm)
        pool = np.argsort(sims)[: max(50, int(0.05 * emb.shape[0]))]
        n_replace = int(len(post) * args.drift_ratio)
        replace_pos = rng.choice(len(post), size=n_replace, replace=False)
        post = post.copy()
        post[replace_pos] = rng.choice(pool, size=n_replace)
        new_sequences[u] = prefix + post.tolist()
        labels[u] = 1

    out_dir = Path(cfg["data"]["processed_dir"]) / f"{name}{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy catalog artifacts unchanged (same item space)
    for f in ("items.pkl", "item_embeddings.npy"):
        src = Path(cfg["data"]["processed_dir"]) / name / f
        if src.exists():
            (out_dir / f).write_bytes(src.read_bytes())
    with open(out_dir / "sequences.pkl", "wb") as f:
        pickle.dump(new_sequences, f)
    with open(out_dir / "social_edges.pkl", "wb") as f:
        pickle.dump([], f)
    with open(out_dir / "drift_labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    print(f"[E3] injected drift for {len(drift_users)}/{len(users)} users "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()