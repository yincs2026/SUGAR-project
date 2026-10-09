"""E4.1: profile corruption (poisoning) test.

With probability p per user, replace that user's profile embedding in a COPY
of the snapshot store with another user's profile embedding. The gated model
should detect the inconsistency (via sim) and close the gate; the fixed-gate
ablation (A3) should degrade much more. The SUGAR-vs-A3 gap on this curve is
the experimental proof of the paper's central claim.

Usage:
  python scripts/corrupt_profiles.py --dataset beauty --p 0.4 --eval_model sugar
  python scripts/corrupt_profiles.py --dataset beauty --p 0.4 --eval_model a3_fixed_gate
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

MODEL_MODES = {"sugar": "full", "a3_fixed_gate": "fixed_gate"}

# Training tags: run_ablation.py saves A-variants as {variant}_{model}_{dataset}
# (e.g. a3_a3_fixed_gate_beauty) while run_all.sh trains SUGAR as {model}_{dataset}
# (e.g. sugar_beauty). Map eval_model -> default training tag.
DEFAULT_TAGS = {
    "sugar": "sugar_{dataset}",
    "a3_fixed_gate": "a3_a3_fixed_gate_{dataset}",
}

def corrupt_store(store: SnapshotStore, p: float, seed: int) -> SnapshotStore:
    rng = np.random.default_rng(seed)
    users = list(store._snapshots.keys())
    out = SnapshotStore()
    out._snapshots = pickle.loads(pickle.dumps(store._snapshots))  # deep copy
    donors = [u for u in users if out.snapshots_of(u)]
    n_swapped = 0
    for u in users:
        if rng.random() >= p:
            continue
        snaps = out._snapshots.get(u, [])
        if not snaps:
            continue
        donor = rng.choice([d for d in donors if d != u]) if len(donors) > 1 else None
        if donor is None:
            continue
        dsnaps = out._snapshots[donor]
        for s in snaps:
            ds = min(dsnaps, key=lambda d: abs(d["t_chk"] - s["t_chk"]))
            if ds["profile_emb"] is not None:
                s["profile_emb"] = ds["profile_emb"]
        n_swapped += 1
    print(f"[E4.1] swapped profiles for {n_swapped}/{len(users)} users (p={p})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--eval_model", default="sugar",
                    choices=list(MODEL_MODES))
    ap.add_argument("--ckpt_tag", default=None,
                    help="training tag of the checkpoint to load; default: "
                         "derived from eval_model+dataset (e.g. sugar_beauty, "
                         "a3_a3_fixed_gate_beauty)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    name = args.dataset
    tag = args.ckpt_tag or DEFAULT_TAGS[args.eval_model].format(dataset=name)

    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"]) + 1
    emb = np.load(Path(cfg["data"]["processed_dir"]) / name / "item_embeddings.npy")
    store = SnapshotStore.load(str(Path(cfg["offline"].get(
        "snapshot_dir", "outputs/snapshots")) / f"{name}.pkl"))

    corrupted = corrupt_store(store, args.p, cfg["seed"])

    train_seqs, test_targets = {}, {}
    for u, s in sequences.items():
        if len(s) < 3:
            continue
        train_seqs[u] = s[:-2]
        test_targets[u] = s[-1]

    model = SUGARModel(n_items=n_items, d=cfg["model"]["d"],
                       d_e=cfg["model"]["de"], n_blocks=cfg["model"]["n_blocks"],
                       n_heads=cfg["model"]["n_heads"],
                       max_len=cfg["model"]["max_len"],
                       kappa=cfg["model"]["kappa"],
                       mode=MODEL_MODES[args.eval_model])
    ckpt = Path(cfg["training"]["checkpoint_dir"])/ f"{tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    window_means = build_window_means(train_seqs, emb,
                                      cfg["offline"]["window_size_k"])
    evaluator = FullRankingEvaluator(n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)
    metrics = evaluator.evaluate(model, train_seqs, cfg["model"]["max_len"],
                                 test_targets, snapshot_store=corrupted,
                                 window_means=window_means)

    out = {"experiment": "E4.1", "model": args.eval_model, "dataset": name,
           "p": args.p, "test": metrics, "seed": cfg["seed"], "tag": tag}
    f = Path("outputs/results") / \
        f"e41_{args.eval_model}_{name}_p{args.p}.json"
    with open(f, "w") as fh:
        json.dump(out, fh, indent=2)
    print("E4.1 TEST:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()