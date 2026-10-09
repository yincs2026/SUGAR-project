"""E9: qualitative case study -- profile before/after revision plus the
sim/rho time series for one user with visible drift.

Usage:
  python scripts/case_study.py --dataset beauty --ckpt_tag sugar_beauty \
      --user_idx -1   # -1 = pick the user with the most revisions
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.data.dataset import load_processed
from sugar.offline.snapshot_store import SnapshotStore
from sugar.online.model import SUGARModel
from sugar.training.trainer import build_window_means

JUDGE_PROMPT = """You are auditing automatically generated content
descriptors. Given the RAW item text and its generated descriptor, rate the
descriptor's faithfulness: 2 = fully consistent, 1 = partially consistent,
0 = hallucinated. Output ONLY {{"score": <0|1|2>}}.

Raw text: {text}

Descriptor: {descriptor}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt_tag", required=True)
    ap.add_argument("--user_idx", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    name = args.dataset
    proc = Path(cfg["data"]["processed_dir"]) / name

    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    emb = np.load(proc / "item_embeddings.npy")
    store = SnapshotStore.load(str(Path(cfg["offline"].get(
        "snapshot_dir", "outputs/snapshots")) / f"{name}.pkl"))

    # pick the user with the most committed revisions (proxy for drift)
    if args.user_idx >= 0:
        user = args.user_idx
    else:
        def n_rev(snaps):
            return sum(1 for s in snaps
                       if s["profile"] and s["profile"].get("_revised"))
        # fallback: pick the user with the largest sim swing
        def sim_swing(snaps):
            sims = [s["sim"] for s in snaps]
            return (max(sims) - min(sims)) if len(sims) > 2 else -1
        user = max(store._snapshots.keys(), key=lambda u: sim_swing(store.snapshots_of(u)))

    snaps = store.snapshots_of(user)
    window_means = build_window_means({user: sequences[user]}, emb,
                                      cfg["offline"]["window_size_k"])

    model = SUGARModel(n_items=len(items["item2idx"]) + 1,
                       d=cfg["model"]["d"], d_e=cfg["model"]["de"],
                       n_blocks=cfg["model"]["n_blocks"],
                       n_heads=cfg["model"]["n_heads"],
                       max_len=cfg["model"]["max_len"],
                       kappa=cfg["model"]["kappa"], mode="full")
    ckpt = Path(cfg["training"]["checkpoint_dir"]) / f"{args.ckpt_tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # rho over time from committed snapshots
    with torch.no_grad():
        rho_series = []
        for s in snaps:
            rho = model.fusion.reliability(
                torch.tensor(float(s["sim"])),
                torch.tensor(float(s["n_consolidated"]))).item()
            rho_series.append({"j": s["j"], "t_chk": s["t_chk"],
                               "sim": round(s["sim"], 4),
                               "rho": round(rho, 4)})

    # profile before/after the largest sim jump
    sims = [s["sim"] for s in snaps]
    k = int(np.argmin(sims))
    before = snaps[max(k - 1, 0)]["profile"]
    after = snaps[min(k + 1, len(snaps) - 1)]["profile"]

    out = {"experiment": "E9", "dataset": name, "user_idx": int(user),
           "seq_len": len(sequences[user]),
           "profile_before": before, "profile_after": after,
           "sim_rho_series": rho_series}
    with open(Path("outputs/results") / f"e9_{name}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[E9] user={user}: {len(snaps)} snapshots, "
          f"sim range [{min(sims):.3f}, {max(sims):.3f}]; "
          f"saved to outputs/results/e9_{name}.json")


if __name__ == "__main__":
    main()