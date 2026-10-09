"""E8: offline/online cost accounting.

Offline: reads the LLM client stats saved by the offline runs and classifies
calls per operation type (refine/build/consolidate/revise) from the M1/M2M3
stats files; token totals from llm cache metadata.
Online: measures per-request latency of SUGAR vs plain SASRec on the same
device (forward + full-catalog scoring), amortized over 200 warm requests.

Usage:
  python scripts/cost_profile.py --dataset beauty
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.data.dataset import load_processed
from sugar.offline.snapshot_store import SnapshotStore
from sugar.online.model import SUGARModel
from sugar.baselines.models import SASRecCE


def measure_latency(model, seq_t, users_t, lens_t, store, window_means,
                    device, repeats: int = 200) -> dict:
    model.eval()
    all_items = torch.arange(1, model.backbone.item_emb.num_embeddings,
                             device=device)
    with torch.no_grad():
        # warmup
        for _ in range(10):
            h = model.forward_with_users(seq_t, users_t, lens_t, store,
                                         window_means=window_means)
            _ = h[:, -1, :] @ model.backbone.embed_items(all_items).T
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeats):
            h = model.forward_with_users(seq_t, users_t, lens_t, store,
                                         window_means=window_means)
            _ = h[:, -1, :] @ model.backbone.embed_items(all_items).T
        if device == "cuda":
            torch.cuda.synchronize()
    per_req_ms = (time.time() - t0) / repeats * 1000
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"latency_ms_per_request": round(per_req_ms, 3),
            "trainable_params": n_par}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    name = args.dataset
    proc = Path(cfg["data"]["processed_dir"]) / name

    # ---- offline costs (from stats files, no recomputation) --------------
    offline = {}
    for f, key in (("m1_stats.json", "m1"), ("m2m3_stats.json", "m2m3")):
        p = proc / f
        if p.exists():
            with open(p) as fh:
                offline[key] = json.load(fh)
    offline["llm_cache_files"] = len(list(
        (Path(cfg["llm"]["cache_dir"]) / "llm").glob("*.json")))

    # ---- online latency: SUGAR vs SASRec ---------------------------------
    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"]) + 1
    emb = np.load(proc / "item_embeddings.npy")
    store = SnapshotStore.load(str(Path(cfg["offline"].get(
        "snapshot_dir", "outputs/snapshots")) / f"{name}.pkl"))

    from sugar.training.trainer import build_window_means
    window_means = build_window_means(
        {u: s[:-2] for u, s in sequences.items() if len(s) >= 3}, emb,
        cfg["offline"]["window_size_k"])

    B, L = 1, cfg["model"]["max_len"]
    rng = np.random.default_rng(cfg["seed"])
    u = next(iter(window_means))
    seq = torch.tensor([sequences[u][-L:]], device=device)
    lens = torch.tensor([min(len(sequences[u]), L)], device=device)
    users_t = torch.tensor([u], device=device)

    sugar = SUGARModel(n_items=n_items, d=cfg["model"]["d"],
                       d_e=cfg["model"]["de"],
                       n_blocks=cfg["model"]["n_blocks"],
                       n_heads=cfg["model"]["n_heads"],
                       max_len=cfg["model"]["max_len"],
                       kappa=cfg["model"]["kappa"], mode="full").to(device)
    sasrec = SASRecCE(n_items, cfg["model"]["d"], cfg["model"]["n_blocks"],
                      cfg["model"]["n_heads"], cfg["model"]["max_len"],
                      cfg["model"]["dropout"]).to(device)

    out = {"offline": offline,
           "online": {
               "sugar": measure_latency(sugar, seq, users_t, lens, store,
                                        window_means, device),
               "sasrec": measure_latency(sasrec, seq, users_t, lens, None,
                                         None, device),
           }}
    ratio = out["online"]["sugar"]["latency_ms_per_request"] / \
        max(out["online"]["sasrec"]["latency_ms_per_request"], 1e-9)
    out["online"]["sugar_over_sasrec_latency_ratio"] = round(ratio, 3)

    with open(Path("outputs/results") / f"e8_{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()