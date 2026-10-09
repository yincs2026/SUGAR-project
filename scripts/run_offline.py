"""Offline stage entry point -- full Algorithm 1.

  phase m1   : M1 item refinement -> cached descriptors + embeddings
  phase m2m3 : M2 profiles + M3 drift detection -> snapshot store
  phase all  : both

Usage:
  python scripts/run_offline.py --phase m1
  python scripts/run_offline.py --phase m2m3
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
from sugar.data.dataset import load_processed, build_item_meta_texts
from sugar.offline.llm_client import LLMClient
from sugar.offline.encoders import SentenceEncoder
from sugar.offline.item_refiner import ItemRefiner
from sugar.offline.profile_builder import ProfileBuilder
from sugar.offline.drift_detector import DriftStateMachine  # noqa: F401 (import check)
from sugar.offline.pipeline import OfflinePipeline
from sugar.offline.snapshot_store import SnapshotStore


def _load_cached_descriptors(cfg: dict, n_items: int) -> list:
    """Read descriptors from the M1 cache (no LLM calls)."""
    name = cfg["data"]["dataset"]
    cache = Path(cfg["llm"]["cache_dir"]) / "items" / "descriptors"
    import re
    by_idx = {}
    for f in cache.glob("*.json"):
        idx = int(f.name.split("_")[0])
        if idx not in by_idx:  # first hash wins; content is identical per idx
            with open(f, "r", encoding="utf-8") as fh:
                by_idx[idx] = json.load(fh)
    assert len(by_idx) >= n_items, \
        f"M1 cache incomplete: {len(by_idx)}/{n_items}. Run --phase m1 first."
    return [by_idx[i] for i in range(n_items)]


def run_m1(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    name = cfg["data"]["dataset"]
    _, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    texts = build_item_meta_texts(items)
    idx2meta = {v: k for k, v in items["item2idx"].items()}
    cats = [items["meta"][idx2meta[i]].get("categories", [])
            for i in range(len(texts))]

    llm = LLMClient(model_name=cfg["llm"]["model_name"],
                    cache_dir=f"{cfg['llm']['cache_dir']}/llm",
                    max_new_tokens=cfg["llm"]["max_new_tokens"])
    encoder = SentenceEncoder(model_name=cfg["encoder"]["model_name"],
                              cache_dir=f"{cfg['encoder']['cache_dir']}/enc")
    refiner = ItemRefiner(llm, encoder,
                          cache_dir=f"{cfg['llm']['cache_dir']}/items")
    descriptors, embeddings = refiner.refine_catalog(texts, cats)

    out_dir = Path(cfg["data"]["processed_dir"]) / name
    np.save(out_dir / "item_embeddings.npy", embeddings)
    stats = {"m1": refiner.stats, "llm": llm.stats_snapshot(),
             "n_items": len(texts), "emb_dim": int(embeddings.shape[1])}
    with open(out_dir / "m1_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[M1] done: {stats['n_items']} items, "
          f"schema_violation_rate={refiner.stats['schema_violation_rate']}, "
          f"llm_calls={llm.stats['calls']}")
    return stats


def run_m2m3(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    name = cfg["data"]["dataset"]
    sequences, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    n_items = len(items["item2idx"])

    embeddings = np.load(Path(cfg["data"]["processed_dir"]) / name
                         / "item_embeddings.npy")
    assert embeddings.shape[0] >= n_items, "run --phase m1 first"
    descriptors = _load_cached_descriptors(cfg, n_items)

    llm = LLMClient(model_name=cfg["llm"]["model_name"],
                    cache_dir=f"{cfg['llm']['cache_dir']}/llm",
                    max_new_tokens=cfg["llm"]["max_new_tokens"])
    encoder = SentenceEncoder(model_name=cfg["encoder"]["model_name"],
                              cache_dir=f"{cfg['encoder']['cache_dir']}/enc")
    builder = ProfileBuilder(llm, encoder,
                             min_profile_n0=cfg["offline"]["min_profile_n0"],
                             edit_batch_medit=cfg["offline"]["edit_batch_medit"],
                             window_size_k=cfg["offline"]["window_size_k"])
    pipeline = OfflinePipeline(
        builder, embeddings,
        m_chk=cfg["offline"]["checkpoint_interval_mchk"],
        commit_lag_lambda=cfg["offline"]["commit_lag_lambda"],
        threshold_tau=cfg["offline"]["threshold_tau"],
        cooldown_nmin=cfg["offline"]["cooldown_nmin"],
        enable_drift=cfg["offline"].get("enable_drift", True))

    suffix = "_nodrift" if not cfg["offline"].get("enable_drift", True) else ""
    out_path = (Path(cfg["offline"].get("snapshot_dir",
                                        "outputs/snapshots"))
                / f"{name}{suffix}.pkl")

    out_path = Path(cfg["offline"].get("snapshot_dir",
                                       "outputs/snapshots")) / f"{name}.pkl"
    store = pipeline.run(sequences, descriptors, str(out_path))

    stats = {"m2m3": pipeline.stats, "m2": builder.stats,
             "llm": llm.stats_snapshot(),
             "n_users": store.n_users, "n_snapshots": store.n_snapshots,
             "snapshot_path": str(out_path)}
    with open(Path(cfg["data"]["processed_dir"]) / name / "m2m3_stats.json",
              "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[M2/M3] done: users={store.n_users}, "
          f"snapshots={store.n_snapshots}, "
          f"revisions={pipeline.stats['revisions_fired']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--phase", default="all", choices=["m1", "m2m3", "all"])
    ap.add_argument("--no_drift", action="store_true",
                    help="A2 ablation: disable M3 drift detection/revisions")
    ap.add_argument("--dataset", default=None,
                    help="override data.dataset from the config "
                         "(e.g. beauty, yelp_filtered, beauty_drift)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.no_drift:
        cfg["offline"]["enable_drift"] = False
    if args.phase in ("m1", "all"):
        run_m1(cfg)
    if args.phase in ("m2m3", "all"):
        run_m2m3(cfg)


if __name__ == "__main__":
    main()