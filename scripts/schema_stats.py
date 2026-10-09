"""E6: schema compliance and fallback statistics from the M1 cache.

No new LLM calls: reads the descriptor cache and reports
  - schema compliance rate (all 5 required fields, valid types)
  - fallback rate (descriptors written by the surface-feature fallback)
  - per-field completeness
Also runs the LLM-as-judge hallucination check on a sample when --judge N
is given: asks the frozen LLM to rate descriptor faithfulness on a 0-2
scale against the raw item text (200 items / 100 profiles per Section 4).

Usage:
  python scripts/schema_stats.py --dataset beauty [--judge 200]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.data.dataset import load_processed, build_item_meta_texts
from sugar.offline.item_refiner import validate_descriptor, REQUIRED_FIELDS

JUDGE_PROMPT = """You are auditing automatically generated content
descriptors. Given the RAW item text and its generated descriptor, rate the
descriptor's faithfulness: 2 = fully consistent, 1 = partially consistent
(minor unsupported detail), 0 = hallucinated (claims not supported by the
raw text). Output ONLY the JSON object {{"score": <0|1|2>}}.

Raw text: {text}

Descriptor: {descriptor}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--judge", type=int, default=0,
                    help="N items to audit with LLM-as-judge (0 = skip)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = args.dataset
    cache = Path(cfg["llm"]["cache_dir"]) / "items" / "descriptors"

    by_idx = {}
    for f in cache.glob("*.json"):
        idx = int(f.name.split("_")[0])
        if idx not in by_idx:
            with open(f, "r", encoding="utf-8") as fh:
                by_idx[idx] = json.load(fh)

    n = len(by_idx)
    valid = sum(1 for d in by_idx.values() if validate_descriptor(d))
    fallback = sum(1 for d in by_idx.values() if d.get("_fallback"))
    field_complete = {f: sum(1 for d in by_idx.values()
                             if d.get(f)) / n for f in REQUIRED_FIELDS}

    stats = {
        "n_items_cached": n,
        "schema_compliance_rate": round(valid / n, 4),
        "fallback_rate": round(fallback / n, 4),
        "field_completeness": {k: round(v, 4)
                               for k, v in field_complete.items()},
    }
    print(json.dumps(stats, indent=2))

    if args.judge > 0:
        _, items, _ = load_processed(name, cfg["data"]["processed_dir"])
        texts = build_item_meta_texts(items)
        from sugar.offline.llm_client import LLMClient
        llm = LLMClient(model_name=cfg["llm"]["model_name"],
                        cache_dir=f"{cfg['llm']['cache_dir']}/llm")
        import numpy as np
        rng = np.random.default_rng(cfg["seed"])
        sample = rng.choice(sorted(by_idx), size=min(args.judge, n),
                            replace=False)
        scores = []
        for idx in sample:
            resp = llm.generate_json(JUDGE_PROMPT.format(
                text=texts[idx][:500],
                descriptor=json.dumps(by_idx[idx], ensure_ascii=False)))
            if resp and "score" in resp:
                scores.append(int(resp["score"]))
        if scores:
            stats["judge_n"] = len(scores)
            stats["judge_mean_score"] = round(float(np.mean(scores)), 3)
            stats["judge_hallucination_rate"] = round(
                sum(1 for s in scores if s == 0) / len(scores), 4)
            print(json.dumps({k: stats[k] for k in
                              ("judge_n", "judge_mean_score",
                               "judge_hallucination_rate")}, indent=2))

    out = Path("outputs/results") / f"e6_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()