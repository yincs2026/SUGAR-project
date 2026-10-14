from collections import Counter
from pathlib import Path
import csv
import subprocess
import time
import numpy as np
from .utils import load_json, save_json, digest
from .train import load_run, evaluate, Collator, Queries, move_batch, rank_metrics


def aggregate(paths, out):
    results = [load_json(p) for p in paths]
    if any(r.get("mock", False) for r in results) and not all(r.get("mock", False) for r in results):
        raise ValueError("Cannot mix mock and real results")
    keys = ["HR@5", "HR@10", "HR@20", "NDCG@10"]
    seeds = []
    for path in paths:
        seeds.append(load_json(Path(path).parent / "config.json")["seed"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds do not constitute independent runs")
    configs = [load_json(Path(p).parent / "config.json") for p in paths]
    configs = [{k: v for k, v in c.items() if k not in {"seed", "llara_backbone_checkpoint"}} for c in configs]
    if len({digest(c) for c in configs}) != 1:
        raise ValueError("Aggregation requires same configuration except random seed")
    provenances = [load_json(Path(p).parent / "provenance.json") for p in paths]
    if len({digest({"data": p["data"], "artifacts": p["artifacts"]}) for p in provenances}) != 1:
        raise ValueError("Aggregation requires the same dataset and semantic artifacts")
    report = {"runs": len(results), "seeds": seeds, "mock": results[0].get("mock", False),
              "metrics": {k: {"mean": float(np.mean([r["metrics"][k] for r in results])),
                               "std": float(np.std([r["metrics"][k] for r in results], ddof=1)) if len(results) > 1 else None} for k in keys}}
    report['buckets'] = {}
    for bucket in sorted(set().union(*(r.get('buckets', {}).keys() for r in results))):
        group = [r['buckets'][bucket] for r in results if bucket in r.get('buckets', {})]
        count = group[0]['users']
        total = len(results[0].get('users', []))
        report['buckets'][bucket] = {'users': count, 'percent_of_test_users': 100 * count / total if total else None,
                                    'metrics': {k: {'mean': float(np.mean([g[k] for g in group])),
                                                     'std': float(np.std([g[k] for g in group], ddof=1)) if len(group) > 1 else None}
                                                for k in ['HR@10', 'NDCG@10', 'gamma', 'gate', 'rho'] if k in group[0]}}
    save_json(out, report)
    return report


def paired_test(paths_a, paths_b, out):
    from scipy.stats import ttest_rel
    if len(paths_a) != len(paths_b) or len(paths_a) < 2:
        raise ValueError("Need equal sets of at least two paired seeded runs")
    seeds_a = [load_json(Path(p).parent / "config.json")["seed"] for p in paths_a]
    seeds_b = [load_json(Path(p).parent / "config.json")["seed"] for p in paths_b]
    if seeds_a != seeds_b or len(set(seeds_a)) != len(seeds_a):
        raise ValueError("Paired runs must use distinct, matching seed IDs in the same order")
    a, b = [load_json(p) for p in paths_a], [load_json(p) for p in paths_b]
    provenance = [load_json(Path(p).parent / "provenance.json") for p in paths_a + paths_b]
    if len({p["data"] for p in provenance}) != 1:
        raise ValueError("Paired comparison datasets differ")
    report = {"unit": "seed", "seeds": seeds_a, "alternative": "two-sided", "metrics": {}}
    for k in ["HR@10", "NDCG@10"]:
        av, bv = np.array([r["metrics"][k] for r in a]), np.array([r["metrics"][k] for r in b])
        statistic, pvalue = ttest_rel(av, bv)
        report["metrics"][k] = {"mean_difference": float(np.mean(av - bv)),
                               "t": float(statistic) if np.isfinite(statistic) else None,
                               "p": float(pvalue) if np.isfinite(pvalue) else None}
    save_json(out, report)
    return report


def stress_test(data_path, artifacts, run, out, device):
    results = {}
    for condition in ["clean", "corrupted", "stale"]:
        model, data, store, c = load_run(data_path, artifacts, run, device, condition)
        results[condition] = evaluate(model, data, store, c, device=device)
    model, data, store, c = load_run(data_path, artifacts, run, device, force_gate=1.0)
    opened = evaluate(model, data, store, c, device=device)
    from scipy.stats import spearmanr
    rho = [r["rho"] for r in opened["users"]]
    error = [1 - r["NDCG@10"] for r in opened["users"]]
    corr = spearmanr(rho, error).statistic if len(set(rho)) > 1 and len(set(error)) > 1 else float("nan")
    results["forced_open_reliability_error_spearman"] = float(corr) if np.isfinite(corr) else None
    results["error_definition"] = "1 minus per-user NDCG@10"
    save_json(out, results)
    return results


def js(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    def kl(x):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / m[mask])))
    return (kl(a) + kl(b)) / 2


def drift_analysis(dataset_files, artifact_dirs, out, epsilon=None, percentile=90):
    if len(dataset_files) != len(artifact_dirs):
        raise ValueError("One artifact directory per dataset required")
    transitions, logs_by_data = [], []
    for index, (file, directory) in enumerate(zip(dataset_files, artifact_dirs)):
        data = load_json(file)
        manifest = load_json(Path(directory) / "manifest.json")
        c = manifest["offline_config"]
        categories = sorted({str(v["category"]) for v in data["catalog"].values()})
        if categories == ["unknown"]:
            raise ValueError("Drift ground truth requires independent item metadata")
        def distribution(events):
            counts = Counter(str(data["catalog"][str(item)]["category"]) for item, _ in events)
            return [counts[cat] for cat in categories]
        for u, seq in data["sequences"].items():
            seq = seq[:-2]
            previous = None
            for pos in range(c["checkpoint"], len(seq) + 1, c["checkpoint"]):
                current = distribution(seq[max(0, pos - c["k"]):pos])
                if previous is not None:
                    transitions.append({"dataset": index, "user": int(u), "checkpoint": pos // c["checkpoint"],
                                        "js": js(previous, current)})
                previous = current
        logs_by_data.append(load_json(Path(directory) / "offline_log.json"))
    if not transitions:
        raise ValueError("Not enough consecutive checkpoints to evaluate drift")
    supplied_epsilon = epsilon is not None
    epsilon = float(np.percentile([t["js"] for t in transitions], percentile)) if epsilon is None else float(epsilon)
    report = {"percentile": percentile if not supplied_epsilon else None, "epsilon": epsilon, "log_base": "natural", "pooled_datasets": len(dataset_files), "datasets": []}
    for index, logs in enumerate(logs_by_data):
        truth = [(t["user"], t["checkpoint"]) for t in transitions if t["dataset"] == index and t["js"] > epsilon]
        triggers = [(t["user"], t["checkpoint"]) for t in logs if t["trigger"]]
        matched_triggers = [t for t in triggers if any(t[0] == d[0] and 0 <= t[1] - d[1] <= 3 for d in truth)]
        delays = [min(t[1] - d[1] for t in triggers if t[0] == d[0] and 0 <= t[1] - d[1] <= 3)
                  for d in truth if any(t[0] == d[0] and 0 <= t[1] - d[1] <= 3 for t in triggers)]
        usage = load_json(Path(artifact_dirs[index]) / "llm_usage.json")
        report["datasets"].append({"events": len(truth), "triggers": len(triggers),
            "events_per_100_checkpoints": 100 * len(truth) / len(logs) if logs else None,
            "precision": len(matched_triggers) / len(triggers) if triggers else None,
            "recall": len(delays) / len(truth) if truth else None,
            "delay_checkpoints": float(np.mean(delays)) if delays else None,
            "revision_calls_per_1000_checkpoints": 1000 * len(triggers) / len(logs) if logs else None,
            "revision_budget_satisfied": (1000 * len(triggers) / len(logs) <= 40) if logs else None,
            "total_prompt_tokens": sum(x.get("prompt_tokens", 0) for x in usage if not x["cache_hit"]),
            "total_completion_tokens": sum(x.get("completion_tokens", 0) for x in usage if not x["cache_hit"]),
            "logical_prompt_tokens_including_cache": sum(x.get("prompt_tokens", 0) for x in usage),
            "logical_completion_tokens_including_cache": sum(x.get("completion_tokens", 0) for x in usage),
            "uncached_seconds": sum(x["seconds"] for x in usage if not x["cache_hit"]),
            "fallbacks": sum(x["fallback"] for x in usage), "calls": len(usage)})
    save_json(out, report)
    return report


def benchmark(data_path, artifacts, run, out, device, requests=1000):
    import torch
    model, data, store, c = load_run(data_path, artifacts, run, device)
    model.eval()
    queries = Queries(data, "test").entries
    batch = move_batch(Collator(data, store, c)(queries[:1]), device)
    sequence, features, _, _ = batch
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(100):
            model(sequence, features)
        sync()
        for _ in range(requests):
            sync()
            start = time.perf_counter()
            model(sequence, features)
            sync()
            times.append(time.perf_counter() - start)
    report = {"scope": "batch-1 model representation, excludes CPU feature join and catalog ranking",
              "requests": requests, "device": device, "p50_ms": float(np.percentile(times, 50) * 1000),
              "p95_ms": float(np.percentile(times, 95) * 1000), "requests_per_second": 1 / float(np.mean(times))}
    save_json(out, report)
    return report


def export_protocol(data_path, artifacts, config, out):
    """Inputs for independent official baseline runners, not relabelled SUGAR approximations."""
    from .features import FeatureStore
    data = load_json(data_path)
    store = FeatureStore(data, artifacts, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "protocol.json", {"data_fingerprint": data["fingerprint"], "full_catalog": True,
                                      "mask_seen": config["mask_seen"], "id_padding": 0,
                                      "validation_selection_only": True, "mock": store.manifest["mock"]})
    save_json(out / "data.json", data)
    np.save(out / "semantic_items.npy", store.embeddings)
    for split in ["train", "validation", "test"]:
        with (out / f"{split}.jsonl").open("w") as f:
            import json
            for u, pos in Queries(data, split).entries:
                fs = store.query(u, pos)
                f.write(json.dumps({"user": int(u), "position": pos,
                    "sequence": [r[0] for r in data["sequences"][u][max(0, pos - config["max_len"]):pos]],
                    "target": data["sequences"][u][pos][0],
                    "target_time": data["sequences"][u][pos][1],
                    "profile": fs["profile"].tolist(), "neighbors": fs["neighbors"].tolist()}) + "\n")


def import_baseline(protocol, results_file, out):
    protocol = Path(protocol)
    data, metadata = load_json(protocol / "data.json"), load_json(protocol / "protocol.json")
    result = load_json(results_file)
    for key in ["model", "source_repository", "source_commit", "seed", "data_fingerprint", "validation_config", "users", "full_catalog", "mask_seen"]:
        if key not in result:
            raise ValueError(f"Baseline result missing {key}")
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", str(result["source_commit"])):
        raise ValueError("External source_commit must be an immutable Git commit SHA")
    if result["data_fingerprint"] != data["fingerprint"] or result["full_catalog"] is not True or result["mask_seen"] != metadata["mask_seen"]:
        raise ValueError("External baseline protocol mismatch")
    expected = {int(u) for u in data["sequences"]}
    observed = [r["user"] for r in result["users"]]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("External results require exactly one record for every test user")
    rows = []
    for row in result["users"]:
        seq = data["sequences"][str(row["user"])]
        rank = row["rank"]
        if row["target"] != seq[-1][0] or type(rank) is not int or not 1 <= rank <= len(data["items"]):
            raise ValueError("Invalid target/rank from external runner")
        rows.append({**row, **rank_metrics(rank)})
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in ["HR@5", "HR@10", "HR@20", "NDCG@10"]}
    result.update({"users": rows, "metrics": metrics, "mock": metadata["mock"]})
    output_dir = Path(out).parent
    save_json(output_dir / "config.json", {**result["validation_config"], "model": result["model"], "seed": result["seed"]})
    save_json(output_dir / "provenance.json", {"data": data["fingerprint"], "artifacts": "external-baseline",
                                               "source_repository": result["source_repository"], "source_commit": result["source_commit"]})
    save_json(out, result)
    return result


def benchmark_pipeline(data_path, artifacts, run, out, device='cpu', requests=100, topk=10, chunk=4096):
    """Wall-clock request: CPU temporal feature join, transfers, model, full-catalog top-K."""
    import torch
    from .train import load_run, Collator, Queries, move_batch
    model, data, store, c = load_run(data_path, artifacts, run, device)
    model.eval()
    queries = Queries(data, 'test').entries
    collate = Collator(data, store, c)
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    def request(query):
        batch = move_batch(collate([query]), device)
        h, _ = model(batch[0], batch[1])
        user, pos = query
        seen = {v for v, _ in data['sequences'][user][:pos]} if c['mask_seen'] else set()
        target_time = data['sequences'][user][pos][1]
        values, candidates = [], []
        for start in range(1, len(data['items']) + 1, chunk):
            ids = torch.arange(start, min(start + chunk, len(data['items']) + 1), device=device)
            scores = model.score_catalog(h, ids).flatten()
            excluded = [int(v) - start for v in ids.tolist() if int(v) in seen or data['catalog'][str(int(v))]['available_at'] >= target_time]
            scores[excluded] = -torch.inf
            k = min(topk, len(ids))
            v, indices = torch.topk(scores, k)
            values.append(v)
            candidates.append(ids[indices])
        merged = torch.cat(values)
        choices = torch.cat(candidates)
        indices = torch.topk(merged, min(topk, len(merged))).indices
        return choices[indices][torch.isfinite(merged[indices])].cpu().tolist()
    times = []
    with torch.inference_mode():
        for i in range(min(10, len(queries))):
            request(queries[i])
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for i in range(requests):
            sync()
            start = time.perf_counter()
            request(queries[i % len(queries)])
            sync()
            times.append(time.perf_counter() - start)
    report = {'scope': 'CPU features + transfers + model + exact full-catalog top-K',
              'requests': requests, 'batch_size': 1, 'device': device, 'topk': topk,
              'catalog_items': len(data['items']), 'mock': store.manifest['mock'] or getattr(model, 'mock', False),
              'p50_ms': float(np.percentile(times, 50) * 1000),
              'p95_ms': float(np.percentile(times, 95) * 1000), 'requests_per_second': 1 / float(np.mean(times)),
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / (1024 ** 2) if str(device).startswith('cuda') else None}
    save_json(out, report)
    return report
