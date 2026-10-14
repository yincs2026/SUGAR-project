"""Canonical inputs and training-only behavior. All IDs mapped from 1; 0 is padding."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .utils import load_json, save_json, file_hash, digest


def timestamp(value):
    try:
        x = float(value)
    except ValueError:
        x = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        x = x.timestamp()
    if not np.isfinite(x):
        raise ValueError("Nonfinite timestamp")
    return x


def prepare(raw, out, config):
    raw, out = Path(raw), Path(out)
    rows = []
    with (raw / "interactions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if config["dataset"] == "douban-book" and "rating" in row and float(row["rating"]) < 4:
                continue
            rows.append((row["user_id"], row["item_id"], timestamp(row["timestamp"])))
    # Iteration is necessary for a genuine core, not just one pass of counts.
    while rows:
        uc, ic = Counter(r[0] for r in rows), Counter(r[1] for r in rows)
        filtered = [r for r in rows if uc[r[0]] >= config["core"] and ic[r[1]] >= config["core"]]
        if len(filtered) == len(rows):
            break
        rows = filtered
    if not rows:
        raise ValueError("Core filtering removed every event")
    users = {u: i + 1 for i, u in enumerate(sorted({r[0] for r in rows}))}
    items = {v: i + 1 for i, v in enumerate(sorted({r[1] for r in rows}))}
    catalog = {}
    import json
    with (raw / "items.jsonl").open(encoding="utf-8") as f:
        for line in f:
            v = json.loads(line)
            if v["item_id"] in items:
                if str(items[v["item_id"]]) in catalog:
                    raise ValueError("Duplicate catalog item")
                catalog[str(items[v["item_id"]])] = {
                    "text": v["text"], "category": v.get("category", "unknown"),
                    "available_at": timestamp(str(v.get("available_at", "-1e30")))}
    if len(catalog) != len(items):
        raise ValueError("Every retained item must have catalog text")
    sequences = defaultdict(list)
    for u, v, t in rows:
        if catalog[str(items[v])]["available_at"] > t:
            raise ValueError("Interaction precedes item availability")
        sequences[str(users[u])].append([items[v], t])
    for seq in sequences.values():
        seq.sort(key=lambda r: (r[1], r[0]))
        if len(seq) < 3:
            raise ValueError("Need at least three events per user")
        if any(a[1] >= b[1] for a, b in zip(seq, seq[1:])):
            raise ValueError("Tied user timestamps: resolve using genuine finer timestamps, not fabricated times")
    graph = defaultdict(dict)
    social = raw / "social.csv"
    if social.exists():
        with social.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["user_id"] in users and r["neighbor_id"] in users:
                    u, w = str(users[r["user_id"]]), users[r["neighbor_id"]]
                    if int(u) != w:
                        time = timestamp(r.get("timestamp") or "-1e30")
                        graph[u][str(w)] = min(time, graph[u].get(str(w), time))
                        if config["dataset"] == "yelp":
                            graph[str(w)][u] = min(time, graph[str(w)].get(u, time))
    counts = {"users": len(users), "items": len(items), "interactions": len(rows),
              "social_edges": sum(map(len, graph.values()))}
    result = {"users": users, "items": items, "catalog": catalog, "sequences": dict(sequences),
              "graph": dict(graph), "counts": counts,
              "sources": {p.name: file_hash(p) for p in [raw / "interactions.csv", raw / "items.jsonl", social] if p.exists()},
              "prepare_config": {k: config[k] for k in ["dataset", "core"]}}
    if (raw / "conversion_notes.json").exists():
        result["input_notes"] = load_json(raw / "conversion_notes.json")
        result["sources"]["conversion_notes.json"] = file_hash(raw / "conversion_notes.json")
    result["fingerprint"] = digest(result)
    save_json(out / "data.json", result)
    print(counts, flush=True)
    return result


def proxy_graph(data, minimum=3):
    inverse = defaultdict(set)
    for u, seq in data["sequences"].items():
        for item in {r[0] for r in seq[:-2]}:
            inverse[item].add(u)
    counts = Counter()
    for group in inverse.values():
        group = sorted(group, key=int)
        for i, u in enumerate(group):
            for w in group[i + 1:]:
                counts[u, w] += 1
    graph = defaultdict(dict)
    for (u, w), n in counts.items():
        if n >= minimum:
            # Static train-only proxy graph: temporality caveat documented.
            graph[u][w] = -1e30
            graph[w][u] = -1e30
    data["graph"] = dict(graph)
    data["proxy_graph"] = {"minimum_common_items": minimum, "scope": "training-only static"}
    data.pop("fingerprint", None)
    data["fingerprint"] = digest(data)
    return data


def demo(raw, seed=123, users=12, items=18, events=65):
    import json
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with (raw / "items.jsonl").open("w", encoding="utf-8") as f:
        for v in range(items):
            topic = ["sport", "books", "technology"][v % 3]
            f.write(json.dumps({"item_id": str(v), "text": f"{topic} tutorial item {v}", "category": topic}) + "\n")
    with (raw / "interactions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "timestamp"])
        for u in range(users):
            for t in range(events):
                topic = (u + (t >= 30)) % 3
                options = [v for v in range(items) if v % 3 == topic]
                v = (t * 3 + u % 3) % items if t < items else int(rng.choice(options))
                w.writerow([u, v, 1000 * t + u])
    with (raw / "social.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "neighbor_id", "timestamp"])
        for u in range(users):
            w.writerow([u, (u + 1) % users, 0])
