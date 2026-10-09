
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


def _load_amazon(reviews_path: str, min_u: int, min_i: int):
    seqs = defaultdict(list)
    item_meta = {}
    opener = gzip.open if str(reviews_path).endswith(".gz") else open
    with opener(reviews_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="amazon raw"):
            r = json.loads(line)
            seqs[r["reviewerID"]].append((r["unixReviewTime"], r["asin"]))
            if r["asin"] not in item_meta:
                item_meta[r["asin"]] = {
                    "title": r.get("title", ""),
                    "categories": r.get("categories", []),
                    "text": (r.get("title", "") + ". " + r.get("summary", "")).strip(),
                }
    return seqs, item_meta


def _load_yelp(reviews_path: str, business_path: str, min_u: int, min_i: int):
    seqs = defaultdict(list)
    item_meta = {}

    # کسب‌وکارها: دسته‌ها + متن
    with open(business_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="yelp business"):
            b = json.loads(line)
            cats = [c.strip() for c in (b.get("categories") or "").split(",") if c.strip()]
            item_meta[b["business_id"]] = {
                "title": b.get("name", ""),
                "categories": cats,
                "text": f"{b.get('name', '')}. {'; '.join(cats)}",
            }

    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="yelp reviews"):
            r = json.loads(line)
            t = int(r["date"].replace("-", ""))
            seqs[r["user_id"]].append((t, r["business_id"]))
    return seqs, item_meta


def k_core_filter(seqs: dict, min_u: int, min_i: int) -> dict:
    while True:
        uc = Counter(i for s in seqs.values() for i in s)
        keep_users = {u for u, s in seqs.items() if len(s) >= min_u}
        keep_items = {i for i, c in uc.items() if c >= min_i}
        filtered = {}
        for u in keep_users:
            s = [i for i in seqs[u] if i in keep_items]
            if len(s) >= min_u:
                filtered[u] = s
        if (len(filtered) == len(keep_users)
                and all(len(s) >= min_u for s in filtered.values())):
            return filtered
        seqs = {u: [(0, i) for i in s] for u, s in filtered.items()}
        # در دور بعد فقط طول‌ها مهم‌اند؛ زمان‌ها در remap بازسازی می‌شوند


def dedup_and_sort(seqs: dict) -> dict:
    """حذف تعامل تکراری متوالی + مرتب‌سازی زمانی."""
    out = {}
    for u, s in seqs.items():
        s = sorted(set(s), key=lambda x: x[0])
        items, last = [], None
        for _, i in s:
            if i != last:         
                items.append(i)
            last = i
        out[u] = items
    return out


def build_social_edges(friends_path: str | None) -> list:
    if not friends_path:
        return []
    edges = []
    with open(friends_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="social edges"):
            r = json.loads(line)
            u = r.get("user_id")
            for w in r.get("friends", "").split(", "):
                if w:
                    edges.append((u, w))
    return edges


def prepare(cfg: dict, processed_root: str = "outputs/processed"):
    name = cfg["name"]
    out_dir = Path(processed_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg["format"]
    if fmt == "amazon":
        seqs, item_meta = _load_amazon(cfg["raw_files"]["reviews"],
                                       cfg["min_interactions"], cfg["min_item_interactions"])
    elif fmt == "yelp":
        seqs, item_meta = _load_yelp(cfg["raw_files"]["reviews"],
                                     cfg["raw_files"]["business"],
                                     cfg["min_interactions"], cfg["min_item_interactions"])
    else:
        raise ValueError(f"فرمت ناشناخته: {fmt}")

    seqs = dedup_and_sort(seqs)
    seqs = k_core_filter(seqs, cfg["min_interactions"], cfg["min_item_interactions"])

    if fmt == "amazon":
        raw_seqs, item_meta = _load_amazon(cfg["raw_files"]["reviews"], 1, 1)
    else:
        raw_seqs, item_meta = _load_yelp(cfg["raw_files"]["reviews"],
                                         cfg["raw_files"]["business"], 1, 1)
    final = {}
    for u, items in seqs.items():
        if u in raw_seqs:
            keep = set(items)
            timed = [x for x in raw_seqs[u] if x[1] in keep]
            timed = sorted(set(timed), key=lambda x: x[0])
            items_sorted, last = [], None
            for _, i in timed:
                if i != last:
                    items_sorted.append(i)
                last = i
            if len(items_sorted) >= cfg["min_interactions"]:
                final[u] = items_sorted

    used_items = sorted({i for s in final.values() for i in s})
    item_meta = {i: item_meta[i] for i in used_items}
    item2idx = {i: idx for idx, i in enumerate(used_items)}

    sequences = {u: [item2idx[i] for i in s] for u, s in final.items()}

    social_edges = []
    if cfg.get("social"):
        social_edges = build_social_edges(cfg["raw_files"].get("friends"))
        u2idx = {u: idx for idx, u in enumerate(final.keys())}
        social_edges = [(u2idx[a], u2idx[b]) for a, b in social_edges
                        if a in u2idx and b in u2idx]

    with open(out_dir / "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)
    with open(out_dir / "items.pkl", "wb") as f:
        pickle.dump({"item2idx": item2idx, "meta": item_meta}, f)
    with open(out_dir / "social_edges.pkl", "wb") as f:
        pickle.dump(social_edges, f)

    n_inter = sum(len(s) for s in sequences.values())
    stats = {
        "n_users": len(sequences),
        "n_items": len(used_items),
        "n_interactions": n_inter,
        "avg_seq_len": round(n_inter / max(len(sequences), 1), 2),
        "n_social_edges": len(social_edges),
        "sparsity": round(1 - n_inter / (len(sequences) * len(used_items)), 6),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[{name}] done: {stats}")