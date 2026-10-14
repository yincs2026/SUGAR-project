#!/usr/bin/env python3
"""Alternative BUSINESS item interpretation; it cannot reproduce the ambiguous review-item definition."""
import argparse
import csv
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--business", required=True)
p.add_argument("--review", required=True)
p.add_argument("--user", required=True)
p.add_argument("--acknowledge-business-items", action="store_true", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with open(a.business, encoding="utf-8") as source, (out / "items.jsonl").open("w", encoding="utf-8") as dest:
    for line in source:
        b = json.loads(line)
        # No future review text aggregated into static item features.
        text = b.get("name", "") + " | " + (b.get("categories") or "")
        dest.write(json.dumps({"item_id": b["business_id"], "text": text,
                               "category": (b.get("categories") or "unknown").split(",")[0].strip()}, ensure_ascii=False) + "\n")
with open(a.review, encoding="utf-8") as source, (out / "interactions.csv").open("w", newline="") as dest:
    writer = csv.writer(dest)
    writer.writerow(["user_id", "item_id", "timestamp", "rating"])
    for line in source:
        r = json.loads(line)
        writer.writerow([r["user_id"], r["business_id"], r["date"], r["stars"]])
with open(a.user, encoding="utf-8") as source, (out / "social.csv").open("w", newline="") as dest:
    writer = csv.writer(dest)
    writer.writerow(["user_id", "neighbor_id"])
    for line in source:
        u = json.loads(line)
        for friend in u.get("friends", "").split(","):
            friend = friend.strip()
            if friend and friend != "None":
                writer.writerow([u["user_id"], friend])
(out / "conversion_notes.json").write_text(json.dumps({"item_unit": "business", "text": "business name and categories only",
    "paper_equivalence": False, "social_edges": "static untimestamped", "no_review_text_used": True}, indent=2))
