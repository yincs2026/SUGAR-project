#!/usr/bin/env python3
"""Normalize timestamped Douban/Epinions exports without inventing metadata/timestamps."""
import argparse
import csv
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--ratings", required=True)
p.add_argument("--user-column", default="user_id")
p.add_argument("--item-column", default="item_id")
p.add_argument("--time-column", default="timestamp")
p.add_argument("--rating-column", default="rating")
p.add_argument("--items", required=True, help="CSV with item_id,text,category,available_at")
p.add_argument("--social", required=True, help="CSV with user_id,neighbor_id[,timestamp]")
p.add_argument("--delimiter", default=",")
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with open(a.ratings, newline="", encoding="utf-8") as source, (out / "interactions.csv").open("w", newline="") as dest:
    reader = csv.DictReader(source, delimiter=a.delimiter)
    required = [a.user_column, a.item_column, a.time_column]
    if not set(required) <= set(reader.fieldnames):
        raise SystemExit("Source must contain real user, item and timestamp columns")
    writer = csv.DictWriter(dest, fieldnames=["user_id", "item_id", "timestamp", "rating"])
    writer.writeheader()
    for r in reader:
        writer.writerow({"user_id": r[a.user_column], "item_id": r[a.item_column], "timestamp": r[a.time_column], "rating": r.get(a.rating_column, "5")})
with open(a.items, newline="", encoding="utf-8") as source, (out / "items.jsonl").open("w", encoding="utf-8") as dest:
    for r in csv.DictReader(source, delimiter=a.delimiter):
        obj = {"item_id": r["item_id"], "text": r["text"], "category": r.get("category") or "unknown"}
        if r.get("available_at"):
            obj["available_at"] = r["available_at"]
        dest.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(a.social, newline="", encoding="utf-8") as source, (out / "social.csv").open("w", newline="") as dest:
    writer = csv.DictWriter(dest, fieldnames=["user_id", "neighbor_id", "timestamp"])
    writer.writeheader()
    for r in csv.DictReader(source, delimiter=a.delimiter):
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
