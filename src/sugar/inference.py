from pathlib import Path
import numpy as np
import torch
from .train import load_run, Collator, move_batch


@torch.inference_mode()
def recommend(data_path, artifacts, run, user, position, k=10, device="cpu"):
    """Replay a recommendation at a dataset timestamp, with no target ID in model inputs."""
    model, data, store, config = load_run(data_path, artifacts, run, device)
    model.eval()
    user = str(user)
    if user not in data["sequences"]:
        raise ValueError("Use a mapped user ID from data.json")
    events = data["sequences"][user]
    if not 1 <= position < len(events) or k <= 0:
        raise ValueError("Position is the number of past events; requires a recorded request timestamp")
    batch = move_batch(Collator(data, store, config)([(user, position)]), device)
    h, gates = model(batch[0], batch[1])
    ids = torch.arange(len(data["items"]) + 1, device=device)
    scores = model.score_catalog(h, ids).squeeze(0).float().cpu().numpy()
    candidates = list(range(1, len(data["items"]) + 1))
    if config["mask_seen"]:
        seen = {v for v, _ in events[:position]}
        candidates = [v for v in candidates if v not in seen]
    candidates = [v for v in candidates if data["catalog"][str(v)]["available_at"] < events[position][1]]
    candidates.sort(key=lambda v: (-float(scores[v]), v))
    original_items = {v: key for key, v in data["items"].items()}
    return {"user": int(user), "observed_events": position, "request_time": events[position][1],
            "scope": "dataset replay", "mock": store.manifest["mock"],
            "items": [{"mapped_id": v, "item_id": original_items[v], "score": float(scores[v])} for v in candidates[:k]],
            "gates": {key: float(value[0]) for key, value in gates.items()}}
