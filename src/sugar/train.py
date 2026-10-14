from pathlib import Path
from time import perf_counter
import os
import platform
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .features import FeatureStore
from .factory import build_model
from .utils import save_json, load_json, seed_all, environment, digest, file_hash


def restore_model(model, state, expected_provenance=None):
    if expected_provenance is not None and (not hasattr(model, "provenance") or model.provenance() != expected_provenance):
        raise ValueError("Frozen model/backbone/semantic provenance changed since checkpoint creation")
    if hasattr(model, "load_checkpoint_state"):
        model.load_checkpoint_state(state)
    else:
        model.load_state_dict(state)


class Queries(Dataset):
    def __init__(self, data, split):
        self.entries = []
        for u, seq in data["sequences"].items():
            positions = range(1, len(seq) - 2) if split == "train" else [len(seq) - (2 if split == "validation" else 1)]
            self.entries.extend((u, pos) for pos in positions)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


class Collator:
    def __init__(self, data, store, config):
        self.data, self.store, self.c = data, store, config

    def __call__(self, queries):
        histories = [[r[0] for r in self.data["sequences"][u][max(0, pos - self.c["max_len"]):pos]] for u, pos in queries]
        length = max(map(len, histories))
        seq = np.zeros((len(queries), length), dtype=np.int64)
        for i, history in enumerate(histories):
            seq[i, :len(history)] = history
        fs = [self.store.query(u, pos) for u, pos in queries]
        features = {k: torch.as_tensor(np.stack([f[k] for f in fs])) for k in ["profile", "present", "n", "sim", "recent", "neighbors", "neighbor_mask", "side_key"]}
        features["n"] = features["n"].float()
        features["sim"] = features["sim"].float()
        features["user_id"] = torch.tensor([int(u) for u, _ in queries], dtype=torch.long)
        targets = torch.tensor([self.data["sequences"][u][pos][0] for u, pos in queries])
        return torch.from_numpy(seq), features, targets, queries


def move_batch(batch, device):
    sequence, features, target, queries = batch
    return sequence.to(device), {k: v.to(device) for k, v in features.items()}, target.to(device), queries


def sample_negatives(data, queries, count, rng):
    nitems = len(data["items"])
    result = []
    for user, _ in queries:
        excluded = {r[0] for r in data["sequences"][user][:-2]}
        if len(excluded) == nitems:
            raise ValueError(f"User {user} has no unobserved training negatives")
        row = []
        # Uniform rejection sampling, no validation/test consulted for negative exclusion.
        while len(row) < count:
            candidate = int(rng.integers(1, nitems + 1))
            if candidate not in excluded:
                row.append(candidate)
        result.append(row)
    return torch.tensor(result, dtype=torch.long)


def rank_metrics(rank):
    return {"HR@5": float(rank <= 5), "HR@10": float(rank <= 10), "HR@20": float(rank <= 20),
            "NDCG@10": float(1 / np.log2(rank + 1)) if rank <= 10 else 0.0}


@torch.inference_mode()
def evaluate(model, data, store, config, split="test", device="cpu", chunk=4096):
    model.eval()
    loader = DataLoader(Queries(data, split), batch_size=config["batch_size"], shuffle=False,
                        collate_fn=Collator(data, store, config), num_workers=0)
    records = []
    for batch in loader:
        seq, features, target, queries = move_batch(batch, device)
        h, gates = model(seq, features)
        positive_scores = model.score_targets(h, target)
        ranks = torch.ones(len(target), dtype=torch.long, device=device)
        # Full-catalog ranking streamed in chunks; deterministic smaller-ID tiebreak.
        for start in range(1, len(data["items"]) + 1, chunk):
            stop = min(start + chunk, len(data["items"]) + 1)
            ids = torch.arange(start, stop, device=device)
            scores = model.score_catalog(h, ids)
            allowed = torch.ones_like(scores, dtype=torch.bool)
            for row, (u, pos) in enumerate(queries):
                if config["mask_seen"]:
                    seen = {r[0] for r in data["sequences"][u][:pos]} - {int(target[row])}
                    masked = [v - start for v in seen if start <= v < stop]
                    allowed[row, masked] = False
                time = data["sequences"][u][pos][1]
                unavailable = [v - start for v in range(start, stop) if data["catalog"][str(v)]["available_at"] >= time]
                allowed[row, unavailable] = False
            better = (scores > positive_scores[:, None]) | ((scores == positive_scores[:, None]) & (ids[None, :] < target[:, None]))
            better &= ids[None, :] != target[:, None]
            ranks += (better & allowed).sum(1)
        for row, (u, pos) in enumerate(queries):
            ntrain = len(data["sequences"][u]) - 2
            bucket = "below5" if ntrain < 5 else "sparse" if ntrain <= 9 else "medium" if ntrain <= 29 else "dense"
            rank = int(ranks[row].item())
            records.append({"user": int(u), "target": int(target[row].item()), "position": pos, "rank": rank,
                            "bucket": bucket, "profile_present": bool(features["present"][row]),
                            "evidence_count": float(features["n"][row]), "freshness": float(features["sim"][row]), **rank_metrics(rank),
                            **{k: float(v[row].item()) for k, v in gates.items()}})
    keys = ["HR@5", "HR@10", "HR@20", "NDCG@10", "rho", "gamma", "gate"]
    metrics = {k: float(np.mean([r[k] for r in records])) for k in keys}
    buckets = {}
    for bucket in sorted({r["bucket"] for r in records}):
        members = [r for r in records if r["bucket"] == bucket]
        buckets[bucket] = {"users": len(members), **{k: float(np.mean([r[k] for r in members])) for k in keys}}
    return {"split": split, "metrics": metrics, "buckets": buckets, "users": records}


def train(data_path, artifacts, out, config, device="cpu", resume=None, evaluate_test=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(config["seed"])
    out = Path(out)
    if (out / "last.pt").exists() and not resume:
        raise ValueError("Run directory already has a checkpoint; use a new directory or --resume")
    data = load_json(data_path)
    store = FeatureStore(data, artifacts, config)
    if (store.manifest["mock"] or (config["model"] == "llara" and config.get("llara_backend") == "tiny")) and not config["allow_nonpaper"]:
        raise ValueError("Mock semantic artifacts require allow_nonpaper: true")
    audit = store.audit()
    save_json(out / "leakage_audit.json", audit)
    model = build_model(data, store, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(Queries(data, "train"), batch_size=config["batch_size"], shuffle=True, generator=generator,
                        collate_fn=Collator(data, store, config), num_workers=0)
    identity = {"config": config, "data": data["fingerprint"], "artifacts": digest(store.manifest)}
    experiment_hash = digest(identity)
    history, best, stale, start_epoch = [], -float("inf"), 0, 0
    if resume:
        if not (out / "best.pt").exists():
            raise ValueError("Resume in the original run directory containing best.pt")
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["experiment_hash"] != experiment_hash:
            raise ValueError("Resume configuration/data/artifacts mismatch")
        restore_model(model, state["model"], state.get("model_provenance"))
        optimizer.load_state_dict(state["optimizer"])
        history, best, stale, start_epoch = state["history"], state["best"], state["stale"], state["epoch"] + 1
        rng.bit_generator.state = state["numpy_rng"]
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    if hasattr(model, "provenance"):
        save_json(out / "model_provenance.json", model.provenance())
    save_json(out / "config.json", config)
    save_json(out / "provenance.json", {**identity, "environment": environment(), "experiment_hash": experiment_hash,
                                       "mock": store.manifest["mock"], "device": device,
                                       "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        start, sums, count = perf_counter(), np.zeros(3), 0
        for batch in loader:
            seq, features, target, queries = move_batch(batch, device)
            neg = sample_negatives(data, queries, config["negatives"], rng).to(device)
            align_neg = sample_negatives(data, queries, config["alignment_negatives"], rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            h, _ = model(seq, features)
            loss, rec, align = model.loss(h, target, neg, align_neg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), rec.item(), align.item()]) * len(target)
            count += len(target)
        validation = evaluate(model, data, store, config, "validation", device)
        ndcg = validation["metrics"]["NDCG@10"]
        improved = ndcg > best
        best, stale = (ndcg, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch + 1, "loss": (sums / count).tolist(), "seconds": perf_counter() - start,
               "validation": validation["metrics"]}
        history.append(row)
        print(row, flush=True)
        state = {"epoch": epoch, "model_provenance": model.provenance() if hasattr(model, "provenance") else None, "model": model.checkpoint_state() if hasattr(model, "checkpoint_state") else model.state_dict(), "optimizer": optimizer.state_dict(),
                 "history": history, "best": best, "stale": stale, "config": config,
                 "experiment_hash": experiment_hash, "numpy_rng": rng.bit_generator.state,
                 "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "python_rng": random.getstate(),
                 "data_fingerprint": data["fingerprint"], "artifact_hash": digest(store.manifest)}
        torch.save(state, out / "last.pt")
        if improved:
            torch.save(state, out / "best.pt")
            save_json(out / "validation.json", validation)
        save_json(out / "history.json", history)
        if stale >= config["patience"]:
            break
    # Re-evaluate only the validation-selected checkpoint, never pick using test results.
    best_state = torch.load(out / "best.pt", map_location=device, weights_only=False)
    restore_model(model, best_state["model"], best_state.get("model_provenance"))
    result = evaluate(model, data, store, config, "test" if evaluate_test else "validation", device)
    result["mock"] = store.manifest["mock"] or getattr(model, "mock", False)
    result["intentional_leakage_ablation"] = config.get("leaky_snapshots", False)
    result["best_epoch"] = best_state["epoch"] + 1
    result["experiment_hash"] = experiment_hash
    save_json(out / ("test.json" if evaluate_test else "selected_validation.json"), result)
    return result


def load_run(data_path, artifacts, run, device="cpu", stress="clean", force_gate=None):
    data = load_json(data_path)
    state = torch.load(Path(run) / "best.pt", map_location=device, weights_only=False)
    config = dict(state["config"])
    config["force_gate"] = force_gate
    seed_all(config["seed"])
    store = FeatureStore(data, artifacts, config, stress)
    if data["fingerprint"] != state["data_fingerprint"] or digest(store.manifest) != state["artifact_hash"]:
        raise ValueError("Checkpoint was trained with different data/artifacts")
    model = build_model(data, store, config, device)
    restore_model(model, state["model"], state.get("model_provenance"))
    return model, data, store, config
