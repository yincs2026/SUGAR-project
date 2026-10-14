from dataclasses import dataclass, field
from pathlib import Path
import math
import time
import numpy as np
from .semantics import SemanticBackend
from .utils import canonical, digest, save_json, load_json, file_hash


def cosine(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def recent_vector(embeddings, ids, decay=0.0):
    vecs = embeddings[ids]
    if decay:
        weights = np.exp(-decay * np.arange(len(ids) - 1, -1, -1))
        return np.average(vecs, axis=0, weights=weights).astype(np.float32)
    return vecs.mean(0)


@dataclass
class Detector:
    threshold: str = "personalized"
    tau: float = 0.7
    xi: float = 1.0
    warmup: int = 2
    cooldown: int = 50
    single: bool = False
    state: str = "Idle"
    cooldown_until: int = 0
    scores: list = field(default_factory=list)

    def check(self, score, position):
        # Past-only statistics, before incorporating the current observation.
        threshold = self.tau
        if self.threshold == "personalized" and len(self.scores) >= self.warmup:
            threshold = float(np.mean(self.scores) - self.xi * np.std(self.scores, ddof=0))
        trigger = False
        if self.state == "Cooldown" and position >= self.cooldown_until:
            self.state = "Idle"
        if self.state not in {"Revising", "Cooldown"}:
            if score < threshold:
                if self.state == "Suspect" or self.single:
                    self.state = "Revising"
                    trigger = True
                else:
                    self.state = "Suspect"
            else:
                self.state = "Idle"
        self.scores.append(float(score))
        return trigger, threshold

    def committed(self, position):
        self.state = "Cooldown"
        self.cooldown_until = position + self.cooldown


def build_snapshots(data, out, config, cache):
    wall_start = time.perf_counter()
    out = Path(out)
    backend = SemanticBackend(config, cache)
    nitems = len(data["items"])
    descriptors = {str(i): backend.generate("item", data["catalog"][str(i)]) for i in range(1, nitems + 1)}
    encoded = backend.encode([canonical(descriptors[str(i)]) for i in range(1, nitems + 1)])
    embeddings = np.concatenate([np.zeros((1, encoded.shape[1]), dtype=np.float32), encoded])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "item_embeddings.npy", embeddings)
    save_json(out / "descriptors.json", descriptors)
    all_snapshots, logs = {}, []
    for user, full in data["sequences"].items():
        sequence = full[:-2] # held-out behavior is categorically excluded
        ids = [r[0] for r in sequence]
        snapshots, consolidated = [], set() # event indices, NOT item IDs (repeats count)
        profile, vec, pending = None, np.zeros(encoded.shape[1], dtype=np.float32), None
        detector = Detector(config["threshold"], config["tau"], config["xi"], config["threshold_warmup"],
                            config["cooldown"], config["single_strike"])
        initial_score = None
        for position in range(config["checkpoint"], len(sequence) + 1, config["checkpoint"]):
            if pending and pending["commit_pos"] <= position:
                profile, vec, consolidated = pending["profile"], pending["vector"], pending["consolidated"]
                if pending["kind"] == "revise":
                    detector.committed(pending["commit_pos"])
                pending = None
            recent_indices = set(range(max(0, position - config["k"]), position))
            recent = recent_vector(embeddings, ids[max(0, position - config["k"]):position], config["recent_decay"])
            kind, next_profile, next_vec, next_consolidated = None, profile, vec, set(consolidated)
            if pending is not None:
                # This later snapshot can reference the staged artifact because its own
                # commit is after the dependent job; it must not overwrite it with old state.
                next_profile, next_vec = pending["profile"], pending["vector"]
                next_consolidated = set(pending["consolidated"])
            if pending is None:
                if profile is None and position >= config["min_profile"]:
                    kind = "build"
                    next_profile = backend.generate(kind, {"descriptors": [descriptors[str(v)] for v in ids[:position]]})
                    next_consolidated = set(range(position))
                elif profile is not None:
                    delta = sorted(set(range(position)) - recent_indices - consolidated)
                    if len(delta) >= config["edit_size"]:
                        kind = "edit"
                        next_profile = backend.generate(kind, {"profile": profile,
                            "descriptors": [descriptors[str(ids[i])] for i in delta]})
                        next_consolidated.update(delta)
                if kind:
                    next_vec = backend.profile_vector(next_profile)
            score = cosine(next_vec, recent) if next_profile else 0.0
            if next_profile and initial_score is None:
                initial_score = score
            trigger, threshold = False, None
            if next_profile and pending is None and not config["no_drift"]:
                trigger, threshold = detector.check(score, position)
                if trigger:
                    # When both consolidation and revision fire, both calls are accounted.
                    kind = "revise"
                    next_profile = backend.generate(kind, {"profile": next_profile,
                        "descriptors": [descriptors[str(ids[i])] for i in sorted(recent_indices)]})
                    next_consolidated.update(recent_indices)
                    next_vec = backend.profile_vector(next_profile)
                    score = cosine(next_vec, recent)
            if config["no_drift"] and initial_score is not None:
                score = initial_score
            commit_pos = position + config["lag"]
            commit_time = sequence[commit_pos - 1][1] if commit_pos <= len(sequence) else None
            # If an earlier edit/revision is still running, publish only old committed state.
            snap = {"user": int(user), "checkpoint": position // config["checkpoint"],
                    "source_pos": position, "source_time": sequence[position - 1][1],
                    "commit_pos": commit_pos, "commit_time": commit_time,
                    "profile": next_profile, "vector": next_vec.tolist(), "sim": score,
                    "n": len(next_consolidated), "evidence_indices": sorted(next_consolidated),
                    "recent_vector": recent.tolist(), "kind": kind}
            snapshots.append(snap)
            if kind:
                pending = {"commit_pos": commit_pos, "kind": kind, "profile": next_profile,
                           "vector": next_vec, "consolidated": next_consolidated}
                if config["lag"] == 0:
                    profile, vec, consolidated = next_profile, next_vec, next_consolidated
                    if kind == "revise":
                        detector.committed(position)
                    pending = None
            logs.append({"user": int(user), "checkpoint": snap["checkpoint"], "position": position,
                         "pre_revision_score": detector.scores[-1] if detector.scores else None,
                         "score": score, "threshold": threshold, "trigger": trigger, "state": detector.state,
                         "kind": kind, "commit_pos": commit_pos})
        all_snapshots[user] = snapshots
    save_json(out / "snapshots.json", all_snapshots)
    save_json(out / "offline_log.json", logs)
    save_json(out / "llm_usage.json", backend.usage)
    keys = ["k", "checkpoint", "edit_size", "min_profile", "cooldown", "lag", "threshold", "xi", "tau",
            "threshold_warmup", "single_strike", "no_drift", "profile_embedding", "recent_decay"]
    manifest = {"data_fingerprint": data["fingerprint"], "semantic_identity": backend.identity,
                "offline_config": {k: config[k] for k in keys}, "mock": backend.mock,
                "files": {name: file_hash(out / name) for name in ["snapshots.json", "descriptors.json", "item_embeddings.npy", "offline_log.json", "llm_usage.json"]}}
    save_json(out / "offline_timing.json", {"wall_seconds": time.perf_counter() - wall_start, "scope": "M1-M3 including encoding, cache IO, and snapshot writing", "mock": backend.mock})
    manifest["files"]["offline_timing.json"] = file_hash(out / "offline_timing.json")
    save_json(out / "manifest.json", manifest)
    return manifest


def select_snapshot(snapshots, time, position=None):
    # Strict target-time bound handles simultaneous users without lookahead.
    selected = None
    for snap in snapshots:
        if snap["commit_time"] is not None and snap["commit_time"] < time and (position is None or snap["commit_pos"] <= position):
            selected = snap
    if selected:
        if selected["source_time"] >= time or selected["source_pos"] > selected["commit_pos"]:
            raise AssertionError("Leaked snapshot")
        if any(i >= selected["source_pos"] for i in selected["evidence_indices"]):
            raise AssertionError("Leaked profile evidence")
    return selected


def verify_manifest(data, directory):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    if digest({k: v for k, v in data.items() if k != "fingerprint"}) != data["fingerprint"]:
        raise ValueError("Processed dataset content hash is invalid")
    if manifest["data_fingerprint"] != data["fingerprint"]:
        raise ValueError("Artifacts belong to another processed dataset")
    for filename, expected in manifest["files"].items():
        if file_hash(directory / filename) != expected:
            raise ValueError(f"Artifact integrity mismatch: {filename}")
    return manifest
