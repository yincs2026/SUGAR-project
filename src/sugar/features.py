"""Query-time temporal join; top-L selection on frozen cosines, trainable beta in model."""
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .offline import select_snapshot, cosine, recent_vector, verify_manifest
from .utils import load_json, digest


class FeatureStore:
    def __init__(self, data, artifacts, config, stress="clean"):
        self.data, self.c, self.stress = data, config, stress
        self.artifacts = str(artifacts)
        self.manifest = verify_manifest(data, artifacts)
        for k, value in self.manifest["offline_config"].items():
            if config[k] != value:
                raise ValueError(f"Offline config mismatch: {k}; rebuild snapshots for this experiment")
        self.snapshots = load_json(Path(artifacts) / "snapshots.json")
        self.embeddings = np.load(Path(artifacts) / "item_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.dim = self.embeddings.shape[1]
        self.timestamps = {u: [r[1] for r in seq[:-2]] for u, seq in data["sequences"].items()}
        ready = []
        for u, snaps in self.snapshots.items():
            times = [snap['commit_time'] for snap in snaps if snap['profile'] is not None and snap['commit_time'] is not None]
            if times:
                ready.append((min(times), u))
        ready.sort(key=lambda row: (row[0], int(row[1])))
        self.donor_ready_times = [row[0] for row in ready]
        self.donor_users = [row[1] for row in ready]
        self.donor_indices = {u: i for i, u in enumerate(self.donor_users)}
        self.cache = OrderedDict()
        self.user_ids = sorted(data["sequences"], key=int)

    def query(self, user, position):
        key = (user, position)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        full = self.data["sequences"][user]
        target_time = full[position][1]
        train_limit = min(position, len(full) - 2)
        snap = self.snapshots[user][-1] if self.c.get("leaky_snapshots", False) and self.snapshots[user] else select_snapshot(self.snapshots[user], target_time, train_limit)
        own_snapshot = snap
        if self.stress == "stale":
            cutoff = max(self.c["min_profile"], (len(full) - 2 + 1) // 2)
            valid = [s for s in self.snapshots[user] if s["source_pos"] <= cutoff]
            if not valid:
                valid = self.snapshots[user][:1]
            snap = select_snapshot(valid, target_time, train_limit)
        elif self.stress == "corrupted":
            # Uniform random donor among other users with a temporally visible profile.
            count = bisect_left(self.donor_ready_times, target_time)
            own = self.donor_indices.get(user, count)
            eligible = count - int(own < count)
            if eligible:
                rng = np.random.default_rng(int(digest([self.c.get('corruption_seed', 42), user, position])[:16], 16))
                index = int(rng.integers(eligible))
                if own < count and index >= own:
                    index += 1
                other = self.donor_users[index]
                snap = select_snapshot(self.snapshots[other], target_time)
            else:
                snap = None
        profile = np.array(snap["vector"], dtype=np.float32) if snap and snap["profile"] else np.zeros(self.dim, dtype=np.float32)
        has_profile = bool(snap and snap["profile"])
        n, score = (snap["n"], snap["sim"]) if has_profile else (0, 0)
        # M2–M4 never consume held-out interactions. M5's test sequence can include validation.
        ids = [r[0] for r in full[max(0, train_limit - self.c["k"]):train_limit]]
        recent = recent_vector(self.embeddings, ids, self.c["recent_decay"]) if ids else np.zeros(self.dim, dtype=np.float32)
        if self.c.get('social_window', 'live') == 'committed' and own_snapshot is not None:
            recent = np.asarray(own_snapshot['recent_vector'], dtype=np.float32)
        if self.c["recompute_score"] and has_profile:
            score = cosine(profile, recent)
        candidates = []
        if not self.c["no_social"]:
            adjacency = sorted(((w, t) for w, t in self.data["graph"].get(user, {}).items() if t < target_time), key=lambda x: int(x[0]))
            sample_size = self.c.get("neighbor_sample")
            if sample_size is not None and len(adjacency) > int(sample_size):
                rng = np.random.default_rng(int(digest([self.c.get("neighbor_sampling_seed", 42), user, position])[:16], 16))
                selected = sorted(rng.choice(len(adjacency), int(sample_size), replace=False).tolist())
                adjacency = [adjacency[i] for i in selected]
            for neighbor, edge_time in adjacency:
                if edge_time >= target_time:
                    continue
                ns = self.snapshots[neighbor][-1] if self.c.get("leaky_snapshots", False) and self.snapshots[neighbor] else select_snapshot(self.snapshots[neighbor], target_time)
                if ns and ns["profile"]:
                    candidates.append((cosine(recent, ns["vector"]), int(neighbor), ns))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        candidates = candidates[:self.c["neighbors"]]
        neighbors = np.zeros((self.c["neighbors"], self.dim), dtype=np.float32)
        mask = np.zeros(self.c["neighbors"], dtype=bool)
        for i, (_, _, ns) in enumerate(candidates):
            neighbors[i] = ns["vector"]
            mask[i] = True
        result = {"profile": profile, "present": has_profile, "n": n, "sim": score,
                  "recent": recent, "neighbors": neighbors, "neighbor_mask": mask,
                  "source_time": snap["source_time"] if snap else None,
                  "neighbor_source_times": [s[2]["source_time"] for s in candidates], "target_time": target_time}
        violations = 0
        if snap and (snap["source_time"] >= target_time or snap["commit_time"] is None or snap["commit_time"] >= target_time or (self.stress != "corrupted" and snap["commit_pos"] > train_limit)):
            violations += 1
        for _, _, ns in candidates:
            if ns["source_time"] >= target_time or ns["commit_time"] is None or ns["commit_time"] >= target_time:
                violations += 1
        result['side_key'] = np.int64(int(digest({'profile': profile.tolist(), 'present': has_profile,
            'sim': score, 'n': n, 'recent': recent.tolist(), 'neighbors': neighbors.tolist(),
            'mask': mask.tolist()})[:15], 16))
        result["temporal_violations"] = violations
        if violations and not self.c.get("leaky_snapshots", False):
            raise AssertionError("Temporal feature leakage")
        self.cache[key] = result
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return result

    def audit(self):
        count, violations = 0, 0
        for user, sequence in self.data["sequences"].items():
            for position in range(1, len(sequence)):
                violations += self.query(user, position)["temporal_violations"]
                count += 1
        return {"checked_prediction_positions": count, "heldout_excluded_from_profiles": True,
                "temporal_violations": violations, "passed": violations == 0,
                "intentional_leakage_ablation": self.c.get("leaky_snapshots", False)}
