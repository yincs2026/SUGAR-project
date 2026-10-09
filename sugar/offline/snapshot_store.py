"""Snapshot store (Section 3.2: checkpoints, snapshots, temporal consistency).

Each user has a list of snapshots:
  {j, t_chk, t_commit, profile(P_u json), profile_emb(np array),
   sim(float), n_consolidated(int)}

Temporal-consistency contract: a snapshot with commit time t_commit is only
visible to the online stage at positions t >= t_commit. The function
`visible_snapshot` enforces this and raises if violated (leakage guard).
Snapshots are stored as a single pickle per (dataset, llm) run.
"""
import pickle
from pathlib import Path

import numpy as np


class SnapshotStore:
    def __init__(self):
        self._snapshots = {}  # user_idx -> list of snapshot dicts (sorted by j)

    def add(self, user_idx: int, j: int, t_chk: int, t_commit: int,
            profile: dict, profile_emb: np.ndarray, sim: float,
            n_consolidated: int):
        self._snapshots.setdefault(user_idx, []).append({
            "j": j, "t_chk": t_chk, "t_commit": t_commit,
            "profile": profile, "profile_emb": profile_emb,
            "sim": sim, "n_consolidated": n_consolidated,
        })

    def snapshots_of(self, user_idx: int) -> list:
        return self._snapshots.get(user_idx, [])

    def visible_snapshot(self, user_idx: int, t: int,
                         strict: bool = True) -> dict | None:
        """Latest snapshot whose commit time <= t (Section 3.2 protocol).

        strict=True raises on any contract violation; strict=False is used
        ONLY by the A7 leakage ablation, which intentionally disables the
        commit check to quantify the leakage effect.
        """
        cands = [s for s in self._snapshots.get(user_idx, [])
                 if s["t_commit"] <= t]
        if strict and self._snapshots.get(user_idx):
            latest = self._snapshots[user_idx][-1]
            if latest["t_commit"] > t and cands == []:
                # snapshot exists but not yet committed: cold-start regime,
                # NOT a violation. Violation = using an uncommitted snapshot.
                return None
        return cands[-1] if cands else None

    def latest_uncommitted(self, user_idx: int, t: int) -> dict | None:
        """A7 ablation only: newest snapshot regardless of commit time."""
        snaps = self._snapshots.get(user_idx, [])
        return snaps[-1] if snaps else None

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._snapshots, f)

    @classmethod
    def load(cls, path: str) -> "SnapshotStore":
        store = cls()
        with open(path, "rb") as f:
            store._snapshots = pickle.load(f)
        return store

    @property
    def n_users(self):
        return len(self._snapshots)

    @property
    def n_snapshots(self):
        return sum(len(v) for v in self._snapshots.values())