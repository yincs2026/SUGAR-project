"""Bridge between the offline snapshot store and the online training loop
(Algorithm 2, lines 4-9).

Contract (enforced by tests/test_snapshot_protocol.py):
- positions sharing the same visible snapshot form one chunk;
- side information is piecewise-constant in t;
- before the first committed snapshot: cold-start regime
  (p_u <- learnable p0, sim <- 0, n <- 0);
- the commit check is bypassed ONLY under the A7 leakage ablation.
"""
import numpy as np
import torch

from sugar.offline.snapshot_store import SnapshotStore


class SnapshotView:
    """Per-user access to snapshot side information during training/inference."""

    def __init__(self, store: SnapshotStore, user_idx: int,
                 d_e: int, strict: bool = True):
        self.snaps = store.snapshots_of(user_idx)
        self.d_e = d_e
        self.strict = strict  # False only for A7 ablation

    def at(self, t: int) -> dict:
        """Side info visible at sequence position t.

        Returns dict with keys: profile_emb (np or None), sim, n.
        Cold-start: profile_emb=None, sim=0, n=0.
        """
        if not self.snaps:
            return {"profile_emb": None, "sim": 0.0, "n": 0}
        if self.strict:
            cands = [s for s in self.snaps if s["t_commit"] <= t]
            snap = cands[-1] if cands else None
        else:
            snap = self.snaps[-1]  # A7: newest regardless of commit
        if snap is None:
            return {"profile_emb": None, "sim": 0.0, "n": 0}
        return {"profile_emb": snap["profile_emb"], "sim": snap["sim"],
                "n": snap["n_consolidated"]}

    def chunks(self, seq_len: int) -> list:
        """Chunk boundaries: (start, end_exclusive, side_info dict)."""
        commits = sorted({s["t_commit"] for s in self.snaps
                          if (self.strict and s["t_commit"] < seq_len)
                          or not self.strict})
        bounds = [0] + commits
        out = []
        for i, start in enumerate(bounds):
            end = bounds[i + 1] if i + 1 < len(bounds) else seq_len
            if self.strict and end == 0:
                continue
            info = self.at(min(start, max(seq_len - 1, 0))) if seq_len > 0 \
                else {"profile_emb": None, "sim": 0.0, "n": 0}
            out.append((start, end, info))
        return out


def build_batch_side_info(store: SnapshotStore, user_indices: list,
                          seq_lens: list, d_e: int,
                          strict: bool = True) -> list:
    """Per-user chunk plans for a training batch.

    Returns list (len B) of chunk lists; each chunk carries the visible
    profile embedding (or None for cold-start), sim, and n. The trainer
    (batch 10) processes positions chunk-by-chunk so that the profile
    vector is computed once per chunk, not per position.
    """
    plans = []
    for u, L in zip(user_indices, seq_lens):
        view = SnapshotView(store, int(u), d_e, strict=strict)
        plans.append(view.chunks(int(L)))
    return plans