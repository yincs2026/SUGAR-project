"""Protocol tests: chunking contract and cold-start regime (Algorithm 2, lines 4-6)."""
import numpy as np
import pytest

from sugar.offline.snapshot_store import SnapshotStore


def chunk_positions(snapshots: list, seq_len: int) -> list:
    """Reference implementation of the chunking rule: consecutive positions
    sharing the same visible snapshot are processed in one chunk.

    Returns list of (chunk_start, chunk_end_exclusive, snapshot_or_None).
    Used here as the specification the training loader must match (batch 9).
    """
    if not snapshots:
        return [(0, seq_len, None)]
    boundaries = [0] + [s["t_commit"] for s in snapshots if s["t_commit"] < seq_len]
    boundaries = sorted(set(boundaries))
    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
        visible = None
        for s in snapshots:
            if s["t_commit"] <= start:
                visible = s
        chunks.append((start, end, visible))
    return chunks


class TestChunking:
    def test_piecewise_constant(self):
        """Side information fed to the backbone is piecewise-constant in t."""
        store = SnapshotStore()
        for j, commit in enumerate([10, 20, 30], start=1):
            store.add(0, j=j, t_chk=j * 10, t_commit=commit,
                      profile={}, profile_emb=np.full(4, j), sim=0.5,
                      n_consolidated=j * 10)
        snaps = store.snapshots_of(0)
        chunks = chunk_positions(snaps, seq_len=45)
        # positions 0-9: cold (None); 10-19: snap j=1; 20-29: j=2; 30-44: j=3
        assert chunks[0] == (0, 10, None)
        assert chunks[1] == (10, 20, snaps[0])
        assert chunks[2] == (20, 30, snaps[1])
        assert chunks[3] == (30, 45, snaps[2])

    def test_cold_start_regime(self):
        """Before the first visible snapshot: p0, sim=0, n=0 (Section 3.2)."""
        store = SnapshotStore()
        store.add(0, j=1, t_chk=10, t_commit=15, profile={},
                  profile_emb=np.ones(4), sim=0.7, n_consolidated=10)
        snap = store.visible_snapshot(0, t=14)
        assert snap is None
        # the online stage substitutes: p_u <- p0 (learnable), sim <- 0, n <- 0
        # so reliability rho reduces to the learned prior sigma(c_rho)

    def test_lag_creates_information_gap(self):
        """Between t_chk and t_commit the model keeps using the OLD snapshot."""
        store = SnapshotStore()
        store.add(0, j=1, t_chk=10, t_commit=13, profile={},
                  profile_emb=np.full(4, 1), sim=0.9, n_consolidated=10)
        store.add(0, j=2, t_chk=20, t_commit=23, profile={},
                  profile_emb=np.full(4, 2), sim=0.2, n_consolidated=20)
        # t=21: checkpoint 2 exists (t_chk=20) but not committed (23 > 21)
        assert store.visible_snapshot(0, t=21)["j"] == 1
        assert store.visible_snapshot(0, t=23)["j"] == 2


class TestUserIsolation:
    def test_users_do_not_see_each_other(self):
        store = SnapshotStore()
        store.add(0, j=1, t_chk=10, t_commit=10, profile={},
                  profile_emb=np.zeros(4), sim=0.5, n_consolidated=10)
        assert store.visible_snapshot(1, t=100) is None
        assert store.snapshots_of(1) == []