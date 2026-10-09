"""Leakage tests: the temporal-consistency contract of Section 3.2.

Core invariant: for a prediction at position t, NO input may come from an
interaction at or after t -- of the user or of any neighbor. The snapshot
protocol enforces this via commit times (t_commit = t_chk + Lambda).
"""
import numpy as np
import pytest

from sugar.offline.snapshot_store import SnapshotStore


def make_store(lag: int, m_chk: int = 10, n_checkpoints: int = 5) -> SnapshotStore:
    store = SnapshotStore()
    for j in range(1, n_checkpoints + 1):
        t_chk = j * m_chk
        store.add(user_idx=0, j=j, t_chk=t_chk, t_commit=t_chk + lag,
                  profile={"dominant_interests": []},
                  profile_emb=np.ones(4) * j, sim=0.5, n_consolidated=t_chk)
    return store


class TestCommitProtocol:
    def test_no_snapshot_visible_before_first_commit(self):
        store = make_store(lag=5)  # first commit at t=15
        assert store.visible_snapshot(0, t=14) is None
        assert store.visible_snapshot(0, t=15) is not None

    def test_visibility_exactly_at_commit(self):
        store = make_store(lag=5)
        # snapshot j=1 committed at 15 must be visible at t=15 (<=, not <)
        snap = store.visible_snapshot(0, t=15)
        assert snap["j"] == 1

    def test_lag_delays_visibility(self):
        store_lag5 = make_store(lag=5)
        store_lag0 = make_store(lag=0)
        # checkpoint j=2 at t=20: visible with lag 0, invisible with lag 5 at t=22
        assert store_lag0.visible_snapshot(0, t=22)["j"] == 2
        assert store_lag5.visible_snapshot(0, t=22)["j"] == 1

    def test_latest_visible_is_most_recent_committed(self):
        store = make_store(lag=5, n_checkpoints=5)
        # at t=32: commits at 15,20,25,30,35 -> latest visible is j=4 (commit 30)
        snap = store.visible_snapshot(0, t=32)
        assert snap["j"] == 4

    def test_never_returns_future_snapshot(self):
        store = make_store(lag=5)
        for t in range(0, 60):
            snap = store.visible_snapshot(0, t=t)
            if snap is not None:
                assert snap["t_commit"] <= t, \
                    f"LEAKAGE: snapshot committed at {snap['t_commit']} used at t={t}"

    def test_no_interaction_from_target_or_later_in_profile(self):
        """Profile consolidation must never include interactions >= prediction t.

        Simulated: user profile built from first t_chk interactions only;
        the snapshot at commit time must not reflect later interactions.
        We check n_consolidated <= t_chk of the snapshot's own checkpoint.
        """
        store = make_store(lag=5)
        for t in range(0, 60):
            snap = store.visible_snapshot(0, t=t)
            if snap is not None:
                assert snap["n_consolidated"] <= snap["t_chk"]


class TestAblationA7:
    def test_latest_uncommitted_ignores_protocol(self):
        """A7 (leakage ablation) intentionally bypasses the commit check."""
        store = make_store(lag=50)  # nothing committed yet at t<55
        assert store.visible_snapshot(0, t=10) is None
        # but the ablation hook returns the newest snapshot regardless
        assert store.latest_uncommitted(0, t=10)["j"] == 5


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        store = make_store(lag=5)
        path = tmp_path / "snap.pkl"
        store.save(str(path))
        loaded = SnapshotStore.load(str(path))
        assert loaded.n_users == 1
        assert loaded.n_snapshots == 5
        s1 = store.visible_snapshot(0, t=30)
        s2 = loaded.visible_snapshot(0, t=30)
        assert s1["j"] == s2["j"]
        np.testing.assert_array_equal(s1["profile_emb"], s2["profile_emb"])