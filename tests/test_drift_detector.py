"""Unit tests for M3: state machine, two-strike policy, cooldown, thresholds."""
import numpy as np
import pytest

from sugar.offline.drift_detector import (DriftStateMachine,
                                          recent_window_mean, freshness_score)


class TestTwoStrikePolicy:
    def test_no_violation_stays_idle(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        assert m.update(0.8, 10) == "Idle"
        assert m.update(0.9, 20) == "Idle"

    def test_single_violation_suspect_only(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        assert m.update(0.3, 10) == "Suspect"
        assert m.update(0.9, 20) == "Idle"  # recovered, no false alarm

    def test_two_consecutive_violations_fire(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        assert m.update(0.3, 10) == "Suspect"
        assert m.update(0.2, 20) == "Revising"
        m.commit_revision(20)
        assert m.state == "Cooldown"

    def test_detection_delay_is_two_checkpoints(self):
        """Design minimum detection delay: two checkpoints (Section 3.4)."""
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        fired_at = None
        for j, sim in enumerate([0.3, 0.25, 0.2], start=1):
            state = m.update(sim, j * 10)
            if state == "Revising":
                fired_at = j
                break
        assert fired_at == 2


class TestCooldown:
    def test_cooldown_blocks_trigger(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        m.update(0.3, 10)
        m.update(0.2, 20)
        m.commit_revision(20)
        # deep violation inside cooldown window: must NOT re-fire
        assert m.update(0.1, 30) == "Cooldown"
        assert m.update(0.1, 39) == "Cooldown"

    def test_cooldown_expires_after_nmin(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        m.update(0.3, 10)
        m.update(0.2, 20)
        m.commit_revision(20)
        assert m.update(0.1, 40) == "Suspect"  # 40-20 >= 20: cooldown over
        assert m.update(0.1, 50) == "Revising"

    def test_no_revision_within_cooldown_even_if_score_recovers(self):
        m = DriftStateMachine(threshold=0.5, cooldown_nmin=20)
        m.update(0.3, 10)
        m.update(0.2, 20)
        m.commit_revision(20)
        assert m.update(0.9, 30) == "Cooldown"  # recovery also gated


class TestThresholds:
    def test_global_threshold_bootstrap(self):
        """Personalized tau falls back to global until 3 past scores exist."""
        m = DriftStateMachine(threshold=0.4, cooldown_nmin=5,
                              personalized=True, xi=1.0)
        m.update(0.9, 10)
        assert m.effective_threshold == 0.4  # only 1 sample -> global tau
        m.update(0.8, 20)
        assert m.effective_threshold == 0.4  # 2 samples -> global tau
        m.update(0.7, 30)
        assert m.effective_threshold < 0.4   # now mu - xi*sigma

    def test_personalized_threshold_less_sensitive_for_volatile_users(self):
        """Volatile users get a LOWER (less sensitive) threshold (Section 3.4)."""
        m_volatile = DriftStateMachine(threshold=0.5, cooldown_nmin=5,
                                       personalized=True, xi=1.0)
        for i, s in enumerate([0.9, 0.2, 0.9, 0.2, 0.9]):
            m_volatile.update(s, (i + 1) * 10)
        m_stable = DriftStateMachine(threshold=0.5, cooldown_nmin=5,
                                     personalized=True, xi=1.0)
        for i, s in enumerate([0.8, 0.75, 0.85, 0.78, 0.82]):
            m_stable.update(s, (i + 1) * 10)
        assert m_volatile.effective_threshold < m_stable.effective_threshold


class TestFreshnessScore:
    def test_perfect_alignment(self):
        emb = np.eye(4, dtype=np.float32)
        p = emb[0]
        assert abs(freshness_score(p, recent_window_mean(emb, [0])) - 1.0) < 1e-6

    def test_total_misalignment(self):
        emb = np.eye(4, dtype=np.float32)
        p = emb[0]
        assert abs(freshness_score(p, recent_window_mean(emb, [1]))) < 1e-6

    def test_zero_vector_guard(self):
        assert freshness_score(np.zeros(4), np.ones(4)) == 0.0
        assert freshness_score(np.ones(4), np.zeros(4)) == 0.0

    def test_empty_window(self):
        emb = np.ones((4, 4), dtype=np.float32)
        assert recent_window_mean(emb, []) is None

    def test_score_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            p = rng.normal(size=8)
            w = rng.normal(size=8)
            s = freshness_score(p, w)
            assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6