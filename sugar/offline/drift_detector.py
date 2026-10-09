"""M3: Interest Drift Detector (Section 3.4).

- sim(u, j) = cos(p_u, mean(e_v : v in R_u))          -- freshness score
- four-state machine per user: Idle / Suspect / Revising / Cooldown
- two-strike policy: revision fires on two consecutive violations
- cooldown: no trigger until n_min new interactions after a revision commit
- threshold: global tau, or personalized tau_u = mu_u - xi * sigma_u

Note: the drift score is computed on the RAW profile embedding p_u, never on
the socially-grounded p~_u (Section 3.4), so social blending cannot mask drift.
"""
import numpy as np


class DriftStateMachine:
    STATES = ("Idle", "Suspect", "Revising", "Cooldown")

    def __init__(self, threshold: float, cooldown_nmin: int,
                 personalized: bool = False, xi: float = 1.0):
        self.tau = threshold
        self.n_min = cooldown_nmin
        self.personalized = personalized
        self.xi = xi

        self.state = "Idle"
        self.hist = []            # past drift scores (for personalized tau)
        self.last_commit_pos = 0  # interaction count at last revision commit

    @property
    def effective_threshold(self) -> float:
        """tau_u = mu_u - xi * sigma_u over the user's own past scores;
        falls back to global tau until at least 3 scores exist (bootstrap)."""
        if not self.personalized or len(self.hist) < 3:
            return self.tau
        a = np.asarray(self.hist, dtype=np.float32)
        return float(a.mean() - self.xi * a.std())

    def update(self, sim: float, t_now: int) -> str:
        """Feed one checkpoint score; returns the new state.

        sim: cosine between raw profile embedding and recent-window mean.
        t_now: interaction count at this checkpoint (for cooldown gating).
        """
        self.hist.append(float(sim))
        tau = self.effective_threshold
        violation = sim < tau

        # cooldown gate: no trigger until n_min new interactions accumulated
        if self.state == "Cooldown":
            if t_now - self.last_commit_pos < self.n_min:
                return self.state          # stay in Cooldown
            self.state = "Idle"            # cooldown expired

        if self.state == "Idle":
            self.state = "Suspect" if violation else "Idle"
        elif self.state == "Suspect":
            self.state = "Revising" if violation else "Idle"
        elif self.state == "Revising":
            # caller performs the revision and then calls commit_revision()
            return self.state
        return self.state

    def commit_revision(self, t_now: int):
        """Called by the pipeline after the LLM revision has been applied."""
        self.state = "Cooldown"
        self.last_commit_pos = t_now

    def force_reset(self):
        """A6 ablation: disable two-strike and cooldown (trigger on any dip)."""
        self.state = "Idle"
        self.last_commit_pos = -10**9


def recent_window_mean(embeddings: np.ndarray, window: list) -> np.ndarray | None:
    """e_bar(R_u) = mean of frozen item embeddings over the window (Section 3.4)."""
    if not window:
        return None
    return embeddings[window].mean(axis=0)


def freshness_score(profile_emb: np.ndarray, window_mean: np.ndarray) -> float:
    """sim(u, j) = cos(p_u, e_bar(R_u)) in [-1, 1]."""
    pn, wn = np.linalg.norm(profile_emb), np.linalg.norm(window_mean)
    if pn == 0 or wn == 0:
        return 0.0
    return float(np.dot(profile_emb, window_mean) / (pn * wn))