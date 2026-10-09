"""Offline pipeline: full Algorithm 1 (M1 cache + M2 + M3 -> snapshot store).

For each user, checkpoints occur every m_chk interactions; at checkpoint j the
pipeline sees only the first t_j = j * m_chk interactions. Snapshots become
visible to the online stage only after the commit lag Lambda (Section 3.2).
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sugar.offline.profile_builder import ProfileBuilder
from sugar.offline.drift_detector import (DriftStateMachine,
                                          recent_window_mean, freshness_score)
from sugar.offline.snapshot_store import SnapshotStore


class OfflinePipeline:
    def __init__(self, builder: ProfileBuilder, item_embeddings: np.ndarray,
                 m_chk: int, commit_lag_lambda: int, threshold_tau: float,
                 cooldown_nmin: int, personalized_threshold: bool = False,
                 xi: float = 1.0, enable_drift: bool = True):
        self.builder = builder
        self.emb = item_embeddings          # (n_items, d_e), frozen
        self.m_chk = m_chk
        self.lag = commit_lag_lambda
        self.tau = threshold_tau
        self.n_min = cooldown_nmin
        self.personalized = personalized_threshold
        self.xi = xi
        self.stats = {"revisions_fired": 0, "checkpoints": 0,
                      "consolidations_fired": 0}
        self.enable_drift = enable_drift

    def run(self, sequences: dict, item_descriptors_by_idx: list,
            out_path: str, show_progress: bool = True) -> SnapshotStore:
        """sequences: {user_idx: [item_idx,...]} from the processed artifacts.

        item_descriptors_by_idx: descriptor dict per item index (from M1 cache).
        """
        store = SnapshotStore()
        users = list(sequences.items())
        iterator = tqdm(users, desc="M2/M3 checkpoints") if show_progress else users

        for u, seq in iterator:
            self._run_user(u, seq, item_descriptors_by_idx, store)

        store.save(out_path)
        return store

    def _run_user(self, u: int, seq: list, descriptors: list,
                  store: SnapshotStore):
        consolidated: set = set()
        profile = None
        profile_emb = None
        machine = DriftStateMachine(self.tau, self.n_min,
                                    self.personalized, self.xi)
        n = len(seq)
        j = 0

        for t_chk in range(self.m_chk, n + 1, self.m_chk):
            j += 1
            self.stats["checkpoints"] += 1
            seen = seq[:t_chk]
            window = seen[-self.builder.k:]

            # ---- M2: initial construction ----------------------------------
            if profile is None and t_chk >= self.builder.n0:
                descs = [descriptors[v] for v in seen]
                profile = self.builder.build(descs)
                consolidated = set(seen)
                profile_emb = self._embed(profile)

            # ---- M2: consolidation edit ------------------------------------
            elif profile is not None:
                delta = self.builder.compute_delta(seq[:t_chk], consolidated,
                                                   window)
                if len(delta) >= self.builder.m_edit:
                    new_profile = self.builder.consolidate(
                        profile, [descriptors[v] for v in delta])
                    if new_profile is not profile:
                        profile = new_profile
                        profile_emb = self._embed(profile)
                    consolidated.update(delta)
                    self.stats["consolidations_fired"] += 1

                # ---- M3: drift detection -----------------------------------
                w_mean = recent_window_mean(self.emb, window)
                if profile_emb is not None and w_mean is not None:
                    sim = freshness_score(profile_emb, w_mean)
                    state = "Idle"
                    if self.enable_drift:
                        state = machine.update(sim, t_chk)    
                        if state == "Revising":
                            new_profile = self.builder.revise(
                                profile, [descriptors[v] for v in window])
                            if new_profile is not profile:
                                profile = new_profile
                                profile_emb = self._embed(profile)
                            consolidated.update(window)
                            machine.commit_revision(t_chk)
                            # rescore against the revised profile
                            sim = (freshness_score(profile_emb, w_mean)
                                if profile_emb is not None and w_mean is not None
                                else 0.0)
                            self.stats["revisions_fired"] += 1
                else:
                    sim = 0.0  # no profile yet / empty window
            else:
                sim = 0.0

            # ---- snapshot with commit lag -----------------------------------
            store.add(user_idx=u, j=j, t_chk=t_chk,
                      t_commit=t_chk + self.lag,
                      profile=profile or {},
                      profile_emb=profile_emb,
                      sim=float(sim),
                      n_consolidated=len(consolidated))

    def _embed(self, profile: dict):
        """p_u = normalized weighted average of Enc(phi_i) (Section 3.4)."""
        interests, weights = ProfileBuilder.extract_interests(profile)
        if not interests:
            return None
        return self.builder.encoder.profile_embedding(interests, weights)