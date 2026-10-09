"""Full-ranking evaluator: HR / NDCG @ {5,10,20} and MRR @ 10.

Protocol (Section 4 / configs/base.yaml): leave-one-out, ranking the FULL
catalog (no sampled negatives at evaluation time -- Krichene & Rendle 2020).
Excluded from ranking: items already in the user's sequence prefix
(to avoid trivially ranking a repeated interacted item on top).
"""
import json
from pathlib import Path

import numpy as np
import torch

PAD = 0


class FullRankingEvaluator:
    def __init__(self, n_items: int, cutoffs: tuple = (5, 10, 20),
                 mrr_cutoff: int = 10, device: str = "cpu"):
        self.n_items = n_items
        self.cutoffs = cutoffs
        self.mrr_cutoff = mrr_cutoff
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, sequences: dict, max_len: int,
                 split_targets: dict, batch_size: int = 128,
                 snapshot_store=None, neighbor_map=None, window_means=None,
                 strict: bool = True) -> dict:
        """sequences: {u: prefix}, split_targets: {u: ground-truth next item}."""
        model.eval()
        users = [u for u in split_targets if len(sequences[u]) >= 1]
        hits = {k: [] for k in self.cutoffs}
        ndcgs = {k: [] for k in self.cutoffs}
        mrrs = []
        rhos = []  # collected for E4.3 (rho vs error correlation)

        all_items = torch.arange(1, self.n_items, device=self.device)
        x_all = model.backbone.embed_items(all_items)  # (n, d)

        for i in range(0, len(users), batch_size):
            batch_users = users[i:i + batch_size]
            seqs, lens = [], []
            for u in batch_users:
                s = sequences[u][-max_len:]
                lens.append(len(s))
                seqs.append([PAD] * (max_len - len(s)) + list(s))
            seq_t = torch.tensor(seqs, dtype=torch.long, device=self.device)
            lens_t = torch.tensor(lens, device=self.device)
            users_t = torch.tensor(batch_users, device=self.device)

            h = model.forward_with_users(
                seq_t, users_t, lens_t, snapshot_store,
                neighbor_map=neighbor_map, window_means=window_means,
                strict=strict)
            # take representation at the last non-padded position
            h_last = h[torch.arange(len(batch_users), device=self.device),
                       torch.tensor(lens, device=self.device) - 1]  # (B, d)

            scores = h_last @ x_all.T  # (B, n_items - 1)

            # mask items already interacted with in the prefix
            for b, u in enumerate(batch_users):
                seen = set(sequences[u])
                for it in seen:
                    if 1 <= it < self.n_items:
                        scores[b, it - 1] = -1e9

            # reliability scores for E4.3 (only meaningful for the full model)
            if hasattr(model, "fusion") and model.fusion.mode == "full" \
                    and snapshot_store is not None:
                for b, u in enumerate(batch_users):
                    snaps = snapshot_store.snapshots_of(int(u))
                    cands = [s for s in snaps if s["t_commit"] <= lens[b] - 1]
                    if cands:
                        sim, n = cands[-1]["sim"], cands[-1]["n_consolidated"]
                        rho = model.fusion.reliability(
                            torch.tensor(sim), torch.tensor(float(n))).item()
                        rhos.append((rho, u))

            ranks = torch.argsort(scores, dim=1, descending=True)  # (B, n-1)
            for b, u in enumerate(batch_users):
                target = split_targets[u]
                rank_pos = (ranks[b] == target - 1).nonzero()
                r = int(rank_pos.item()) + 1 if rank_pos.numel() else \
                    self.n_items  # masked-out target: worst rank

                for k in self.cutoffs:
                    hits[k].append(1.0 if r <= k else 0.0)
                    ndcgs[k].append(1.0 / np.log2(r + 1) if r <= k else 0.0)
                mrrs.append(1.0 / r if r <= self.mrr_cutoff else 0.0)

        out = {}
        for k in self.cutoffs:
            out[f"HR@{k}"] = float(np.mean(hits[k]))
            out[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
        out["MRR@10"] = float(np.mean(mrrs))
        out["n_users_evaluated"] = len(users)
        return out