"""Negative sampling for BCE training loss (Section 3.6.3: N negatives per position)."""
import numpy as np
import torch


class NegativeSampler:
    """Uniform negative sampling over the full catalog, excluding interacted items.

    Excluded items are the ones inside the sequence prefix + target, matching
    the false-negative reduction rule of the alignment loss.
    """

    def __init__(self, n_items: int, seed: int = 42):
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def sample(self, seq: list, target: int, n_neg: int) -> list:
        excluded = set(seq) | {target}
        negs = []
        while len(negs) < n_neg:
            cand = self.rng.integers(1, self.n_items, size=n_neg * 2).tolist()
            for c in cand:
                if c not in excluded and c not in negs:
                    negs.append(c)
                    if len(negs) == n_neg:
                        break
        return negs


def sample_batch_negatives(batch_seqs: torch.Tensor, batch_lens: torch.Tensor,
                           batch_targets: torch.Tensor, n_items: int,
                           n_neg: int, seed: int = 42) -> torch.Tensor:
    """Vectorized per-position sampling for a training batch.

    Returns tensor of shape (B, n_neg). Fallback numpy path is used per row
    because exclusion sets differ per user; cost is negligible.
    """
    B = batch_seqs.size(0)
    out = torch.empty(B, n_neg, dtype=torch.long)
    sampler = NegativeSampler(n_items, seed)
    for i in range(B):
        L = int(batch_lens[i])
        seq = batch_seqs[i, :L].tolist()
        out[i] = torch.tensor(sampler.sample(seq, int(batch_targets[i]), n_neg),
                              dtype=torch.long)
    return out