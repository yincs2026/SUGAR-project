"""Training objectives (Section 3.6.3).

L = L_rec + lambda * L_align

L_rec : BCE with one positive and N uniformly sampled negatives per position
        (BPR variant available for the ablation appendix).
L_align: InfoNCE pulling the ID embedding x_v toward the projected frozen
        embedding W_p e_v, pushing away from a batch of other items that do
        not appear in the user's sequence (false-negative reduction).
        The frozen embedding e_v is a CONSTANT: only x_v and W_p receive
        gradients. The projection W_p is shared with the profile projection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecommendationLoss(nn.Module):
    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        assert loss_type in ("bce", "bpr")
        self.loss_type = loss_type

    def forward(self, scores: torch.Tensor, pos_idx: torch.Tensor,
                neg_idx: torch.Tensor) -> torch.Tensor:
        """scores: (B, n_items) dot products h_t^T x_v over the catalog
        (or over candidate subset). pos_idx: (B,). neg_idx: (B, N)."""
        pos = scores.gather(1, pos_idx.unsqueeze(1)).squeeze(1)       # (B,)
        neg = scores.gather(1, neg_idx)                               # (B, N)

        if self.loss_type == "bce":
            pos_loss = F.binary_cross_entropy_with_logits(
                pos, torch.ones_like(pos))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg, torch.zeros_like(neg))
            return pos_loss + neg_loss
        # BPR variant
        return -F.logsigmoid(pos.unsqueeze(1) - neg).mean()


class AlignmentLoss(nn.Module):
    """InfoNCE between collaborative and projected semantic spaces."""

    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = tau

    def forward(self, x_v: torch.Tensor, e_v_projected: torch.Tensor,
                batch_other_projected: torch.Tensor) -> torch.Tensor:
        """x_v: (B, d) ID embeddings of interacted items.
        e_v_projected: (B, d) W_p e_v for the same items (detached source).
        batch_other_projected: (B, M, d) projected descriptors of OTHER items
        not in the user's sequence."""
        e = F.normalize(e_v_projected, dim=-1)
        x = F.normalize(x_v, dim=-1)
        others = F.normalize(batch_other_projected, dim=-1)

        pos_sim = (x * e).sum(-1) / self.tau                          # (B,)
        neg_sim = torch.einsum("bd,bmd->bm", x, others) / self.tau    # (B, M)

        # log pos / (pos + sum neg)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return F.cross_entropy(logits, labels)