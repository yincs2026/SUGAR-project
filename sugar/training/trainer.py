"""Training loop -- Algorithm 2, lines 1-16.

Per epoch: iterate training samples (user, prefix, target); forward via
SUGARModel.forward_with_users; BCE over the sampled-candidate set; alignment
loss on interacted items; early stopping on validation NDCG@10.

The trainer precomputes once per training run:
  - recent-window means e_bar(R_u) from frozen item embeddings (M3 inputs)
  - neighbor profile cache from the snapshot store (M4 inputs)
Both are FIXED during training (piecewise-constant), matching Section 3.5.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sugar.data.dataset import SequentialDataset, ValidTestDataset
from sugar.data.sampler import sample_batch_negatives
from sugar.training.evaluator import FullRankingEvaluator

def score_negative_candidates(
    h_last: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-user scores for sampled negative items.

    Args:
        h_last: (B, D) user representations.
        neg_emb: (B, N, D) negative-item embeddings.

    Returns:
        scores_neg: (B, N)
    """
    if h_last.ndim != 2:
        raise ValueError(
            f"h_last must have shape (B, D), got {tuple(h_last.shape)}"
        )

    if neg_emb.ndim != 3:
        raise ValueError(
            f"neg_emb must have shape (B, N, D), got {tuple(neg_emb.shape)}"
        )

    if h_last.shape[0] != neg_emb.shape[0]:
        raise ValueError("Batch dimensions of h_last and neg_emb must match.")

    if h_last.shape[1] != neg_emb.shape[2]:
        raise ValueError("Embedding dimensions of h_last and neg_emb must match.")

    return torch.einsum("bd,bnd->bn", h_last, neg_emb)

def build_window_means(sequences: dict, item_embeddings: np.ndarray,
                       window_size_k: int) -> dict:
    """e_bar(R_u): mean of frozen item embeddings over the last k interactions."""
    out = {}
    for u, seq in sequences.items():
        w = seq[-window_size_k:]
        if w:
            out[u] = item_embeddings[w].mean(axis=0)
    return out


def build_neighbor_map(social_graph, snapshot_store, window_means: dict,
                       d_e: int, max_neighbors: int) -> dict:
    """{u: [profile_emb of up to top-L followees]} -- cached frozen vectors.

    Affinity is precomputed here with e_bar(R_u); the final top-L selection
    and renormalization still happen inside SocialGate.aggregate.
    """
    import numpy as np
    out = {}
    for u in window_means:
        nbs = social_graph.neighbors(u)
        if not nbs:
            continue
        wm = window_means[u]
        scored = []
        for w in nbs:
            snaps = snapshot_store.snapshots_of(w)
            cands = [s for s in snaps if s["profile_emb"] is not None]
            if not cands:
                continue
            p_w = np.asarray(cands[-1]["profile_emb"], dtype=np.float32)
            cos = float(np.dot(p_w, wm) /
                        (np.linalg.norm(p_w) * np.linalg.norm(wm) + 1e-9))
            scored.append((cos, p_w))
        scored.sort(key=lambda t: -t[0])
        out[u] = [p for _, p in scored[:max_neighbors]]
    return out


class Trainer:
    def __init__(self, model, cfg: dict, sequences: dict, snapshot_store=None,
                 social_graph=None, item_embeddings: np.ndarray = None,
                 device: str = "cuda", log_dir: str = "outputs/logs",
                 checkpoint_dir: str = "outputs/checkpoints",
                 strict_snapshots: bool = True, tag: str = "sugar"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.sequences = sequences
        self.snapshot_store = snapshot_store
        self.strict = strict_snapshots
        self.tag = tag

        m = cfg["model"]
        self.max_len = m["max_len"]
        self.n_negatives = m["n_negatives"]
        self.lambda_align = m["lambda_align"]

        # side-info caches (fixed during training)
        self.window_means = None
        self.neighbor_map = None
        if snapshot_store is not None and item_embeddings is not None:
            self.window_means = build_window_means(
                sequences, item_embeddings, cfg["offline"]["window_size_k"])
            if social_graph is not None:
                self.neighbor_map = build_neighbor_map(
                    social_graph, snapshot_store, self.window_means,
                    m["de"], m["n_social_neighbors_L"])

        self.evaluator = FullRankingEvaluator(
            model.n_items if hasattr(model, "n_items") else None,
            cutoffs=tuple(cfg["evaluation"]["cutoffs"]),
            device=device)

        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        from sugar.online.losses import RecommendationLoss, AlignmentLoss
        self.loss_rec = RecommendationLoss("bce")
        self.loss_align = AlignmentLoss(m["tau_align"])
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                                    weight_decay=cfg["training"]["weight_decay"])

    def _candidate_set(self, n_items: int, batch: dict) -> torch.Tensor:
        """Return N sampled negatives per user on the batch device."""
        negs = sample_batch_negatives(
            batch["seq"],
            batch["seq_len"].clone(),
            batch["target"].detach().cpu(),
            n_items,
            self.n_negatives,
        )
        return negs.to(device=batch["seq"].device)

    def train(self, valid_targets: dict, epochs: int = None,
              patience: int = None) -> dict:
        t_cfg = self.cfg["training"]
        epochs = epochs or t_cfg["epochs"]
        patience = patience or t_cfg["patience"]

        ds = SequentialDataset(self.sequences, self.max_len)
        dl = DataLoader(ds, batch_size=t_cfg["batch_size"], shuffle=True,
                        num_workers=2, collate_fn=_collate)

        n_items = self.model.backbone.item_emb.num_embeddings
        best_ndcg, best_epoch, best_state = -1.0, -1, None
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            t0, total_loss, n_batches = time.time(), 0.0, 0
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                users = batch["user"]
                h = self.model.forward_with_users(
                    batch["seq"], users, batch["seq_len"], self.snapshot_store,
                    neighbor_map=self.neighbor_map,
                    window_means=self.window_means, strict=self.strict)

                # last non-padded position per row
                idx = (batch["seq_len"] - 1).clamp(min=0)
                h_last = h[torch.arange(h.size(0), device=self.device), idx]

                cand = torch.arange(1, n_items, device=self.device)
                # sample N negatives + include the positive: score subset
                
                # Sample N negatives for each user.
                negs = self._candidate_set(n_items, batch)  # (B, N)

                pos_emb = self.model.backbone.embed_items(batch["target"])
                neg_emb = self.model.backbone.embed_items(negs)  # (B, N, D)

                scores_pos = (h_last * pos_emb).sum(dim=-1)  # (B,)
                scores_neg = score_negative_candidates(
                    h_last, neg_emb
                )  # (B, N)

                # Pointwise logistic recommendation loss:
                # one positive and N negative candidates per user.
                loss_rec = -(
                    torch.nn.functional.logsigmoid(scores_pos)
                    + torch.nn.functional.logsigmoid(-scores_neg).sum(dim=1)
                ).mean()


                # alignment loss on the target items
                x_v = self.model.backbone.embed_items(batch["target"])
                # projected frozen embeddings come from cached e_v (detached);
                # for simplicity the trainer uses the projection of the model
                # on cached item embeddings when provided:
                e_v = self._projected_frozen(batch["target"])
                others = self._projected_frozen(
                    torch.randint(1, n_items, (x_v.size(0), 8),
                                  device=self.device))
                loss_align = self.loss_align(x_v, e_v, others) \
                    if self.lambda_align > 0 else torch.tensor(0.0)

                loss = loss_rec + self.lambda_align * loss_align
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item()); n_batches += 1

            metrics = self.evaluator.evaluate(
                self.model, self.sequences, self.max_len, valid_targets,
                snapshot_store=self.snapshot_store,
                neighbor_map=self.neighbor_map,
                window_means=self.window_means, strict=self.strict)
            metrics["epoch"] = epoch
            metrics["train_loss"] = total_loss / max(n_batches, 1)
            metrics["sec"] = round(time.time() - t0, 1)
            history.append(metrics)
            print(f"[{self.tag}] epoch {epoch}: NDCG@10={metrics['NDCG@10']:.4f} "
                  f"HR@10={metrics['HR@10']:.4f} loss={metrics['train_loss']:.4f}")

            if metrics["NDCG@10"] > best_ndcg:
                best_ndcg = metrics["NDCG@10"]; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                torch.save(self.model.state_dict(),
                           self.ckpt_dir / f"{self.tag}_best.pt")
            if epoch - best_epoch >= patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

        with open(self.log_dir / f"{self.tag}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "best_valid_ndcg10": best_ndcg}

    def _projected_frozen(self, item_indices: torch.Tensor) -> torch.Tensor:
        """W_p e_v on cached frozen embeddings; zeros if not available."""
        if getattr(self, "_frozen_emb", None) is None:
            return torch.zeros(item_indices.size(0),
                               self.model.projection.W_p.out_features,
                               device=self.device)
        e = self._frozen_emb[item_indices]
        return self.model.projection(e)

    def attach_frozen_embeddings(self, item_embeddings: np.ndarray):
        """Trainer-side cache of frozen e_v (torch) for the alignment loss."""
        self._frozen_emb = torch.tensor(item_embeddings, dtype=torch.float32,
                                        device=self.device)
        self._frozen_emb.requires_grad_(False)


def _collate(batch):
    out = {}
    for k in ("user", "seq", "target"):
        out[k] = torch.tensor([b[k] for b in batch]) if k != "seq" else \
            torch.stack([b[k] for b in batch])
    out["seq_len"] = torch.tensor([b["seq_len"] for b in batch])
    return out