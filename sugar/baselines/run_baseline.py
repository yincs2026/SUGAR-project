"""Unified baseline training/evaluation loop -- same splits, same evaluator,
same seed discipline as SUGAR (fair comparison, Section 4).

  GRU4Rec  : BCE with sampled negatives (original setup)
  SASRecCE : full-catalog cross-entropy
  BERT4Rec : masked-item training, full-catalog CE on masked positions
  UniSRecV : full-catalog CE (text-enhanced)

Usage:
  python scripts/train.py --model sasrec --dataset beauty
(see scripts/train_baseline.py below)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sugar.data.dataset import SequentialDataset, ValidTestDataset
from sugar.training.evaluator import FullRankingEvaluator


def _collate(batch):
    return {
        "user": torch.tensor([b["user"] for b in batch]),
        "seq": torch.stack([b["seq"] for b in batch]),
        "seq_len": torch.tensor([b["seq_len"] for b in batch]),
        "target": torch.tensor([b["target"] for b in batch]),
    }


def train_baseline(model, cfg: dict, train_seqs: dict, valid_targets: dict,
                   device: str, tag: str, log_dir: str = "outputs/logs",
                   ckpt_dir: str = "outputs/checkpoints",
                   frozen_emb: np.ndarray = None) -> dict:
    """Unified loop; the loss differs per family, everything else is shared."""
    model = model.to(device)
    t_cfg = cfg["training"]
    max_len = cfg["model"]["max_len"]
    n_items = model.n_items

    if frozen_emb is not None and hasattr(model, "attach_frozen_embeddings"):
        model.attach_frozen_embeddings(
            torch.tensor(frozen_emb, dtype=torch.float32, device=device))

    opt = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"],
                           weight_decay=t_cfg["weight_decay"])
    ds = SequentialDataset(train_seqs, max_len)
    dl = DataLoader(ds, batch_size=t_cfg["batch_size"], shuffle=True,
                    num_workers=2, collate_fn=_collate)

    evaluator = FullRankingEvaluator(n_items,
                                     tuple(cfg["evaluation"]["cutoffs"]),
                                     device=device)
    family = cfg["model"].get("family", "bce")  # bce | ce | mlm
    n_neg = cfg["model"].get("n_negatives", 4)
    rng = np.random.default_rng(cfg["seed"])

    best_ndcg, best_epoch, best_state = -1.0, -1, None
    history = []
    log_dir, ckpt_dir = Path(log_dir), Path(ckpt_dir)

    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        for batch in dl:
            seq = batch["seq"].to(device)
            lens = batch["seq_len"].to(device)
            tgt = batch["target"].to(device)
            h = model(seq)
            idx = (lens - 1).clamp(min=0)
            h_last = h[torch.arange(h.size(0), device=device), idx]

            if family == "bce":
                x_pos = model.embed_items(tgt)
                negs = torch.randint(1, n_items, (tgt.size(0), n_neg),
                                     device=device)
                x_neg = model.embed_items(negs)
                pos_s = (h_last * x_pos).sum(-1)
                neg_s = h_last @ x_neg.transpose(0, 1)
                loss = -(torch.logsigmoid(pos_s)
                         + torch.logsigmoid(-neg_s).sum(1)).mean()
            elif family == "ce":
                scores = h_last @ model.embed_items(
                    torch.arange(1, n_items, device=device)).T  # (B, n-1)
                loss = F.cross_entropy(scores, tgt - 1)
            elif family == "mlm":
                # mask a random non-pad position, predict the original item
                B, L = seq.shape
                cand = (seq != model.backbone.pad_idx)
                r = torch.rand(seq.shape, device=device) * cand
                mask_pos = r.argmax(1)  # one masked position per row
                masked = seq.clone()
                orig = seq[torch.arange(B), mask_pos]
                masked[torch.arange(B), mask_pos] = model.mask_idx
                h_m = model(masked)
                h_pick = h_m[torch.arange(B), mask_pos]
                scores = h_pick @ model.embed_items(
                    torch.arange(1, n_items, device=device)).T
                loss = F.cross_entropy(scores, orig - 1)
            else:
                raise ValueError(family)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()); nb += 1

        # prefix for eval: training prefix per user
        metrics = evaluator.evaluate(
            model, train_seqs, max_len, valid_targets, batch_size=128)
        metrics.update({"epoch": epoch,
                        "train_loss": total / max(nb, 1),
                        "sec": round(time.time() - t0, 1)})
        history.append(metrics)
        print(f"[{tag}] epoch {epoch}: NDCG@10={metrics['NDCG@10']:.4f} "
              f"HR@10={metrics['HR@10']:.4f} loss={metrics['train_loss']:.4f}")

        if metrics["NDCG@10"] > best_ndcg:
            best_ndcg, best_epoch = metrics["NDCG@10"], epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), ckpt_dir / f"{tag}_best.pt")
        if epoch - best_epoch >= t_cfg["patience"]:
            print(f"early stop at epoch {epoch}")
            break

    with open(log_dir / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    if best_state:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_valid_ndcg10": best_ndcg}