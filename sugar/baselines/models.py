"""Baseline sequential models trained with the SAME data loader and the SAME
full-ranking evaluator as SUGAR (protocol fairness, Section 4).

  GRU4Rec  : standard GRU over item embeddings
  SASRecCE : the SUGAR Transformer backbone trained with full-catalog CE
             (Petrov & Macdonald 2023 setup -- avoids the known BCE underfit)
  BERT4Rec : bidirectional Transformer with masked-item training,
             fully trained (not the known-underfit RecBole defaults)
  UniSRecV : text-enhanced SASRec variant: item representation = trainable
             ID embedding + projection of the FROZEN text embedding e_v
             (following the UniSRec idea of text-grounded item representations)

All models expose h_seq with the same interface as SUGAR's backbone, so
FullRankingEvaluator.evaluate works unchanged.
"""
import torch
import torch.nn as nn

from sugar.online.backbone import TransformerBackbone
from sugar.online.projection import ProfileProjection


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, num_layers: int = 1,
                 max_len: int = 50, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, d, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d)
        self.gru = nn.GRU(d, d, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        lengths = (seq != self.pad_idx).sum(1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=L)
        return self.out_norm(self.dropout(h))

    def embed_items(self, idx: torch.Tensor) -> torch.Tensor:
        return self.item_emb(idx)

    @property
    def backbone(self):
        return self


class SASRecCE(nn.Module):
    """Same architecture as the SUGAR backbone; trained with full-catalog CE."""

    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2,
                 n_heads: int = 2, max_len: int = 50, dropout: float = 0.2,
                 pad_idx: int = 0):
        super().__init__()
        self.backbone = TransformerBackbone(
            n_items, d, n_blocks, n_heads, max_len, dropout, pad_idx)
        self.n_items = n_items

    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone(seq)

    def embed_items(self, idx: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_items(idx)


class BERT4Rec(nn.Module):
    """Bidirectional Transformer with masked-item (MLM) training."""

    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2,
                 n_heads: int = 2, max_len: int = 50, dropout: float = 0.2,
                 pad_idx: int = 0, mask_idx: int = None):
        super().__init__()
        # reserve index n_items as [MASK] (n_items passed here = catalog + 1)
        self.backbone = TransformerBackbone(
            n_items + 1, d, n_blocks, n_heads, max_len, dropout, pad_idx)
        self.n_items = n_items
        self.mask_idx = mask_idx if mask_idx is not None else n_items

    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone(seq)

    def embed_items(self, idx: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_items(idx)


class UniSRecV(nn.Module):
    """Text-enhanced SASRec: x_v = ID embedding + W_e e_v (frozen e_v).

    The frozen text embeddings are attached via attach_frozen_embeddings()
    before training; W_e is trainable and acts as the text-to-ID translator.
    """

    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2,
                 n_heads: int = 2, max_len: int = 50, dropout: float = 0.2,
                 pad_idx: int = 0, d_e: int = 1024):
        super().__init__()
        self.backbone = TransformerBackbone(
            n_items, d, n_blocks, n_heads, max_len, dropout, pad_idx)
        self.n_items = n_items
        self.text_proj = ProfileProjection(d_e, d)
        self._frozen = None  # (n_items, d_e) registered by the trainer

    def attach_frozen_embeddings(self, emb: torch.Tensor):
        self._frozen = emb  # detached constant

    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.backbone(seq)
        if self._frozen is None:
            return h
        # add text grounding of the SEEN items at each position
        text_part = self.text_proj(self._frozen[seq])  # (B, L, d)
        text_part = text_part * (seq != self.backbone.pad_idx).unsqueeze(-1)
        return h + text_part

    def embed_items(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.backbone.embed_items(idx)
        if self._frozen is not None:
            x = x + self.text_proj(self._frozen[idx])
        return x