"""M5 sequential backbone: self-attentive SASRec-style encoder (Section 3.6.1).

A stack of B Transformer blocks with causal masking produces h_seq at every
position t, summarizing interactions 1..t. Item ID embeddings x_v are
trainable; padding (index 0) is masked out of attention.
"""
import torch
import torch.nn as nn


class TransformerBackbone(nn.Module):
    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2,
                 n_heads: int = 2, max_len: int = 50, dropout: float = 0.2,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.d = d

        self.item_emb = nn.Embedding(n_items, d, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.out_norm = nn.LayerNorm(d)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        """Causal mask: position t attends to positions <= t only."""
        return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )

    def _key_padding(self, seq: torch.Tensor) -> torch.Tensor:
        return seq == self.pad_idx  # (B, L)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item indices (padded with pad_idx on the LEFT).

        Returns h_seq: (B, L, d) contextual representation per position.
        """
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) * (self.d ** 0.5) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        h = self.blocks(
            x,
            mask=self._causal_mask(L, seq.device),
            src_key_padding_mask=self._key_padding(seq),
        )
        return self.out_norm(h)

    def embed_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Trainable ID embedding x_v of arbitrary item indices."""
        return self.item_emb(item_indices)