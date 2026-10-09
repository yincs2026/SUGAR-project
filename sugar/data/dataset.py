"""Sequential dataset with temporal leave-one-out split.

Split convention (matches Section 3.2 / evaluation protocol):
  train: interactions [0 .. n-3]
  valid: interaction n-2 (target for ranking during dev)
  test : interaction n-1

Items are integer indices produced by data/prepare.py.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    """Training samples: (sequence prefix up to t, next item)."""

    def __init__(self, sequences: dict, max_len: int, pad_idx: int = 0):
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.samples = []
        for u, seq in sequences.items():
            if len(seq) < 3:
                continue
            for t in range(1, len(seq) - 2):
                self.samples.append((u, seq[:t], seq[t]))

    def __len__(self):
        return len(self.samples)

    def _pad(self, seq):
        seq = seq[-self.max_len:]
        return [self.pad_idx] * (self.max_len - len(seq)) + list(seq)

    def __getitem__(self, idx):
        u, prefix, target = self.samples[idx]
        return {
            "user": u,
            "seq": torch.tensor(self._pad(prefix), dtype=torch.long),
            "seq_len": min(len(prefix), self.max_len),
            "target": target,
        }


class ValidTestDataset(Dataset):
    """One sample per user for validation or test (leave-one-out)."""

    def __init__(self, sequences: dict, max_len: int, split: str = "valid",
                 pad_idx: int = 0):
        assert split in ("valid", "test")
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.samples = []
        for u, seq in sequences.items():
            if len(seq) < 3:
                continue
            idx = -2 if split == "valid" else -1
            self.samples.append((u, seq[:idx], seq[idx]))

    def __len__(self):
        return len(self.samples)

    def _pad(self, seq):
        seq = seq[-self.max_len:]
        return [self.pad_idx] * (self.max_len - len(seq)) + list(seq)

    def __getitem__(self, idx):
        u, prefix, target = self.samples[idx]
        return {
            "user": u,
            "seq": torch.tensor(self._pad(prefix), dtype=torch.long),
            "seq_len": min(len(prefix), self.max_len),
            "target": target,
        }


def load_processed(dataset_name: str, processed_root: str = "outputs/processed"):
    """Load processed artifacts produced by scripts/prepare_data.py."""
    d = Path(processed_root) / dataset_name
    with open(d / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    with open(d / "items.pkl", "rb") as f:
        items = pickle.load(f)
    with open(d / "social_edges.pkl", "rb") as f:
        social_edges = pickle.load(f)
    return sequences, items, social_edges


def build_item_meta_texts(items: dict) -> list:
    """Raw text c_v for every item index (input of M1)."""
    item2idx = items["item2idx"]
    meta = items["meta"]
    n = len(item2idx)
    texts = [""] * n
    for raw_id, idx in item2idx.items():
        m = meta[raw_id]
        cats = ", ".join(m.get("categories", [])[:5])
        texts[idx] = f"{m.get('title', '')}. Categories: {cats}".strip(". ")
    return texts