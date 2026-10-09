"""Frozen sentence encoder Enc(.) (Sections 3.3.1 and 3.4).

Default: BAAI/bge-m3 (dim 1024). Provides:
  embed_texts(texts) -> (n, d_e) L2-normalized
  profile_embedding(interests, weights) -> normalized weighted average of
    per-interest embeddings, exactly the p_u construction of Section 3.4
"""
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class SentenceEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None,
                 batch_size: int = 64, cache_dir: str = "outputs/cache/enc"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        embs = self._model.encode(texts, batch_size=self.batch_size,
                                  show_progress_bar=True,
                                  normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def embed_text_cached(self, text: str, tag: str = "") -> np.ndarray:
        """Single-text embedding with on-disk cache keyed by (model, tag, text)."""
        key = hashlib.sha256(f"{self.model_name}|{tag}|{text}".encode()).hexdigest()
        f = self.cache_dir / f"{key}.npy"
        if f.exists():
            return np.load(f)
        vec = self.embed_texts([text])[0]
        np.save(f, vec)
        return vec

    def profile_embedding(self, interests: List[str],
                          weights: List[float]) -> Optional[np.ndarray]:
        """p_u = normalized weighted average of Enc(phi_i), Section 3.4.

        interests: phi_i strings from the profile JSON (dominant_interests).
        weights:   w_i float weights aligned with interests.
        Returns None if no interests (caller keeps previous embedding).
        """
        if not interests:
            return None
        embs = self.embed_texts(list(interests))
        w = np.asarray(weights, dtype=np.float32)
        if w.shape[0] != embs.shape[0]:
            w = np.ones(embs.shape[0], dtype=np.float32)
        v = (w[:, None] * embs).sum(axis=0)
        norm = np.linalg.norm(v)
        if norm == 0:
            return None
        return v / norm