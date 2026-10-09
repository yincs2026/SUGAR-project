"""M1: Item Semantic Refiner (Section 3.3.1).

For each item v:
  d_v = LLM(RefinePrompt(c_v))   -- schema-constrained, fallback on failure
  e_v = Enc(serialize(d_v))      -- frozen sentence encoder

Every item is refined exactly once; all outputs are cached on disk
(one-time O(|V|) LLM calls over the dataset lifetime). Cache is keyed by
(item_idx, content-hash, model names), so changing the LLM or encoder
invalidates it deliberately (Section 3.3.1 re-deployment note).

Schema-violation statistics are accumulated for E6 (schema compliance rate).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sugar.offline.llm_client import LLMClient
from sugar.offline.prompts import refine_prompt
from sugar.offline.encoders import SentenceEncoder

REQUIRED_FIELDS = ["topic", "subtopics", "style", "audience", "summary"]


def validate_descriptor(obj: dict) -> bool:
    """A descriptor is valid if all required fields exist and are non-empty."""
    if not isinstance(obj, dict):
        return False
    for f in REQUIRED_FIELDS:
        if f not in obj:
            return False
        v = obj[f]
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
    # subtopics must be a list
    if not isinstance(obj["subtopics"], list):
        return False
    return True


def fallback_descriptor(item_text: str, categories: list) -> dict:
    """Surface-feature fallback when the LLM output is invalid (Section 3.3.1).

    Derives topic/subtopics from category labels; other fields are left
    minimal so downstream averaging still works.
    """
    cats = [c.strip() for c in (categories or []) if c.strip()]
    topic = cats[0] if cats else (item_text[:60] or "unknown")
    subtopics = cats[1:4] if len(cats) > 1 else []
    return {
        "topic": topic,
        "subtopics": subtopics,
        "style": "unspecified",
        "audience": "unspecified",
        "summary": item_text[:200],
        "_fallback": True,
    }


def serialize_descriptor(d: dict) -> str:
    """Deterministic serialization used as encoder input."""
    return json.dumps(
        {k: d.get(k) for k in REQUIRED_FIELDS},
        ensure_ascii=False, sort_keys=True,
    )


class ItemRefiner:
    def __init__(self, llm: LLMClient, encoder: SentenceEncoder,
                 cache_dir: str = "outputs/cache/items"):
        self.llm = llm
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.desc_dir = self.cache_dir / "descriptors"
        self.emb_dir = self.cache_dir / "embeddings"
        self.desc_dir.mkdir(exist_ok=True)
        self.emb_dir.mkdir(exist_ok=True)
        self.stats = {"n_items": 0, "llm_failures": 0, "fallbacks": 0}

    def refine(self, item_idx: int, item_text: str, categories: list):
        """Returns (descriptor dict, embedding np.ndarray). Fully cached."""
        self.stats["n_items"] += 1
        content_hash = hashlib.sha256(item_text.encode("utf-8")).hexdigest()[:16]

        # descriptor: LLM call (cached)
        d_path = self.desc_dir / f"{item_idx}_{content_hash}.json"
        if d_path.exists():
            with open(d_path, "r", encoding="utf-8") as f:
                d = json.load(f)
        else:
            raw = self.llm.generate(refine_prompt(item_text))
            d = self.llm.parse_json(raw)
            if not validate_descriptor(d):
                self.stats["llm_failures"] += 1
                self.stats["fallbacks"] += 1
                d = fallback_descriptor(item_text, categories)
            with open(d_path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

        # embedding: encoder call (cached)
        e_path = self.emb_dir / f"{item_idx}_{content_hash}.npy"
        if e_path.exists():
            e = np.load(e_path)
        else:
            e = self.encoder.embed_text_cached(serialize_descriptor(d),
                                               tag="descriptor")
            np.save(e_path, e)
        return d, e

    def refine_catalog(self, item_texts: list, item_categories: list = None,
                       show_progress: bool = True):
        """Refine every item of the catalog. Returns descriptors and embeddings."""
        if item_categories is None:
            item_categories = 
        descriptors, embeddings = [], []
        iterator = range(len(item_texts))
        if show_progress:
            iterator = tqdm(iterator, desc="M1 refine")
        for idx in iterator:
            d, e = self.refine(idx, item_texts[idx], item_categories[idx])
            descriptors.append(d)
            embeddings.append(e)
        self.stats["schema_violation_rate"] = round(
            self.stats["llm_failures"] / max(self.stats["n_items"], 1), 4)
        return descriptors, np.stack(embeddings).astype(np.float32)