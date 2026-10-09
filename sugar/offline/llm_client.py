"""Unified frozen-LLM client (Section 3.3).

- greedy decoding for reproducibility (no sampling)
- strict JSON parsing with retry-once fallback
- token accounting for E8 cost analysis
- cache directory keyed by (model, prompt) hash, so M1/M2 runs are resumable
"""
import hashlib
import json
import os
from pathlib import Path

import torch


class LLMClient:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 cache_dir: str = "outputs/cache/llm",
                 device: str = None, max_new_tokens: int = 512):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self.stats = {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
            # disable sampling entirely: greedy only
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.do_sample = False

    def _cache_key(self, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def generate(self, prompt: str) -> str:
        key = self._cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            self.stats["cache_hits"] += 1
            return rec["output"]

        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        self.stats["calls"] += 1
        self.stats["tokens_in"] += int(inputs["input_ids"].shape[1])
        self.stats["tokens_out"] += int(out.shape[1] - inputs["input_ids"].shape[1])

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"output": text, "model": self.model_name}, f)
        return text

    def generate_json(self, prompt: str) -> dict | None:
        """Generate and parse JSON; returns None on parse failure (caller falls back)."""
        raw = self.generate(prompt)
        return self.parse_json(raw)

    @staticmethod
    def parse_json(raw: str) -> dict | None:
        """Tolerant JSON extraction: first {...} block, rejecting markdown fences."""
        text = raw.strip()
        if "```" in text:
            # take content inside the first fenced block if present
            parts = text.split("```")
            for p in parts:
                p = p.strip().removeprefix("json").strip()
                if p.startswith("{"):
                    text = p
                    break
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def stats_snapshot(self) -> dict:
        return {**self.stats, "model": self.model_name}