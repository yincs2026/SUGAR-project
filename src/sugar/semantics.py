"""Frozen, cache-keyed semantic production. Mock mode is only for infrastructure tests."""
from collections import Counter
import json
import re
import time
import urllib.request
from pathlib import Path
import numpy as np
from .utils import canonical, digest, load_json, save_json

ITEM_PROMPT = ('You are a content analyzer for a social media platform. Given the item content below, '
               'output a JSON object with fields: (1) topic: the dominant topic as a short phrase; '
               '(2) subtopics: up to three secondary themes; (3) style: the content type '
               '(news, opinion, tutorial, entertainment, promotion); (4) audience: the intended audience; '
               '(5) summary: a one-sentence neutral summary. Content: {content}')
BUILD_PROMPT = ('Build a text-grounded user profile from topic clusters and their relative frequencies. '
                'Return dominant_interests as a ranked list of objects with topic and positive weight; '
                'emerging_interests, content_preferences and excluded_topics as lists of strings. '
                'Do not infer avoidance from the absence of positive feedback. Evidence: {evidence}')
EDIT_PROMPT = ('Edit only affected fields of this user profile. Reinforce recurring topics, promote '
               'emerging interests when supported, demote unsupported interests, and preserve other fields. '
               'Do not infer avoidance from missing positive feedback. Profile: {profile}\nNew evidence: {evidence}')
REVISE_PROMPT = ('Reconcile the conflict between this profile and recent behavior. Strengthen supported '
                 'emerging topics, demote contradicted interests, preserve uncontradicted long-term interests. '
                 'Do not invent excluded topics. Profile: {profile}\nRecent evidence: {evidence}')
STRING = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "items": STRING}
ITEM_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["topic", "subtopics", "style", "audience", "summary"],
               "properties": {"topic": STRING, "subtopics": {**STRINGS, "maxItems": 3},
                              "style": {"enum": ["news", "opinion", "tutorial", "entertainment", "promotion"]},
                              "audience": STRING, "summary": STRING}}
PROFILE_SCHEMA = {"type": "object", "additionalProperties": False,
                  "required": ["dominant_interests", "emerging_interests", "content_preferences", "excluded_topics"],
                  "properties": {"dominant_interests": {"type": "array", "minItems": 1, "items": {
                      "type": "object", "additionalProperties": False, "required": ["topic", "weight"],
                      "properties": {"topic": STRING, "weight": {"type": "number", "exclusiveMinimum": 0}}}},
                                 **{k: STRINGS for k in ["emerging_interests", "content_preferences", "excluded_topics"]}}}


def validate(obj, schema):
    """Small strict validator also works in offline tests without jsonschema installed."""
    if schema.get("type") == "object":
        if not isinstance(obj, dict) or set(obj) != set(schema["required"]):
            raise ValueError("Invalid schema keys")
        for k, spec in schema["properties"].items():
            validate(obj[k], spec)
    elif schema.get("type") == "array":
        if not isinstance(obj, list) or len(obj) < schema.get("minItems", 0) or len(obj) > schema.get("maxItems", 10**9):
            raise ValueError("Invalid array")
        for x in obj:
            validate(x, schema["items"])
    elif schema.get("type") == "string":
        if not isinstance(obj, str) or not obj.strip():
            raise ValueError("Empty text")
    elif schema.get("type") == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)) or not np.isfinite(obj) or obj <= 0:
            raise ValueError("Invalid weight")
    if "enum" in schema and obj not in schema["enum"]:
        raise ValueError("Invalid style")


def topic_clusters(descriptors):
    groups = {}
    for d in descriptors:
        topic = d["topic"].strip()
        if topic not in groups:
            groups[topic] = {"topic": topic, "count": 0, "styles": Counter(), "examples": []}
        group = groups[topic]
        group["count"] += 1
        group["styles"][d["style"]] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(d["summary"])
    total = max(1, len(descriptors))
    return [{**g, "frequency": g["count"] / total, "styles": dict(g["styles"])}
            for g in sorted(groups.values(), key=lambda g: (-g["count"], g["topic"]))]


def item_fallback(text, entities):
    """Use source hashtags and actual NER outputs; never invented categories."""
    tags = re.findall(r"#[\w]+", text)
    topics = list(dict.fromkeys(tags + [str(entity).strip() for entity in entities if str(entity).strip()]))
    return {"topic": topics[0] if topics else "unknown", "subtopics": topics[1:4],
            "style": "opinion", "audience": "general",
            "summary": " ".join(text.split()[:128]) or "Unspecified content"}


class SemanticBackend:
    def __init__(self, config, cache):
        self.c = config
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mock = config["semantic_backend"] == "mock"
        if not self.mock:
            for key in ["llm_revision", "encoder_revision"]:
                if not re.fullmatch(r"[0-9a-f]{40}", str(config[key])):
                    raise ValueError(f"{key} must be an immutable 40-character Hugging Face commit; run resolve-models")
        self.identity = {k: config[k] for k in ["semantic_backend", "llm_model", "llm_revision", "encoder_model", "encoder_revision", "llm_seed", "llm_max_tokens", "encoder_max_len", "mock_dim"]}
        if not self.mock:
            server = load_json(config["llm_server_provenance"])
            if server["model"] != config["llm_model"] or server["revision"] != config["llm_revision"]:
                raise ValueError("Server launch provenance does not match requested Qwen model/revision")
            if not server["versions"]["vllm"].startswith("0.5."):
                raise ValueError("This implementation sends the vLLM 0.5 guided_json API; use a 0.5 server")
            self.identity["server_provenance"] = server
        self.identity["fallback_policy"] = {"enabled": config.get("allow_fallback", False), "ner_model": config.get("fallback_ner_model", "en_core_web_sm"), "ner_version": config.get("fallback_ner_version", "3.7.1")}
        self.ner = None
        self.enc = None
        self.usage = []

    def generate(self, kind, payload):
        schema = ITEM_SCHEMA if kind == "item" else PROFILE_SCHEMA
        if kind == "item":
            prompt = ITEM_PROMPT.format(content=payload["text"])
        else:
            template = {"build": BUILD_PROMPT, "edit": EDIT_PROMPT, "revise": REVISE_PROMPT}[kind]
            prompt = template.format(evidence=canonical(topic_clusters(payload["descriptors"])),
                                     profile=canonical(payload.get("profile")))
        key = digest({"identity": self.identity, "prompt": prompt, "schema": schema})
        path = self.cache / (key + ".json")
        if path.exists():
            record = load_json(path)
            validate(record["output"], schema)
            self.usage.append({**record["usage"], "kind": kind, "cache_hit": True})
            return record["output"]
        start = time.perf_counter()
        fallback = False
        if self.mock:
            if kind == "item":
                text = payload["text"]
                output = {"topic": text.split()[0], "subtopics": [], "style": "tutorial", "audience": "general", "summary": text}
            else:
                counts = Counter(d["topic"] for d in payload["descriptors"])
                if kind == "edit":
                    for entry in payload["profile"]["dominant_interests"]:
                        counts[entry["topic"]] += entry["weight"]
                output = {"dominant_interests": [{"topic": t, "weight": float(n)} for t, n in counts.most_common()],
                          "emerging_interests": [], "content_preferences": ["tutorial"], "excluded_topics": []}
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
        else:
            # Check real tokenizer lengths; never silently truncate profile evidence.
            from transformers import AutoTokenizer
            if not hasattr(self, "tokenizer"):
                self.tokenizer = AutoTokenizer.from_pretrained(self.c["llm_model"], revision=self.c["llm_revision"])
            messages = [{"role": "user", "content": prompt}]
            length = len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            if length + self.c["llm_max_tokens"] > self.c["llm_context"]:
                raise ValueError("Semantic prompt exceeds context. Increase context or document a new compaction policy")
            body = {"model": self.c["llm_model"], "messages": messages, "temperature": 0,
                    "seed": self.c["llm_seed"], "max_tokens": self.c["llm_max_tokens"], "guided_json": schema}
            req = urllib.request.Request(self.c["llm_url"], data=canonical(body).encode(),
                                         headers={"Content-Type": "application/json"})
            # Network and service errors are fatal, never converted into scientific results.
            with urllib.request.urlopen(req, timeout=self.c["llm_timeout"]) as r:
                response = json.load(r)
            usage = response.get("usage", {})
            raw = response["choices"][0]["message"]["content"]
            try:
                output = json.loads(raw)
                validate(output, schema)
            except (ValueError, TypeError, KeyError) as error:
                if kind != "item" or not self.c["allow_fallback"]:
                    raise ValueError(f"Invalid {kind} output; raw response retained in {key}.invalid.json") from self._record_invalid(key, response, error)
                if self.ner is None:
                    import spacy
                    self.ner = spacy.load(self.identity["fallback_policy"]["ner_model"])
                    if self.ner.meta["version"] != self.identity["fallback_policy"]["ner_version"]:
                        raise ValueError("NER checkpoint version does not match configured fallback provenance")
                entities = [entity.text for entity in self.ner(payload["text"]).ents]
                output = item_fallback(payload["text"], entities)
                fallback = True
        validate(output, schema)
        usage = {**usage, "seconds": time.perf_counter() - start, "fallback": fallback}
        save_json(path, {"key": key, "identity": self.identity, "prompt": prompt, "schema": schema,
                         "output": output, "usage": usage})
        self.usage.append({**usage, "kind": kind, "cache_hit": False})
        return output

    def _record_invalid(self, key, response, error):
        save_json(self.cache / (key + ".invalid.json"), response)
        return error

    def encode(self, texts):
        outputs = []
        missing, indices = [], []
        for i, text in enumerate(texts):
            path = self.cache / (digest({"encoder": self.identity, "text": text}) + ".npy")
            if path.exists():
                outputs.append(np.load(path, allow_pickle=False))
            else:
                outputs.append(None)
                missing.append(text)
                indices.append((i, path))
        if missing:
            if self.mock:
                vectors = []
                for text in missing:
                    vec = np.zeros(self.c["mock_dim"], dtype=np.float32)
                    for token in re.findall(r"\w+", text.lower()):
                        token_hash = int(digest(token)[:16], 16)
                        vec[token_hash % len(vec)] += 1 if token_hash & 1 else -1
                    vec /= max(float(np.linalg.norm(vec)), 1e-12)
                    vectors.append(vec)
            else:
                from sentence_transformers import SentenceTransformer
                if self.enc is None:
                    self.enc = SentenceTransformer(self.c["encoder_model"], revision=self.c["encoder_revision"], device=self.c["encoder_device"])
                    self.enc.max_seq_length = self.c["encoder_max_len"]
                    self.enc.eval()
                    for param in self.enc.parameters():
                        param.requires_grad_(False)
                vectors = self.enc.encode(missing, batch_size=self.c["encoder_batch"], normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)
            for (i, path), vec in zip(indices, vectors):
                np.save(path, np.asarray(vec, dtype=np.float32))
                outputs[i] = vec
        return np.stack(outputs).astype(np.float32)

    def profile_vector(self, profile):
        if self.c["profile_embedding"] == "json":
            return self.encode([canonical(profile)])[0]
        entries = profile["dominant_interests"]
        vec = (self.encode([x["topic"] for x in entries]) * np.array([x["weight"] for x in entries])[:, None]).sum(0)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            raise ValueError("Degenerate weighted profile embedding")
        return (vec / norm).astype(np.float32)
