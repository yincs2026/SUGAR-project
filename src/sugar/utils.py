import hashlib
import json
import os
from pathlib import Path
import random


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    for key in ["k", "checkpoint", "edit_size", "min_profile", "cooldown", "neighbors", "epochs", "batch_size", "negatives", "alignment_negatives"]:
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if c["lag"] < 0 or c["kappa"] <= 0 or c["social_temperature"] <= 0:
        raise ValueError("Invalid lag, kappa or social temperature")
    if c["dim"] % c["heads"]:
        raise ValueError("dim must be divisible by heads")
    if c.get("neighbor_sample") is not None and int(c["neighbor_sample"]) <= 0:
        raise ValueError("neighbor_sample must be positive or null")
    if not 0 < c.get("bert_mask_probability", 0.2) <= 1:
        raise ValueError("bert_mask_probability must lie in (0,1]")
    if not 0 <= c.get("rlm_mask_ratio", 0.2) < 1:
        raise ValueError("rlm_mask_ratio must lie in [0,1)")
    return c


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment():
    import platform
    import importlib.metadata
    versions = {}
    for pkg in ["torch", "numpy", "transformers", "sentence-transformers", "PyYAML", "scipy", "scikit-learn", "peft", "accelerate"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
