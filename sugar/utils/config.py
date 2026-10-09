
import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_dotted(cfg: dict, key: str, value):
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path: str, base: str = None, overrides: list = None) -> dict:
    cfg = {}
    if base is not None:
        with open(base, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # implicit inheritance: merge over base.yaml when the file sits in configs/
    if base is None and raw.pop("inherits", None) == "base":
        base_path = Path(path).resolve().parents[1] / "base.yaml"
        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"invalid override: '{ov}' (format: key=value)")
        key, value = ov.split("=", 1)
        _apply_dotted(cfg, key, yaml.safe_load(value))
    return cfg


def config_paths(config_name: str, root: Path = None) -> list:
    root = root or Path(__file__).resolve().parents[2]
    p = Path(config_name)
    return [p if p.is_absolute() else root / p]