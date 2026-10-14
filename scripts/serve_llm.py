#!/usr/bin/env python3
"""Run in a separate vLLM environment using a resolved model revision."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import yaml
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--port", type=int, default=8000)
p.add_argument("--provenance", default="artifacts/llm_server.json")
a = p.parse_args()
c = yaml.safe_load(Path(a.config).read_text())
if not re.fullmatch(r"[0-9a-f]{40}", str(c["llm_revision"])):
    raise SystemExit("Resolve llm_revision before launching server")
args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", c["llm_model"],
        "--revision", c["llm_revision"], "--tokenizer-revision", c["llm_revision"], "--port", str(a.port),
        "--max-model-len", str(c["llm_context"]), "--dtype", "bfloat16", "--seed", str(c["llm_seed"])]
versions = {pkg: importlib.metadata.version(pkg) for pkg in ["vllm", "torch", "transformers"]}
Path(a.provenance).parent.mkdir(parents=True, exist_ok=True)
Path(a.provenance).write_text(json.dumps({"model": c["llm_model"], "revision": c["llm_revision"], "versions": versions, "argv": args}, indent=2))
os.execv(sys.executable, args)
