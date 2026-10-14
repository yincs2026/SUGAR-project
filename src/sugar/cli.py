import argparse
from pathlib import Path
from .utils import load_config, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="SUGAR auditable reproduction pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    schemas = {
        "export-tables": ["suite", "out"], "seed-outcomes": ["suite", "out"], "drift-percentiles": ["data", "artifacts", "out"], "offline-costs": ["artifacts", "out"], "parameter-counts": ["data", "artifacts", "run", "out"], "verify-paper": ["suite", "out"], "suite": ["spec", "out"], "coherence": ["data", "artifacts", "out"], "static-profiles": ["data", "artifacts"],
        "prepare": ["raw", "out"], "demo-data": ["out"], "resolve-models": ["out"],
        "offline": ["data", "out"], "audit": ["data", "artifacts", "out"],
        "proxy-graph": ["data", "out"], "train": ["data", "artifacts", "out"],
        "multi-seed": ["data", "artifacts", "out"], "evaluate": ["data", "artifacts", "run", "out"],
        "benchmark-pipeline": ["data", "artifacts", "run", "out"], "stress": ["data", "artifacts", "run", "out"], "benchmark": ["data", "artifacts", "run", "out"],
        "drift": ["data", "artifacts", "out"], "aggregate": ["results", "out"],
        "significance": ["a", "b", "out"], "variant": ["name", "out"],
        "export-protocol": ["data", "artifacts", "out"], "import-baseline": ["protocol", "results", "out"],
        "recommend": ["data", "artifacts", "run", "user", "position", "out"],
        "search": ["data", "artifacts", "out"], "sweep": ["raw", "out", "parameter", "values"], "smoke": ["out"]}
    configurable = {"prepare", "resolve-models", "offline", "audit", "train", "multi-seed", "variant", "export-protocol", "search", "sweep", "smoke", "static-profiles"}
    device_commands = {"train", "multi-seed", "evaluate", "stress", "benchmark", "benchmark-pipeline", "search", "sweep", "recommend", "suite", "parameter-counts"}
    for name, fields in schemas.items():
        p = commands.add_parser(name)
        for field in fields:
            multiple = (name in {"drift", "drift-percentiles"} and field in {"data", "artifacts"}) or (name == "aggregate" and field == "results") or (name == "significance" and field in {"a", "b"}) or field == "values"
            opts = {"required": True}
            if multiple:
                opts["nargs"] = "+"
            if field == "values":
                opts["type"] = float
            p.add_argument("--" + field, **opts)
        if name in configurable:
            p.add_argument("--config", default="configs/paper.yaml")
        if name in device_commands:
            p.add_argument("--device", default="cpu")
        if name in {"offline", "sweep", "static-profiles"}:
            p.add_argument("--cache", default="cache/semantic")
        if name == "suite":
            p.add_argument("--resume", action="store_true")
        if name == "coherence":
            p.add_argument("--seed", type=int, default=42)
        if name == "train":
            p.add_argument("--seed", type=int)
            p.add_argument("--resume")
        if name == "evaluate":
            p.add_argument("--split", choices=["test", "validation"], default="test")
        if name == "recommend":
            p.add_argument("--k", type=int, default=10)
        if name in {"benchmark", "benchmark-pipeline"}:
            p.add_argument("--requests", type=int, default=1000)
        if name == "drift":
            p.add_argument("--epsilon", type=float)
        if name == "variant":
            p.add_argument("--fusion-weight", type=float, default=0.5)
        if name == "search":
            p.add_argument("--trials", type=int, default=20)
        if name == "smoke":
            p.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    c = load_config(args.config) if hasattr(args, "config") else None
    cmd = args.command
    if cmd == "export-tables":
        from .coverage import export_paper_tables
        print(export_paper_tables(args.suite, args.out))
    elif cmd == "seed-outcomes":
        from .coverage import seed_outcomes
        print(seed_outcomes(args.suite, args.out))
    elif cmd == "drift-percentiles":
        from .coverage import drift_percentiles
        print(drift_percentiles(args.data, args.artifacts, args.out))
    elif cmd == "offline-costs":
        from .coverage import offline_costs
        print(offline_costs(args.artifacts, args.out))
    elif cmd == "parameter-counts":
        from .coverage import parameter_counts
        print(parameter_counts(args.data, args.artifacts, args.run, args.out, args.device))
    elif cmd == "verify-paper":
        from .verification import verify_paper
        print(verify_paper(args.suite, args.out))
    elif cmd == "suite":
        from .suite import run_suite
        run_suite(args.spec, args.out, args.device, args.resume)
    elif cmd == "coherence":
        from .coherence import encoder_quality
        print(encoder_quality(args.data, args.artifacts, args.out, args.seed))
    elif cmd == "static-profiles":
        from .coherence import static_baseline_profiles
        static_baseline_profiles(args.data, args.artifacts, c, args.cache)
    elif cmd == "prepare":
        from .data import prepare
        prepare(args.raw, args.out, c)
    elif cmd == "demo-data":
        from .data import demo
        demo(args.out)
    elif cmd == "resolve-models":
        from huggingface_hub import HfApi
        import yaml
        api = HfApi()
        for kind in ["llm", "encoder"] + (["llara_llm"] if c.get("llara_llm_model") else []):
            c[f"{kind}_revision"] = api.model_info(c[f"{kind}_model"], revision=c[f"{kind}_revision"] or "main").sha
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(c, sort_keys=False))
        print("Resolved immutable commits into", args.out)
    elif cmd == "offline":
        from .offline import build_snapshots
        print(build_snapshots(load_json(args.data), args.out, c, args.cache))
    elif cmd == "audit":
        from .features import FeatureStore
        report = FeatureStore(load_json(args.data), args.artifacts, c).audit()
        save_json(args.out, report)
        print(report)
    elif cmd == "proxy-graph":
        from .data import proxy_graph
        save_json(args.out, proxy_graph(load_json(args.data)))
    elif cmd == "train":
        from .train import train
        if args.seed is not None:
            c["seed"] = args.seed
        print(train(args.data, args.artifacts, args.out, c, args.device, args.resume)["metrics"])
    elif cmd == "multi-seed":
        from .train import train
        from .experiments import aggregate
        paths = []
        for seed in c["seeds"]:
            directory = Path(args.out) / f"seed-{seed}"
            train(args.data, args.artifacts, directory, {**c, "seed": seed}, args.device)
            paths.append(str(directory / "test.json"))
        print(aggregate(paths, Path(args.out) / "summary.json"))
    elif cmd == "evaluate":
        from .train import load_run, evaluate
        model, data, store, c = load_run(args.data, args.artifacts, args.run, args.device)
        result = evaluate(model, data, store, c, args.split, args.device)
        result["mock"] = store.manifest["mock"]
        save_json(args.out, result)
        print(result["metrics"])
    elif cmd == "recommend":
        from .inference import recommend
        result = recommend(args.data, args.artifacts, args.run, args.user, int(args.position), args.k, args.device)
        save_json(args.out, result)
        print(result)
    elif cmd == "stress":
        from .experiments import stress_test
        stress_test(args.data, args.artifacts, args.run, args.out, args.device)
    elif cmd == "benchmark-pipeline":
        from .experiments import benchmark_pipeline
        print(benchmark_pipeline(args.data, args.artifacts, args.run, args.out, args.device, args.requests))
    elif cmd == "benchmark":
        from .experiments import benchmark
        print(benchmark(args.data, args.artifacts, args.run, args.out, args.device, args.requests))
    elif cmd == "drift":
        from .experiments import drift_analysis
        print(drift_analysis(args.data, args.artifacts, args.out, args.epsilon))
    elif cmd == "aggregate":
        from .experiments import aggregate
        print(aggregate(args.results, args.out))
    elif cmd == "significance":
        from .experiments import paired_test
        print(paired_test(args.a, args.b, args.out))
    elif cmd == "variant":
        import yaml
        patches = {"sasrec": {"model": "sasrec", "alignment_weight": 0}, "lm-feature": {"model": "lm-feature", "alignment_weight": 0},
                   "gru4rec": {"model": "gru4rec", "alignment_weight": 0, "loss": "bpr-max"},
                   "caser": {"model": "caser", "alignment_weight": 0},
                   "bert4rec": {"model": "bert4rec", "alignment_weight": 0},
                   "socialmf": {"model": "socialmf", "alignment_weight": 0},
                   "lightgcn-plus": {"model": "lightgcn-plus", "alignment_weight": 0},
                   "rlmrec": {"model": "rlmrec", "alignment_weight": 0, "loss": "bpr"},
                   "rlmrec-gen": {"model": "rlmrec", "rlm_variant": "generative", "alignment_weight": 0, "loss": "bpr"},
                   "llara": {"model": "llara", "alignment_weight": 0, "lr": 0.0001, "batch_size": 4},
                   "separate-projection": {"separate_alignment_projection": True},
                   "sampled-neighbors": {"neighbor_sample": 100},
                   "recency-window": {"recent_decay": 0.1},
                   "leaky-snapshots": {"leaky_snapshots": True}, "no-drift": {"no_drift": True}, "no-social": {"no_social": True}, "no-alignment": {"alignment_weight": 0},
                   "fixed-fusion": {"fixed_fusion": args.fusion_weight}, "bpr": {"loss": "bpr"},
                   "single-strike": {"single_strike": True}, "live-score": {"recompute_score": True}, "json-profile": {"profile_embedding": "json"}}
        if args.name not in patches or not 0 <= args.fusion_weight <= 1:
            raise ValueError("Invalid variant/fusion weight")
        c.update(patches[args.name])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(c, sort_keys=False))
    elif cmd == "export-protocol":
        from .experiments import export_protocol
        export_protocol(args.data, args.artifacts, c, args.out)
    elif cmd == "import-baseline":
        from .experiments import import_baseline
        print(import_baseline(args.protocol, args.results, args.out)["metrics"])
    elif cmd == "search":
        from .train import train
        import numpy as np
        space = {"dim": [32, 64, 128], "blocks": [1, 2, 3], "negatives": [1, 4], "alignment_weight": [0, 0.01, 0.05, 0.1, 0.3],
                 "alignment_temperature": [0.05, 0.1, 0.2], "kappa": [10, 25, 50, 100], "neighbors": [5, 10, 20, 50], "dropout": [0.1, 0.2, 0.5]}
        rng = np.random.default_rng(c["seed"])
        best = None
        for trial in range(args.trials):
            cc = {**c, **{k: values[int(rng.integers(len(values)))] for k, values in space.items()}}
            directory = Path(args.out) / f"trial-{trial:03}"
            train(args.data, args.artifacts, directory, cc, args.device, evaluate_test=False)
            validation = load_json(directory / "validation.json")["metrics"]["NDCG@10"]
            if best is None or validation > best["validation_ndcg"]:
                best = {"trial": trial, "validation_ndcg": validation, "config": cc}
        if best is None:
            raise ValueError("Need positive number of trials")
        save_json(Path(args.out) / "best_configuration.json", best)
    elif cmd == "sweep":
        from .data import prepare
        from .offline import build_snapshots
        from .train import train
        if args.parameter not in {"lag", "neighbors", "kappa", "xi", "tau"}:
            raise ValueError("Unknown sweep parameter")
        rows = []
        for value in args.values:
            cc = dict(c)
            cc[args.parameter] = int(value) if args.parameter in {"lag", "neighbors"} else value
            if args.parameter == "tau":
                cc["threshold"] = "global"
            directory = Path(args.out) / f"{args.parameter}-{value:g}"
            data = prepare(args.raw, directory / "prepared", cc)
            build_snapshots(data, directory / "artifacts", cc, args.cache)
            result = train(directory / "prepared/data.json", directory / "artifacts", directory / "run", cc, args.device)
            logs = load_json(directory / "artifacts/offline_log.json")
            frequency = 1000 * sum(x["trigger"] for x in logs) / len(logs) if logs else None
            rows.append({"value": value, "metrics": result["metrics"], "revision_calls_per_1000": frequency,
                         "within_budget": frequency is not None and frequency <= 40})
        save_json(Path(args.out) / "sweep.json", {"parameter": args.parameter, "seed": c["seed"], "results": rows})
    elif cmd == "smoke":
        from .data import demo, prepare
        from .offline import build_snapshots
        from .train import train
        import torch
        torch.set_num_threads(1)
        c.update({"semantic_backend": "mock", "allow_nonpaper": True, "epochs": args.epochs, "dim": 16,
                  "max_len": 80, "batch_size": 32, "dropout": 0.0, "alignment_negatives": 8,
                  "checkpoint": 5, "k": 5, "lag": 2, "edit_size": 5, "cooldown": 10, "neighbors": 3})
        root = Path(args.out)
        demo(root / "raw")
        data = prepare(root / "raw", root / "prepared", c)
        build_snapshots(data, root / "artifacts", c, root / "cache")
        report = train(root / "prepared/data.json", root / "artifacts", root / "run", c)
        save_json(root / "smoke_summary.json", {"infrastructure_test_only": True, "paper_results_reproduced": False, "metrics": report["metrics"]})
        print("SMOKE ONLY; these are synthetic data and mock semantics.")


if __name__ == "__main__":
    main()
