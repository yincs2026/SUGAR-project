"""One entry point for paper tables, ablations and additional experiments.
Computes fresh results only. Missing inputs fail rather than fabricating benchmark outputs.
"""
from pathlib import Path
import csv
import json
import numpy as np
import yaml
from .utils import load_config, load_json, save_json, digest
from .data import prepare, demo, proxy_graph
from .offline import build_snapshots, verify_manifest
from .coherence import static_baseline_profiles, encoder_quality
from .factory import MODELS
from .coverage import drift_percentiles, offline_costs, parameter_counts, sweep_frontier, seed_outcomes, export_paper_tables
from .train import train
from .experiments import aggregate, paired_test, stress_test, drift_analysis, benchmark, benchmark_pipeline

VARIANTS = {'no-drift': {'no_drift': True}, 'no-social': {'no_social': True},
            'no-alignment': {'alignment_weight': 0}, 'leaky-snapshots': {'leaky_snapshots': True},
            'json-profile': {'profile_embedding': 'json'}, 'recency-window': {'recent_decay': 0.1},
            'separate-projection': {'separate_alignment_projection': True},
            'sampled-neighbors': {'neighbor_sample': 100}, 'bpr': {'loss': 'bpr'},
            'single-strike': {'single_strike': True}, 'live-score': {'recompute_score': True}}

MODEL_PATCHES = {'sasrec': {'alignment_weight': 0}, 'lm-feature': {'alignment_weight': 0},
                 'gru4rec': {'alignment_weight': 0, 'loss': 'bpr-max'},
                 'caser': {'alignment_weight': 0}, 'bert4rec': {'alignment_weight': 0},
                 'socialmf': {'alignment_weight': 0}, 'lightgcn-plus': {'alignment_weight': 0},
                 'rlmrec': {'alignment_weight': 0, 'loss': 'bpr'},
                 'llara': {'alignment_weight': 0, 'lr': 1e-4, 'batch_size': 4}}

SEARCH_SPACES = {
    'sasrec': {'dim': [32, 64, 128], 'blocks': [1, 2, 3], 'dropout': [0.1, 0.2, 0.5]},
    'lm-feature': {'dim': [32, 64, 128], 'blocks': [1, 2, 3], 'dropout': [0.1, 0.2, 0.5]},
    'gru4rec': {'dim': [32, 64, 128], 'gru_layers': [1, 2], 'dropout': [0.1, 0.2, 0.5], 'loss': ['bce', 'bpr-max']},
    'caser': {'dim': [32, 64, 128], 'caser_len': [3, 5, 10], 'caser_horizontal': [8, 16, 32], 'caser_vertical': [2, 4, 8]},
    'bert4rec': {'dim': [32, 64, 128], 'blocks': [1, 2, 3], 'bert_mask_probability': [0.1, 0.2, 0.4]},
    'socialmf': {'dim': [32, 64, 128], 'social_regularization': [0.01, 0.1, 1], 'embedding_regularization': [1e-6, 1e-5, 1e-4]},
    'lightgcn-plus': {'dim': [32, 64, 128], 'graph_layers': [1, 2, 3], 'dropout': [0.1, 0.2, 0.5]},
    'rlmrec': {'dim': [32, 64, 128], 'graph_layers': [1, 2, 3], 'rlm_alignment_weight': [0.01, 0.1, 1], 'rlm_temperature': [0.1, 0.2, 0.5]},
    'llara': {'llara_lora_rank': [4, 8, 16], 'llara_lora_alpha': [8, 16, 32], 'lr': [5e-5, 1e-4, 2e-4], 'llara_curriculum_epochs': [3, 5, 10]}}


def run_suite(spec_path, out, device='cpu', resume=False):
    spec = yaml.safe_load(Path(spec_path).read_text())
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    resume_manifest = root / 'suite_manifest.json'
    if resume_manifest.exists() and not resume:
        raise ValueError('Suite output exists; use --resume or a new directory')
    if resume_manifest.exists() and load_json(resume_manifest)['spec_hash'] != digest(spec):
        raise ValueError('Suite specification changed; use a fresh output directory')
    manifest = {'spec_hash': digest(spec), 'spec': spec, 'device': device, 'completed': [],
                'paper_numbers_verified': False, 'status': 'running'}
    completed_before = load_json(resume_manifest)['completed'] if resume_manifest.exists() else []
    manifest['completed'] = completed_before
    save_json(resume_manifest, manifest)
    summaries, datasets, artifact_dirs = [], [], []

    def checkpoint_step(key):
        if key not in manifest['completed']:
            manifest['completed'].append(key)
        save_json(resume_manifest, manifest)

    def execute_train(data_file, artifacts, directory, c, evaluate_test=True):
        directory = Path(directory)
        result_file = directory / ('test.json' if evaluate_test else 'selected_validation.json')
        if resume and result_file.exists():
            saved = load_json(directory / 'config.json')
            if saved != c:
                raise ValueError(f'Resume configuration changed for {directory}')
            provenance = load_json(directory / 'provenance.json')
            data = load_json(data_file)
            artifacts_manifest = verify_manifest(data, artifacts)
            if provenance['data'] != data['fingerprint'] or provenance['artifacts'] != digest(artifacts_manifest):
                raise ValueError(f'Resume artifacts/data changed for {directory}')
            return load_json(result_file)
        checkpoint = directory / 'last.pt'
        try:
            return train(data_file, artifacts, directory, c, device,
                         resume=str(checkpoint) if resume and checkpoint.exists() else None,
                         evaluate_test=evaluate_test)
        except Exception as error:
            save_json(directory / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'seed': c['seed']})
            raise

    try:
        for ds in spec['datasets']:
            name = ds['name']
            directory = root / name
            c = load_config(ds['config'])
            c.update(spec.get('overrides', {}))
            c.update(ds.get('overrides', {}))
            c['dataset'] = name
            seeds = spec.get('seeds', c['seeds'])
            if len(seeds) != len(set(seeds)):
                raise ValueError('Suite seeds must be distinct')
            c['seeds'] = seeds
            if spec.get('research_only', False) and (ds.get('demo', False) or c['semantic_backend'] != 'real' or c.get('llara_backend') != 'real' or c.get('allow_nonpaper', False)):
                raise ValueError('Research suite requires real semantic/LoRA backends and forbids demo data and allow_nonpaper')
            if ds.get('demo', False):
                c.update({'semantic_backend': 'mock', 'allow_nonpaper': True, 'llara_backend': 'tiny'})
                raw = directory / 'raw'
                demo(raw, users=6, items=12, events=25)
            else:
                raw = ds.get('raw')
            data_file = Path(ds['prepared']) if ds.get('prepared') else directory / 'prepared/data.json'
            if ds.get('prepared'):
                data = load_json(data_file)
            elif resume and data_file.exists():
                data = load_json(data_file)
            else:
                data = prepare(raw, directory / 'prepared', c)
            artifacts = directory / 'artifacts'
            cache = Path(spec.get('cache', 'cache/suite')) / name
            if not (resume and (artifacts / 'static_user_embeddings.npy').exists()):
                build_snapshots(data, artifacts, c, cache)
                static_baseline_profiles(data_file, artifacts, c, cache)
            verify_manifest(data, artifacts)
            datasets.append(str(data_file))
            artifact_dirs.append(str(artifacts))
            save_json(directory / 'dataset_statistics.json', data['counts'])
            encoder_quality(data_file, artifacts, directory / 'encoder_quality.json', c['seed'])
            models = spec.get('models', ['sasrec', 'sugar', 'gru4rec', 'caser', 'bert4rec', 'lm-feature', 'socialmf', 'lightgcn-plus', 'rlmrec', 'llara'])
            for name_model in models:
                if name_model not in MODELS:
                    raise ValueError(f'Unknown requested suite model {name_model}')
                cc = {**c, **MODEL_PATCHES.get(name_model, {}), 'model': name_model}
                if name_model == 'llara':
                    if 'sasrec' not in models:
                        raise ValueError('Put sasrec before llara in models, or provide an explicit backbone checkpoint')
                    cc['llara_backbone_checkpoint'] = str(directory / 'models/sasrec' / f'seed-{seeds[0]}' / 'best.pt')
                trials = int(spec.get('tune_trials', 20)) if name_model != 'sugar' else 0
                if trials:
                    rng = np.random.default_rng(c['seed'])
                    best = None
                    for trial in range(trials):
                        trial_config = dict(cc)
                        for key, values in SEARCH_SPACES[name_model].items():
                            trial_config[key] = values[int(rng.integers(len(values)))]
                        trial_dir = directory / 'tuning' / name_model / f'trial-{trial:03}'
                        execute_train(data_file, artifacts, trial_dir, trial_config, evaluate_test=False)
                        validation = load_json(trial_dir / 'validation.json')['metrics']['NDCG@10']
                        if best is None or validation > best['validation_ndcg']:
                            best = {'config': trial_config, 'validation_ndcg': validation, 'trial': trial}
                    save_json(directory / 'tuning' / name_model / 'selected.json', best)
                    cc = best['config']
                paths = []
                for seed in seeds:
                    run_config = {**cc, 'seed': seed}
                    if name_model == 'llara':
                        run_config['llara_backbone_checkpoint'] = str(directory / 'models/sasrec' / f'seed-{seed}' / 'best.pt')
                    run_dir = directory / 'models' / name_model / f'seed-{seed}'
                    execute_train(data_file, artifacts, run_dir, run_config)
                    paths.append(str(run_dir / 'test.json'))
                    checkpoint_step(f'{name}/{name_model}/{seed}')
                # Backbone file paths vary across seeds, but architecture/configuration do not.
                summary = aggregate(paths, directory / 'models' / name_model / 'summary.json')
                summaries.append({'dataset': name, 'model': name_model, 'summary': summary})
            if 'sugar' in models:
                variants = list(spec.get('variants', VARIANTS.keys()))
                for variant in variants:
                    if variant not in VARIANTS and variant != 'fixed-fusion':
                        raise ValueError(f'Unknown variant {variant}')
                    cc = {**c, **VARIANTS.get(variant, {})}
                    variant_artifacts = directory / 'variants' / variant / 'artifacts'
                    if not (resume and (variant_artifacts / 'manifest.json').exists()):
                        build_snapshots(data, variant_artifacts, cc, cache)
                    if variant == 'fixed-fusion':
                        best = None
                        for weight in spec.get('fixed_fusion_grid', [0, 0.25, 0.5, 0.75, 1]):
                            tuned = {**cc, 'fixed_fusion': weight}
                            td = directory / 'variants' / variant / f'tune-{weight:g}'
                            execute_train(data_file, variant_artifacts, td, tuned, False)
                            metric = load_json(td / 'validation.json')['metrics']['NDCG@10']
                            if best is None or metric > best['validation_ndcg']:
                                best = {'weight': weight, 'validation_ndcg': metric}
                        cc['fixed_fusion'] = best['weight']
                        save_json(directory / 'variants' / variant / 'selected.json', best)
                    paths = []
                    for seed in seeds:
                        rd = directory / 'variants' / variant / f'seed-{seed}'
                        execute_train(data_file, variant_artifacts, rd, {**cc, 'seed': seed})
                        paths.append(str(rd / 'test.json'))
                        checkpoint_step(f'{name}/{variant}/{seed}')
                    summary = aggregate(paths, directory / 'variants' / variant / 'summary.json')
                    summaries.append({'dataset': name, 'model': variant, 'summary': summary})
                first_run = directory / 'models/sugar' / f'seed-{seeds[0]}'
                offline_costs(artifacts, directory / 'offline_costs.json')
                parameter_counts(data_file, artifacts, first_run, directory / 'parameter_counts.json', device)
                stress_test(data_file, artifacts, first_run, directory / 'stress.json', device)
                stress_rows = {'sugar': load_json(directory / 'stress.json')}
                if 'sasrec' in models:
                    sasrec_run = directory / 'models/sasrec' / f'seed-{seeds[0]}'
                    stress_rows['sasrec'] = stress_test(data_file, artifacts, sasrec_run, directory / 'stress_sasrec.json', device)
                if 'fixed-fusion' in variants:
                    fixed_root = directory / 'variants/fixed-fusion'
                    stress_rows['fixed-fusion'] = stress_test(data_file, fixed_root / 'artifacts', fixed_root / f'seed-{seeds[0]}', directory / 'stress_fixed_fusion.json', device)
                save_json(directory / 'stress_comparison.json', {
                    'conditions': {condition: {method: report[condition]['metrics']['HR@10'] for method, report in stress_rows.items()}
                                   for condition in ['clean', 'corrupted', 'stale']},
                    'seed': seeds[0], 'reports': stress_rows})
                if 'sasrec' in models and spec.get('benchmark', True):
                    requests = int(spec.get('benchmark_requests', 1000))
                    sug = benchmark(data_file, artifacts, first_run, directory / 'latency_sugar.json', device, requests)
                    sas = benchmark(data_file, artifacts, directory / 'models/sasrec' / f'seed-{seeds[0]}', directory / 'latency_sasrec.json', device, requests)
                    if spec.get('benchmark_pipeline', True):
                        benchmark_pipeline(data_file, artifacts, first_run, directory / 'latency_pipeline_sugar.json', device, requests)
                        benchmark_pipeline(data_file, artifacts, directory / 'models/sasrec' / f'seed-{seeds[0]}', directory / 'latency_pipeline_sasrec.json', device, requests)
                    save_json(directory / 'latency_comparison.json', {'sugar': sug, 'sasrec': sas,
                              'throughput_overhead_percent': 100 * (1 - sug['requests_per_second'] / sas['requests_per_second'])})
                if spec.get('proxy_graph', True):
                    proxy = proxy_graph(json.loads(json.dumps(data)))
                    proxy_file = directory / 'proxy/data.json'
                    save_json(proxy_file, proxy)
                    proxy_artifacts = directory / 'proxy/artifacts'
                    if not (resume and (proxy_artifacts / 'manifest.json').exists()):
                        build_snapshots(proxy, proxy_artifacts, c, cache)
                    for model_name in ['sugar', 'lightgcn-plus']:
                        if model_name not in models:
                            continue
                        paths = []
                        for seed in seeds:
                            rd = directory / 'proxy' / model_name / f'seed-{seed}'
                            execute_train(proxy_file, proxy_artifacts, rd, {**c, **MODEL_PATCHES.get(model_name, {}), 'model': model_name, 'seed': seed})
                            paths.append(str(rd / 'test.json'))
                        aggregate(paths, directory / 'proxy' / model_name / 'summary.json')
                if len(seeds) > 1:
                    sugar_paths = [str(directory / 'models/sugar' / f'seed-{seed}' / 'test.json') for seed in seeds]
                    # Strongest comparator selected by validation, never by observed test accuracy.
                    choices = []
                    for model_name in models:
                        if model_name != 'sugar':
                            vals = [load_json(directory / 'models' / model_name / f'seed-{seed}/validation.json')['metrics']['NDCG@10'] for seed in seeds]
                            choices.append((float(np.mean(vals)), model_name))
                    if choices:
                        best_baseline = max(choices)[1]
                        baseline_paths = [str(directory / 'models' / best_baseline / f'seed-{seed}' / 'test.json') for seed in seeds]
                        paired_test(sugar_paths, baseline_paths, directory / 'significance.json')
                        save_json(directory / 'strongest_baseline.json', {'name': best_baseline, 'selection': 'mean validation NDCG@10'})
                for parameter, values in spec.get('sweeps', {}).items():
                    for value in values:
                        sweep_config = {**c, parameter: value}
                        if parameter == 'tau':
                            sweep_config['threshold'] = 'global'
                        ad = directory / 'sweeps' / f'{parameter}-{value}' / 'artifacts'
                        if not (resume and (ad / 'manifest.json').exists()):
                            build_snapshots(data, ad, sweep_config, cache)
                        rd = directory / 'sweeps' / f'{parameter}-{value}' / 'run'
                        execute_train(data_file, ad, rd, sweep_config)
                        checkpoint_step(f'{name}/sweep/{parameter}/{value}')
            sweep_frontier(directory, directory / 'sensitivity_frontier.json')
            seed_outcomes(directory, directory / 'seed_outcomes.json')
            save_json(root / 'computed_results.json', summaries)
        drift_analysis(datasets, artifact_dirs, root / 'drift_detection.json')
        drift_percentiles(datasets, artifact_dirs, root / 'drift_percentile_robustness.json', spec.get('drift_percentiles', [85, 90, 95]))
        for variant in ['single-strike', 'no-drift']:
            dirs = [str(root / ds['name'] / 'variants' / variant / 'artifacts') for ds in spec['datasets']]
            if all((Path(d) / 'manifest.json').exists() for d in dirs):
                drift_analysis(datasets, dirs, root / f'drift_comparison_{variant}.json')
        with (root / 'computed_results.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dataset', 'model', 'seeds', 'mock', 'metric', 'mean', 'std'])
            for row in summaries:
                for metric, value in row['summary']['metrics'].items():
                    writer.writerow([row['dataset'], row['model'], row['summary']['runs'], row['summary']['mock'], metric, value['mean'], value['std']])
        export_paper_tables(root, root / 'tables')
        manifest['status'] = 'completed'
        manifest['paper_numbers_verified'] = False
        manifest['reason'] = 'Execution completeness does not establish equality with reported manuscript numbers'
        save_json(resume_manifest, manifest)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        save_json(resume_manifest, manifest)
        raise
    return manifest
