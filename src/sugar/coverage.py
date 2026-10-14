"""Additional measurements and a traceable paper experiment-coverage inventory."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .utils import load_json, save_json
from .experiments import drift_analysis
from .train import load_run


def drift_percentiles(dataset_files, artifact_dirs, out, percentiles=(85, 90, 95)):
    result = {'percentiles': list(percentiles), 'results': {}}
    root = Path(out).parent
    for percentile in percentiles:
        if not 0 < percentile < 100:
            raise ValueError('Percentile must lie strictly between 0 and 100')
        report = drift_analysis(dataset_files, artifact_dirs, root / f'drift_percentile_{percentile}.json', percentile=percentile)
        result['results'][str(percentile)] = report
    save_json(out, result)
    return result


def offline_costs(artifacts, out):
    artifacts = Path(artifacts)
    usage = load_json(artifacts / 'llm_usage.json')
    groups = defaultdict(lambda: {'logical_calls': 0, 'executed_calls': 0, 'cache_hits': 0,
                                 'logical_tokens': 0, 'executed_tokens': 0, 'executed_seconds': 0.0,
                                 'fallbacks': 0})
    for record in usage:
        group = groups[record['kind']]
        tokens = record.get('prompt_tokens', 0) + record.get('completion_tokens', 0)
        group['logical_calls'] += 1
        group['logical_tokens'] += tokens
        group['fallbacks'] += bool(record['fallback'])
        if record['cache_hit']:
            group['cache_hits'] += 1
        else:
            group['executed_calls'] += 1
            group['executed_tokens'] += tokens
            group['executed_seconds'] += record['seconds']
    total_logical = sum(g['logical_tokens'] for g in groups.values())
    total_executed = sum(g['executed_tokens'] for g in groups.values())
    for g in groups.values():
        g['logical_token_share_percent'] = 100 * g['logical_tokens'] / total_logical if total_logical else None
        g['executed_token_share_percent'] = 100 * g['executed_tokens'] / total_executed if total_executed else None
    manifest = load_json(artifacts / 'manifest.json')
    descriptor_calls = groups.get('item', {}).get('logical_calls', 0)
    report = {'by_module_operation': dict(groups), 'total_logical_tokens': total_logical,
              'total_executed_tokens': total_executed,
              'item_fallback_rate': groups.get('item', {}).get('fallbacks', 0) / descriptor_calls if descriptor_calls else None,
              'total_executed_llm_seconds': sum(g['executed_seconds'] for g in groups.values()),
              'mock': manifest['mock'], 'catalog_calls': descriptor_calls,
              'wall_clock_measurement': load_json(artifacts / 'offline_timing.json') if (artifacts / 'offline_timing.json').exists() else None,
              'scope': 'Main SUGAR M1-M3 only; independent static baseline profiling excluded'}
    save_json(out, report)
    return report


def parameter_counts(data_path, artifacts, run, out, device='cpu'):
    model, _, store, config = load_run(data_path, artifacts, run, device)
    backbone_prefixes = ('items.', 'positions.', 'blocks.', 'final_norm.', 'input_dropout.')
    counts = defaultdict(int)
    frozen = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen += parameter.numel()
        else:
            group = 'backbone' if name.startswith(backbone_prefixes) else 'augmentation'
            counts[group] += parameter.numel()
    trainable = sum(counts.values())
    report = {'model': config['model'], 'trainable': trainable, 'frozen_parameters': frozen,
              'components': dict(counts), 'augmentation_percent_of_backbone':
              100 * counts['augmentation'] / counts['backbone'] if counts['backbone'] else None,
              'mock': store.manifest['mock'], 'scope': 'Classification of SUGAR/SASRec parameter names; buffers excluded'}
    save_json(out, report)
    return report


def sweep_frontier(dataset_dir, out):
    root = Path(dataset_dir)
    points = []
    for directory in sorted((root / 'sweeps').glob('*')):
        run = directory / 'run'
        if not (run / 'test.json').exists():
            continue
        c = load_json(run / 'config.json')
        metrics = load_json(run / 'test.json')['metrics']
        logs = load_json(directory / 'artifacts/offline_log.json')
        frequency = 1000 * sum(r['trigger'] for r in logs) / len(logs) if logs else 0
        points.append({'configuration': directory.name, 'threshold_policy': c['threshold'], 'tau': c['tau'],
                       'xi': c['xi'], 'revision_calls_per_1000': frequency,
                       'within_40_call_budget': frequency <= 40, 'NDCG@10': metrics['NDCG@10']})
    threshold_points = [p for p in points if p['configuration'].startswith(('tau-', 'xi-'))]
    pareto = [p for p in threshold_points if not any(
        q['revision_calls_per_1000'] <= p['revision_calls_per_1000'] and q['NDCG@10'] >= p['NDCG@10'] and
        (q['revision_calls_per_1000'] < p['revision_calls_per_1000'] or q['NDCG@10'] > p['NDCG@10'])
        for q in threshold_points)]
    report = {'points': points, 'threshold_pareto_front': pareto,
              'pareto_definition': 'maximize observed NDCG, minimize revision calls; does not select main configuration on test'}
    save_json(out, report)
    return report


def seed_outcomes(dataset_dir, out):
    """Record executed seeds; completion is not an inferred convergence label."""
    rows = []
    for directory in sorted(Path(dataset_dir).glob('models/*/seed-*')):
        rows.append({'model': directory.parent.name, 'seed': directory.name,
                     'status': 'completed' if (directory / 'test.json').exists() else
                     'failed' if (directory / 'failure.json').exists() else 'incomplete',
                     'failure': load_json(directory / 'failure.json') if (directory / 'failure.json').exists() else None})
    report = {'runs': rows, 'convergence_policy': 'No unpublished convergence criterion is inferred. Failures are recorded and abort the suite; seeds are never silently excluded.'}
    save_json(out, report)
    return report


def export_paper_tables(suite, out):
    """Export computed measurements, never the reference paper's scores."""
    import csv
    root, destination = Path(suite), Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {str(i): [] for i in range(4, 10)}
    for ds in sorted(root.iterdir()):
        if not ds.is_dir() or not (ds / 'dataset_statistics.json').exists():
            continue
        stats = load_json(ds / 'dataset_statistics.json')
        tables['4'].append({'dataset': ds.name, **stats,
                            'average_sequence_length': stats['interactions'] / stats['users'] if stats['users'] else None,
                            'average_social_out_degree': stats['social_edges'] / stats['users'] if stats['users'] else None})
        main_path = ds / 'models/sugar/summary.json'
        main = load_json(main_path) if main_path.exists() else None
        for model_dir in sorted((ds / 'models').glob('*')):
            summary_path = model_dir / 'summary.json'
            if not summary_path.exists():
                continue
            summary = load_json(summary_path)
            for metric, value in summary['metrics'].items():
                row = {'dataset': ds.name, 'model': model_dir.name, 'metric': metric, **value,
                       'seeds': summary['seeds'], 'mock': summary['mock']}
                tables['9'].append(row)
                if model_dir.name == 'sugar':
                    tables['5'].append(row)
        if main:
            for bucket, group in main.get('buckets', {}).items():
                for metric, value in group['metrics'].items():
                    tables['6'].append({'dataset': ds.name, 'bucket': bucket, 'users': group['users'],
                                        'percent_of_test_users': group['percent_of_test_users'],
                                        'metric': metric, **value, 'mock': main['mock']})
        for variant in sorted((ds / 'variants').glob('*/summary.json')):
            summary = load_json(variant)
            for metric, value in summary['metrics'].items():
                tables['7'].append({'dataset': ds.name, 'variant': variant.parent.name, 'metric': metric,
                                    **value, 'delta_from_sugar': value['mean'] - main['metrics'][metric]['mean'] if main else None,
                                    'mock': summary['mock']})
        stress_path = ds / 'stress_comparison.json'
        if stress_path.exists():
            stress = load_json(stress_path)
            for condition, models in stress['conditions'].items():
                for model, value in models.items():
                    tables['8'].append({'dataset': ds.name, 'condition': condition, 'model': model,
                                        'HR@10': value, 'seed': stress['seed'], 'mock': main['mock'] if main else None,
                                        'details': str(stress_path.relative_to(root))})
    for number, rows in tables.items():
        save_json(destination / f'table_{number}.json', rows)
        columns = sorted(set().union(*(r.keys() for r in rows))) if rows else ['dataset']
        with (destination / f'table_{number}.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            writer.writerows(rows)
    report = {'tables': {k: len(v) for k, v in tables.items()},
              'note': 'Computed results only. Empty tables indicate missing runs; stress details retain per-user gates and correlation.'}
    save_json(destination / 'export_manifest.json', report)
    return report
