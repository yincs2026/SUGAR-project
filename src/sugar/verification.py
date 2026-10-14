"""Numerical reproduction verification. Reported values never enter any model/loss."""
from pathlib import Path
from .utils import load_json, save_json


def verify_paper(suite, out, tolerance=0.00005):
    root = Path(suite)
    reference = load_json(Path(__file__).resolve().parent / 'paper_reference.json')
    checks, blockers = [], []
    datasets = ['yelp', 'douban-book', 'epinions']
    for dataset in datasets:
        directory = root / dataset
        stats = directory / 'dataset_statistics.json'
        if not stats.exists():
            blockers.append(f'Missing dataset: {dataset}')
            continue
        observed_stats = load_json(stats)
        for key, expected in reference['counts'][dataset].items():
            actual = observed_stats.get(key)
            checks.append({'dataset': dataset, 'type': 'count', 'name': key, 'reported': expected,
                           'observed': actual, 'match': actual == expected})
        for model, reported in reference['baselines'][dataset].items():
            file = directory / 'models' / model / 'summary.json'
            if not file.exists():
                blockers.append(f'Missing results: {dataset}/{model}')
                continue
            summary = load_json(file)
            if summary.get('mock'):
                blockers.append(f'Synthetic artifacts are not reproduction evidence: {dataset}/{model}')
            if summary['runs'] != 5:
                blockers.append(f'{dataset}/{model} has {summary["runs"]} runs; paper main protocol states five')
            for metric, expected in reported.items():
                actual = summary['metrics'][metric]['mean']
                checks.append({'dataset': dataset, 'type': 'metric', 'model': model, 'name': metric,
                               'reported': expected, 'observed': actual,
                               'difference': actual - expected, 'match': abs(actual - expected) <= tolerance})
    numeric_match = bool(checks) and all(x['match'] for x in checks) and not blockers
    report = {'numerical_match_at_reported_precision': numeric_match,
              'bitwise_original_experiment_reproduced': False,
              'scope': 'Table 4 counts, Table 5 means, Table 9 means; not every prose claim/other table',
              'tolerance': tolerance, 'checks': checks, 'blockers': blockers,
              'original_provenance_status': 'Original input hashes, caches and model commits have not been supplied'}
    save_json(out, report)
    return report
