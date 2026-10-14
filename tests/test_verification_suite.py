import json
import tempfile
import unittest
from pathlib import Path
from sugar.utils import load_json, save_json
from sugar.verification import verify_paper
from sugar.suite import run_suite


class VerificationTests(unittest.TestCase):
    def test_missing_results_never_pass(self):
        with tempfile.TemporaryDirectory() as root:
            report = verify_paper(root, Path(root) / 'report.json')
            self.assertFalse(report['numerical_match_at_reported_precision'])
            self.assertFalse(report['bitwise_original_experiment_reproduced'])
            self.assertEqual(len(report['blockers']), 3)

    def test_mock_even_matching_reference_never_passes(self):
        reference = load_json(Path(__file__).resolve().parents[1] / 'src/sugar/paper_reference.json')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for ds, counts in reference['counts'].items():
                save_json(root / ds / 'dataset_statistics.json', counts)
                for model, values in reference['baselines'][ds].items():
                    save_json(root / ds / 'models' / model / 'summary.json', {
                        'mock': True, 'runs': 5, 'metrics': {key: {'mean': value, 'std': 0} for key, value in values.items()}})
            report = verify_paper(root, root / 'report.json')
            self.assertFalse(report['numerical_match_at_reported_precision'])
            self.assertTrue(all(c['match'] for c in report['checks']))
            self.assertTrue(report['blockers'])

    def test_suite_unknown_variant_rejected(self):
        from sugar.suite import VARIANTS
        self.assertNotIn('invented-numbers', VARIANTS)

    def test_search_does_not_evaluate_test(self):
        from unittest.mock import patch
        import tempfile
        from test_core import config
        from sugar.data import demo, prepare
        from sugar.offline import build_snapshots
        from sugar.train import train, evaluate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = {**config(), 'epochs': 1}
            demo(root / 'raw', users=6, items=12, events=25)
            data = prepare(root / 'raw', root / 'prepared', c)
            build_snapshots(data, root / 'artifacts', c, root / 'cache')
            observed_splits = []
            def inspected(*args, **kwargs):
                split = args[4] if len(args) > 4 else kwargs.get('split', 'test')
                observed_splits.append(split)
                return evaluate(*args, **kwargs)
            with patch('sugar.train.evaluate', inspected):
                train(root / 'prepared/data.json', root / 'artifacts', root / 'run', c, evaluate_test=False)
            self.assertNotIn('test', observed_splits)
            self.assertFalse((root / 'run/test.json').exists())
