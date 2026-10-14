import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_core import config
from sugar.data import demo, prepare
from sugar.offline import build_snapshots, select_snapshot
from sugar.features import FeatureStore
from sugar.coverage import drift_percentiles, offline_costs, sweep_frontier
from sugar.utils import load_json, save_json


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.c = config()
        demo(cls.root / 'raw')
        cls.data = prepare(cls.root / 'raw', cls.root / 'prepared', cls.c)
        cls.path = cls.root / 'prepared/data.json'
        build_snapshots(cls.data, cls.root / 'artifacts', cls.c, cls.root / 'cache')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_three_ground_truth_percentiles(self):
        report = drift_percentiles([self.path], [self.root / 'artifacts'], self.root / 'percentiles.json')
        self.assertEqual(set(report['results']), {'85', '90', '95'})
        eps = [report['results'][str(p)]['epsilon'] for p in [85, 90, 95]]
        self.assertLessEqual(eps[0], eps[1])
        self.assertLessEqual(eps[1], eps[2])
        self.assertEqual(report['results']['85']['percentile'], 85)

    def test_cost_accounting_splits_modules_and_cache(self):
        report = offline_costs(self.root / 'artifacts', self.root / 'cost.json')
        self.assertEqual(report['catalog_calls'], 18)
        self.assertEqual(report['total_executed_tokens'], 0)
        self.assertTrue(report['mock'])
        self.assertGreater(report['wall_clock_measurement']['wall_seconds'], 0)
        self.assertIn('build', report['by_module_operation'])

    def test_future_interaction_injected_into_evidence_is_rejected(self):
        snapshots = load_json(self.root / 'artifacts/snapshots.json')
        for u, snaps in snapshots.items():
            if snaps:
                snap = copy.deepcopy(snaps[0])
                target_position = snap['commit_pos'] + 1
                snap['evidence_indices'].append(target_position)
                with self.assertRaises(AssertionError):
                    select_snapshot([snap], self.data['sequences'][u][target_position][1], target_position)

    def test_random_corruption_repeatability_and_temporal_validity(self):
        a = FeatureStore(self.data, self.root / 'artifacts', self.c, 'corrupted')
        b = FeatureStore(self.data, self.root / 'artifacts', self.c, 'corrupted')
        for u in self.data['sequences']:
            x, y = a.query(u, 42), b.query(u, 42)
            np.testing.assert_array_equal(x['profile'], y['profile'])
            self.assertLess(x['source_time'], x['target_time'])
        self.assertTrue(a.audit()['passed'])

    def test_frontier_uses_computed_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, calls, accuracy in [('tau-0.6', 1, .1), ('tau-0.7', 2, .09), ('xi-1.0', 3, .2)]:
                r = root / 'sweeps' / name
                save_json(r / 'run/config.json', {'threshold': 'global' if name.startswith('tau') else 'personalized', 'tau': .6, 'xi': 1})
                save_json(r / 'run/test.json', {'metrics': {'NDCG@10': accuracy}})
                save_json(r / 'artifacts/offline_log.json', [{'trigger': i < calls} for i in range(10)])
            report = sweep_frontier(root, root / 'frontier.json')
            self.assertEqual({p['configuration'] for p in report['threshold_pareto_front']}, {'tau-0.6', 'xi-1.0'})


class ProjectionCacheTests(unittest.TestCase):
    def test_identical_cached_side_branch_and_backbone_remains_live(self):
        from sugar.model import SUGAR
        torch.set_num_threads(1)
        c = config()
        model = SUGAR(20, 32, c, np.ones((21, 32), dtype=np.float32)).eval()
        features = {'profile': torch.randn(1, 32), 'present': torch.tensor([True]),
                    'n': torch.tensor([50.]), 'sim': torch.tensor([.8]), 'recent': torch.randn(1, 32),
                    'neighbors': torch.randn(1, 3, 32), 'neighbor_mask': torch.tensor([[True, True, False]]),
                    'side_key': torch.tensor([123])}
        with torch.inference_mode():
            a, _ = model(torch.tensor([[1, 2, 3]]), features)
            count = model.semantic_computations
            b, _ = model(torch.tensor([[1, 2, 3]]), features)
            self.assertEqual(model.semantic_computations, count)
            torch.testing.assert_close(a, b)
            h, _ = model(torch.tensor([[4, 5, 6]]), features)
            self.assertEqual(model.semantic_computations, count)
            self.assertFalse(torch.equal(h, a))
        model.train()
        self.assertFalse(model.semantic_cache)

    def test_no_cache_during_gradient_training(self):
        from sugar.model import SUGAR
        c = config()
        model = SUGAR(20, 32, c, np.ones((21, 32), dtype=np.float32))
        f = {'profile': torch.randn(1, 32), 'present': torch.tensor([True]), 'n': torch.tensor([5.]),
             'sim': torch.tensor([.5]), 'recent': torch.randn(1, 32), 'neighbors': torch.randn(1, 3, 32),
             'neighbor_mask': torch.ones(1, 3, dtype=torch.bool), 'side_key': torch.tensor([1])}
        h, _ = model(torch.tensor([[1, 2, 3]]), f)
        h.sum().backward()
        self.assertFalse(model.semantic_cache)
        self.assertIsNotNone(model.projection.weight.grad)


class PaperExportTests(unittest.TestCase):
    def test_exports_computed_scores_and_seed_status(self):
        from sugar.coverage import export_paper_tables, seed_outcomes
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = root / 'yelp'
            save_json(ds / 'dataset_statistics.json', {'users': 2, 'items': 5, 'interactions': 12, 'social_edges': 2})
            save_json(ds / 'models/sugar/summary.json', {'seeds': [42], 'mock': True,
                      'metrics': {'HR@10': {'mean': .321, 'std': None}}, 'buckets': {}})
            save_json(ds / 'models/sugar/seed-42/test.json', {'metrics': {'HR@10': .321}})
            report = export_paper_tables(root, root / 'tables')
            self.assertEqual(report['tables']['5'], 1)
            self.assertEqual(load_json(root / 'tables/table_5.json')[0]['mean'], .321)
            self.assertTrue(load_json(root / 'tables/table_5.json')[0]['mock'])
            self.assertEqual(seed_outcomes(ds, ds / 'seed_outcomes.json')['runs'][0]['status'], 'completed')

    def test_entity_fallback_uses_real_extractor_outputs(self):
        from sugar.semantics import item_fallback, validate, ITEM_SCHEMA
        descriptor = item_fallback('Reading #Books in Paris', ['Paris', 'Paris'])
        validate(descriptor, ITEM_SCHEMA)
        self.assertEqual(descriptor['topic'], '#Books')
        self.assertEqual(descriptor['subtopics'], ['Paris'])

    def test_research_suite_rejects_fixture_mode(self):
        from sugar.suite import run_suite
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {'research_only': True, 'datasets': [{'name': 'yelp', 'config': 'configs/paper.yaml', 'demo': True}]}
            path = root / 'spec.yaml'
            path.write_text(yaml.safe_dump(spec))
            with self.assertRaisesRegex(ValueError, 'requires real'):
                run_suite(path, root / 'run')
