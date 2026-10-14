import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_core import config
from sugar.data import demo, prepare
from sugar.offline import build_snapshots
from sugar.features import FeatureStore
from sugar.train import train, load_run, evaluate, Collator, Queries
from sugar.factory import build_model, MODELS
from sugar.baselines import normalized_graph, propagate
from sugar.coherence import encoder_quality, static_baseline_profiles
from sugar.utils import load_json


class RemainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.c = {**config(), 'epochs': 1, 'batch_size': 32, 'llara_backend': 'tiny',
                 'llara_test_hidden': 16, 'llara_test_vocab': 40, 'llara_history_items': 3,
                 'llara_text_tokens': 4, 'llara_score_batch': 20, 'llara_text_epochs': 0,
                 'llara_curriculum_epochs': 1, 'llara_max_tokens': 256}
        demo(cls.root / 'raw', users=6, items=12, events=25)
        cls.data = prepare(cls.root / 'raw', cls.root / 'prepared', cls.c)
        cls.path = cls.root / 'prepared/data.json'
        build_snapshots(cls.data, cls.root / 'artifacts', cls.c, cls.root / 'cache')
        static_baseline_profiles(cls.path, cls.root / 'artifacts', cls.c, cls.root / 'cache')
        cls.store = FeatureStore(cls.data, cls.root / 'artifacts', cls.c)
        cls.batch = Collator(cls.data, cls.store, cls.c)(Queries(cls.data, 'train').entries[8:12])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_models_forward_loss_and_gradient(self):
        for name in MODELS:
            with self.subTest(model=name):
                c = {**self.c, 'model': name}
                model = build_model(self.data, self.store, c)
                if hasattr(model, 'set_epoch'):
                    model.set_epoch(2)
                model.train()
                sequence, features, target, _ = self.batch
                h, _ = model(sequence, features)
                negatives = torch.tensor([[4, 5], [6, 7], [8, 9], [10, 11]])
                loss, _, _ = model.loss(h, target, negatives, negatives)
                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                model.eval()
                with torch.inference_mode():
                    h, _ = model(sequence, features)
                    scores = model.score_catalog(h, torch.arange(1, 13))
                    positive = model.score_targets(h, target)
                self.assertEqual(scores.shape, (4, 12))
                self.assertTrue(torch.isfinite(scores).all())
                torch.testing.assert_close(scores[torch.arange(4), target-1], positive, rtol=1e-4, atol=1e-4)

    def test_all_new_models_train_checkpoint_reload_full_catalog(self):
        for name in ['gru4rec', 'caser', 'bert4rec', 'socialmf', 'lightgcn-plus', 'rlmrec', 'llara']:
            with self.subTest(model=name):
                c = {**self.c, 'model': name}
                out = self.root / ('run-' + name)
                result = train(self.path, self.root / 'artifacts', out, c)
                self.assertTrue(result['mock'])
                model, data, store, saved = load_run(self.path, self.root / 'artifacts', out)
                again = evaluate(model, data, store, saved)
                self.assertEqual([r['rank'] for r in result['users']], [r['rank'] for r in again['users']])

    def test_bert_masked_training_does_not_access_holdouts(self):
        c = {**self.c, 'model': 'bert4rec'}
        model = build_model(self.data, self.store, c)
        captured = []
        original = model.masked_loss
        def inspect(full):
            captured.append(full.clone())
            return original(full)
        model.masked_loss = inspect
        h, _ = model(self.batch[0], self.batch[1])
        target = self.batch[2]
        model.loss(h, target, target[:, None], target[:, None])
        for row, source, t in zip(captured[0], self.batch[0], target):
            self.assertEqual(row[row.ne(0)].tolist(), source[source.ne(0)].tolist() + [int(t)])

    def test_llara_base_frozen_and_adapter_projector_trainable(self):
        model = build_model(self.data, self.store, {**self.c, 'model': 'llara'})
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertFalse(model.llm.embedding.weight.requires_grad)
        self.assertTrue(model.llm.adapter_up.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.projector.parameters()))

    def test_llara_curriculum(self):
        model = build_model(self.data, self.store, {**self.c, 'model': 'llara', 'llara_text_epochs': 2, 'llara_curriculum_epochs': 4})
        model.train()
        model.set_epoch(0)
        self.assertEqual(model.hybrid_probability(), 0)
        model.set_epoch(3)
        self.assertEqual(model.hybrid_probability(), 0.5)
        model.eval()
        self.assertEqual(model.hybrid_probability(), 1)

    def test_separate_projection_has_own_gradient(self):
        model = build_model(self.data, self.store, {**self.c, 'model': 'sugar', 'separate_alignment_projection': True})
        h, _ = model(self.batch[0], self.batch[1])
        negatives = torch.tensor([[4, 5], [6, 7], [8, 9], [10, 11]])
        loss, _, _ = model.loss(h, self.batch[2], negatives, negatives)
        loss.backward()
        self.assertGreater(model.alignment_projection.weight.grad.abs().sum(), 0)
        self.assertGreater(model.projection.weight.grad.abs().sum(), 0)

    def test_sampling_reproducible(self):
        c = {**self.c, 'neighbor_sample': 1}
        a = FeatureStore(self.data, self.root / 'artifacts', c).query('1', 20)
        b = FeatureStore(self.data, self.root / 'artifacts', c).query('1', 20)
        np.testing.assert_array_equal(a['neighbors'], b['neighbors'])
        self.assertLessEqual(a['neighbor_mask'].sum(), 1)

    def test_ari_reports_computed_value(self):
        report = encoder_quality(self.path, self.root / 'artifacts', self.root / 'coherence.json')
        self.assertTrue(report['mock'])
        self.assertEqual(report['items'], 12)
        self.assertTrue(-1 <= report['adjusted_rand_index'] <= 1)

    def test_static_profiles_count_all_training_events(self):
        profiles = np.load(self.root / 'artifacts/static_user_embeddings.npy', allow_pickle=False)
        self.assertEqual(profiles.shape, (7, self.store.dim))
        self.assertTrue((np.linalg.norm(profiles[1:], axis=1) > 0.99).all())

    def test_normalized_graph_equation(self):
        graph = normalized_graph(4, [(1, 2), (1, 3)]).to_dense()
        self.assertAlmostEqual(float(graph[1, 2]), 1 / np.sqrt(2), places=6)
        self.assertEqual(float(graph[0].sum()), 0)

    def test_frozen_semantic_provenance_cannot_change_on_restore(self):
        from sugar.train import restore_model
        model = build_model(self.data, self.store, {**self.c, 'model': 'rlmrec'})
        provenance = copy.deepcopy(model.provenance())
        state = model.state_dict()
        model.semantic_input_hashes['users'] = 'changed'
        with self.assertRaises(ValueError):
            restore_model(model, state, provenance)

    def test_generative_rlm_reconstruction_gradients(self):
        c = {**self.c, 'model': 'rlmrec', 'rlm_variant': 'generative', 'rlm_mask_ratio': 0.8}
        model = build_model(self.data, self.store, c)
        h, _ = model(self.batch[0], self.batch[1])
        negatives = torch.tensor([[4, 5], [6, 7], [8, 9], [10, 11]])
        loss, _, _ = model.loss(h, self.batch[2], negatives, negatives)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.decoder.parameters()))
        self.assertIsNotNone(model.mask_token.grad)


if __name__ == '__main__':
    unittest.main()
