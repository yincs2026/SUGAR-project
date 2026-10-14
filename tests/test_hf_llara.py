"""Exercises real Transformers+PEFT APIs with a randomly initialized tiny Qwen.
It does not download or evaluate any pretrained 7B model.
"""
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from test_core import config
from sugar.data import demo, prepare
from sugar.offline import build_snapshots
from sugar.features import FeatureStore
from sugar.model import SUGAR
from sugar.train import Collator, Queries

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ['transformers', 'peft', 'accelerate'])


@unittest.skipUnless(AVAILABLE, 'Install the llara extra for real Transformers/PEFT integration tests')
class HuggingFaceLLaRATests(unittest.TestCase):
    def test_real_api_lora_loss_and_adapter_checkpoint(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from sugar.llara import LLaRA, TinyTokenizer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = {**config(), 'model': 'llara', 'llara_backend': 'real',
                 'llara_llm_model': 'unit-test/tiny-random-qwen', 'llara_llm_revision': 'a' * 40,
                 'llara_dtype': 'float32', 'llara_lora_rank': 2, 'llara_lora_alpha': 4,
                 'llara_lora_dropout': 0, 'llara_history_items': 3, 'llara_text_tokens': 4,
                 'llara_text_epochs': 0, 'llara_curriculum_epochs': 1, 'llara_score_batch': 12,
                 'llara_gradient_checkpointing': True, 'llara_max_tokens': 256}
            demo(root / 'raw', users=6, items=12, events=25)
            data = prepare(root / 'raw', root / 'prepared', c)
            build_snapshots(data, root / 'artifacts', c, root / 'cache')
            store = FeatureStore(data, root / 'artifacts', c)
            base_c = {**c, 'model': 'sasrec'}
            backbone = SUGAR(12, store.dim, base_c, store.embeddings)
            torch.save({'config': base_c, 'model': backbone.state_dict(), 'data_fingerprint': data['fingerprint']}, root / 'sasrec.pt')
            c['llara_backbone_checkpoint'] = str(root / 'sasrec.pt')
            architecture = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=512, pad_token_id=0, eos_token_id=1)
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer(64)), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=lambda *a, **k: Qwen2ForCausalLM(architecture)):
                model = LLaRA(12, data, c, store, 'cpu')
            model.train()
            model.set_epoch(2)
            sequence, features, target, _ = Collator(data, store, c)(Queries(data, 'train').entries[8:10])
            h, _ = model(sequence, features)
            loss, _, _ = model.loss(h, target, target[:, None], target[:, None])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            adapter_grads = [p.grad for n, p in model.named_parameters() if 'lora_' in n]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in adapter_grads))
            self.assertTrue(any(p.grad is not None for p in model.projector.parameters()))
            state = model.checkpoint_state()
            self.assertFalse(any('backbone' in name for name in state))
            self.assertTrue(any('lora_' in name for name in state))
            model.load_checkpoint_state(state)
            model.eval()
            with torch.inference_mode():
                h, _ = model(sequence, features)
                scores = model.score_catalog(h, torch.arange(1, 13))
                correct = model.score_targets(h, target)
            self.assertEqual(scores.shape, (2, 12))
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores[torch.arange(2), target-1], correct, rtol=1e-4, atol=1e-4)
