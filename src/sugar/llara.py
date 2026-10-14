"""LLaRA hybrid text/ID prompts and curriculum LoRA training.
Full-catalog conditional likelihood adapts the original candidate-generation evaluator.
The tiny backend is an explicitly synthetic infrastructure test, never a real baseline.
"""
import hashlib
import math
import re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .model import SUGAR
from .utils import file_hash


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def __init__(self, vocab):
        self.vocab = vocab
    def encode(self, text, add_special_tokens=False):
        words = re.findall(r'\w+|[^\s\w]', text)
        return [2 + int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % (self.vocab - 2) for w in words] or [2]


class TinyDecoder(nn.Module):
    def __init__(self, vocab, hidden, rank):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, vocab)
        for p in self.parameters():
            p.requires_grad_(False)
        self.adapter_down = nn.Linear(hidden, rank, bias=False)
        self.adapter_up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
    def get_input_embeddings(self):
        return self.embedding
    def logits(self, embeddings, attention_mask):
        x = embeddings + self.adapter_up(self.adapter_down(embeddings))
        output, _ = self.gru(x)
        return self.output(output)


class LLaRA(nn.Module):
    def __init__(self, nitems, data, config, store, device):
        super().__init__()
        self.c, self.data, self.nitems = config, data, nitems
        self.mock = config.get('llara_backend', 'real') == 'tiny'
        if self.mock and not config.get('allow_nonpaper', False):
            raise ValueError('Tiny LLaRA requires allow_nonpaper=true; it is not a pretrained LLM')
        baseconfig = {**config, 'model': 'sasrec', 'alignment_weight': 0}
        self.backbone = SUGAR(nitems, store.dim, baseconfig, store.embeddings)
        self.backbone_hash = None
        path = config.get('llara_backbone_checkpoint')
        if path:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('data_fingerprint') != data['fingerprint'] or checkpoint['config']['model'] != 'sasrec':
                raise ValueError('LLaRA requires a SASRec checkpoint trained on this exact mapped dataset')
            self.backbone = SUGAR(nitems, store.dim, checkpoint['config'], store.embeddings)
            self.backbone.load_state_dict(checkpoint['model'])
            self.backbone_hash = file_hash(path)
        elif not self.mock:
            raise ValueError('Train SASRec first, then set llara_backbone_checkpoint to its best.pt')
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        if self.mock:
            hidden = config.get('llara_test_hidden', 32)
            self.tokenizer = TinyTokenizer(config.get('llara_test_vocab', 64))
            self.llm = TinyDecoder(self.tokenizer.vocab, hidden, config.get('llara_lora_rank', 8))
        else:
            revision = config.get('llara_llm_revision')
            if not re.fullmatch(r'[0-9a-f]{40}', str(revision)):
                raise ValueError('Resolve llara_llm_revision to an immutable model commit before training')
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
            dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[config.get('llara_dtype', 'bfloat16')]
            if str(device) == 'cpu' and dtype != torch.float32:
                raise ValueError('Real LLaRA on CPU requires llara_dtype=float32 and sufficient RAM')
            self.tokenizer = AutoTokenizer.from_pretrained(config['llara_llm_model'], revision=revision)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config['llara_llm_model'], revision=revision,
                                                        torch_dtype=dtype, attn_implementation='eager')
            model.config.use_cache = False
            adapter = LoraConfig(task_type='CAUSAL_LM', r=config.get('llara_lora_rank', 8),
                                 lora_alpha=config.get('llara_lora_alpha', 16),
                                 lora_dropout=config.get('llara_lora_dropout', 0.05),
                                 target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            self.llm = get_peft_model(model, adapter)
            if config.get('llara_gradient_checkpointing', True):
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                self.llm.enable_input_require_grads()
            hidden = model.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(self.backbone.items.embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.epoch = 0
        self.item_tokens = {v: self.tokenizer.encode(self.data['catalog'][str(v)]['text'], add_special_tokens=False)[:config.get('llara_text_tokens', 32)]
                            for v in range(1, nitems + 1)}
        self.head_tokens = self.tokenizer.encode('Predict the next item from the chronological interaction history. History: ', add_special_tokens=False)
        self.answer_prefix = self.tokenizer.encode(' Next item: ', add_special_tokens=False)
        self.separator = self.tokenizer.encode(' ; ', add_special_tokens=False)
        self.items = self.backbone.items

    def provenance(self):
        return {'model': 'LLaRA', 'implementation': 'hybrid-prompt curriculum, full-catalog likelihood adaptation',
                'source_reference': 'ljy0ustc/LLaRA@9733f732399e95e236f4f2e1d1139c1f0437090e',
                'llm_model': self.c.get('llara_llm_model'), 'llm_revision': self.c.get('llara_llm_revision'),
                'backbone_sha256': self.backbone_hash, 'mock': self.mock,
                'score': 'conditional log-likelihood of mapped item ID and item text, optionally length normalized'}

    def checkpoint_state(self):
        # Immutable frozen base is reloaded by revision; only learned deltas are saved.
        parameters = dict(self.named_parameters())
        return {name: value for name, value in self.state_dict().items() if name in parameters and parameters[name].requires_grad}

    def load_checkpoint_state(self, state):
        required = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != required:
            raise ValueError('LLaRA trainable checkpoint keys differ from model configuration')
        self.load_state_dict(state, strict=False)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def hybrid_probability(self):
        if not self.training:
            return 1.0
        warm = self.c.get('llara_text_epochs', 1)
        return max(0.0, min(1.0, (self.epoch - warm + 1) / max(1, self.c.get('llara_curriculum_epochs', 5))))

    def forward(self, sequence, features):
        self._histories = [row[row.ne(0)].tolist()[-self.c.get('llara_history_items', 20):] for row in sequence]
        self._prompt_embeds = [self.make_prompt(history) for history in self._histories]
        h = self.backbone.encode_sequence(sequence)
        zero = h.new_zeros(len(h))
        return h, {'rho': zero, 'gamma': zero, 'gate': zero}

    def tokens_embed(self, ids):
        device = self.projector[0].weight.device
        return self.llm.get_input_embeddings()(torch.tensor(ids, dtype=torch.long, device=device))

    def make_prompt(self, history):
        pieces = [self.tokens_embed(self.head_tokens)]
        probability = self.hybrid_probability()
        for item in history:
            pieces.append(self.tokens_embed(self.item_tokens[item]))
            if probability == 1 or (probability > 0 and torch.rand(()).item() < probability):
                with torch.no_grad():
                    cf = self.backbone.items.weight[item]
                projected = self.projector(cf.float()).to(pieces[0].dtype)
                pieces.append(projected[None])
            pieces.append(self.tokens_embed(self.separator))
        pieces.append(self.tokens_embed(self.answer_prefix))
        return torch.cat(pieces, dim=0)

    def answer_ids(self, item):
        # Mapped ID disambiguates equal item texts in full-catalog ranking.
        ids = self.tokenizer.encode(f'item {item}: ', add_special_tokens=False) + self.item_tokens[item]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.append(eos)
        return ids

    def pair_nll(self, prompts, items):
        embeddings, labels = [], []
        for prompt, item in zip(prompts, items):
            ids = self.answer_ids(int(item))
            answer = self.tokens_embed(ids)
            if len(prompt) + len(answer) > self.c.get('llara_max_tokens', 2048):
                raise ValueError('LLaRA context overflow: reduce explicit history/text lengths; no silent truncation')
            embeddings.append(torch.cat([prompt, answer]))
            labels.append(torch.cat([torch.full((len(prompt),), -100, dtype=torch.long, device=prompt.device),
                                     torch.tensor(ids, dtype=torch.long, device=prompt.device)]))
        x = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        y = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        lengths = torch.tensor([len(t) for t in labels], device=x.device)
        attention = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        if self.mock:
            logits = self.llm.logits(x, attention)
        else:
            logits = self.llm(inputs_embeds=x, attention_mask=attention.long(), use_cache=False).logits
        token_loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), y[:, 1:].flatten(),
                                     reduction='none', ignore_index=-100).view(len(items), -1)
        nll = token_loss.sum(1)
        if self.c.get('llara_length_normalize', False):
            nll = nll / y[:, 1:].ne(-100).sum(1).clamp_min(1)
        return nll

    def score_targets(self, h, target):
        return -self.pair_nll(self._prompt_embeds, target.tolist())

    def score_catalog(self, h, ids):
        scores = []
        batch = self.c.get('llara_score_batch', 4)
        # All items are evaluated, never a sampled candidate shortlist.
        for prompt in self._prompt_embeds:
            parts = []
            for start in range(0, len(ids), batch):
                chunk = ids[start:start + batch].tolist()
                valid = [v for v in chunk if 1 <= v <= self.nitems]
                values = -self.pair_nll([prompt] * len(valid), valid) if valid else h.new_empty(0)
                iterator = iter(values)
                parts.append(torch.stack([next(iterator) if 1 <= v <= self.nitems else h.new_tensor(-torch.inf) for v in chunk]))
            scores.append(torch.cat(parts))
        return torch.stack(scores)

    def loss(self, h, positive, negatives, alignment_negatives):
        rec = self.pair_nll(self._prompt_embeds, positive.tolist()).mean()
        return rec, rec.detach(), rec.new_zeros(())
