import math
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


class SASBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-8)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, padding):
        length = x.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        query = self.attention_norm(x)
        attended, _ = self.attention(query, x, x, attn_mask=causal, key_padding_mask=padding, need_weights=False)
        x = query + attended
        x = self.ff_norm(x)
        x = x + self.ff(x)
        return x.masked_fill(padding.unsqueeze(-1), 0)


class SUGAR(nn.Module):
    def __init__(self, nitems, semantic_dim, config, semantic_embeddings=None):
        super().__init__()
        self.c, self.semantic_dim = config, semantic_dim
        self.semantic_cache = OrderedDict()
        self.semantic_computations = 0
        d = config["dim"]
        self.items = nn.Embedding(nitems + 1, d, padding_idx=0)
        self.positions = nn.Embedding(config["max_len"], d)
        self.input_dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList([SASBlock(d, config["heads"], config["dropout"]) for _ in range(config["blocks"])])
        self.final_norm = nn.LayerNorm(d, eps=1e-8)
        self.projection = nn.Linear(semantic_dim, d)
        self.alignment_projection = nn.Linear(semantic_dim, d, bias=False) if config.get('separate_alignment_projection', False) else None
        self.default_profile = nn.Parameter(torch.zeros(semantic_dim))
        self.social_gate = nn.Linear(3 * semantic_dim, 1)
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.expm1(config["social_temperature"]))))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.reliability_bias = nn.Parameter(torch.tensor(0.0))
        self.fusion_gate = nn.Linear(2 * d + 1, d)
        self.lm_input = nn.Linear(semantic_dim + d, d) if config["model"] == "lm-feature" else None
        if semantic_embeddings is not None:
            self.register_buffer("semantics", torch.as_tensor(semantic_embeddings.copy(), dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("semantics", torch.empty(0), persistent=False)
        nn.init.normal_(self.items.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode_sequence(self, sequence):
        padding = sequence.eq(0)
        x = self.items(sequence) * math.sqrt(self.c["dim"])
        if self.lm_input is not None:
            x = self.lm_input(torch.cat([x, self.semantics[sequence]], dim=-1))
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = self.input_dropout(x + self.positions(positions)[None, :, :])
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, padding)
        x = self.final_norm(x)
        last = sequence.ne(0).sum(1) - 1
        return x[torch.arange(sequence.shape[0], device=sequence.device), last]

    def forward(self, sequence, features):
        hseq = self.encode_sequence(sequence)
        batch = sequence.shape[0]
        if self.c["model"] in {"sasrec", "lm-feature"}:
            zero = hseq.new_zeros(batch)
            return hseq, {"rho": zero, "gamma": zero, "gate": zero}
        hprof, rho, gamma = self.semantic_branch(features)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([hseq, hprof, rho[:, None]], dim=-1)))
        if self.c["fixed_fusion"] is not None:
            gate = torch.full_like(gate, float(self.c["fixed_fusion"]))
        if self.c["force_gate"] is not None:
            gate = torch.full_like(gate, float(self.c["force_gate"]))
        fused = (1 - gate) * hseq + gate * hprof
        return fused, {"rho": rho, "gamma": gamma, "gate": gate.mean(-1)}

    def clear_semantic_cache(self):
        self.semantic_cache.clear()

    def train(self, mode=True):
        self.clear_semantic_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_semantic_cache()
        return super()._apply(fn)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_semantic_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def semantic_branch(self, features):
        use_cache = (not self.training and not torch.is_grad_enabled() and self.c.get('cache_semantic_projection', True)
                     and 'side_key' in features)
        if not use_cache:
            return self.compute_semantic_branch(features)
        keys = features['side_key'].tolist()
        missing = [i for i, key in enumerate(keys) if key not in self.semantic_cache]
        if missing:
            selected = torch.tensor(missing, dtype=torch.long, device=features['profile'].device)
            subset = {k: v.index_select(0, selected) for k, v in features.items()}
            computed = self.compute_semantic_branch(subset)
            for j, i in enumerate(missing):
                self.semantic_cache[keys[i]] = tuple(x[j].detach() for x in computed)
        results = tuple(torch.stack([self.semantic_cache[key][j] for key in keys]) for j in range(3))
        for key in keys:
            self.semantic_cache.move_to_end(key)
        while len(self.semantic_cache) > self.c.get('semantic_cache_size', 256):
            self.semantic_cache.popitem(last=False)
        return results

    def compute_semantic_branch(self, features):
        self.semantic_computations += len(features['profile'])
        raw = torch.where(features["present"][:, None], features["profile"], self.default_profile[None, :])
        neighbors, valid, recent = features["neighbors"], features["neighbor_mask"], features["recent"]
        affinity = F.cosine_similarity(recent[:, None, :], neighbors, dim=-1)
        beta = F.softplus(self.raw_beta) + 1e-8
        logits = (affinity / beta).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        social = (weights[:, :, None] * neighbors).sum(1)
        gamma = torch.sigmoid(self.social_gate(torch.cat([raw, social, recent], dim=-1))).squeeze(-1)
        gamma = gamma * valid.any(1)
        grounded = (1 - gamma[:, None]) * raw + gamma[:, None] * social
        hprof = self.projection(grounded)
        sat = features["n"] / (features["n"] + self.c["kappa"])
        rho = torch.sigmoid(self.a * features["sim"] + self.b * sat + self.reliability_bias)
        return hprof, rho, gamma

    def score_targets(self, h, item_ids):
        return (h * self.items(item_ids)).sum(-1)

    def score_catalog(self, h, item_ids):
        return h @ self.items(item_ids).T

    def scores(self, h, item_ids):
        return (h[:, None, :] * self.items(item_ids)).sum(-1)

    def loss(self, h, positive, negatives, alignment_negatives):
        pos = (h * self.items(positive)).sum(-1)
        neg = self.scores(h, negatives)
        if self.c["loss"] == "bpr":
            rec = F.softplus(neg - pos[:, None]).sum(1).mean()
        else:
            rec = (F.softplus(-pos) + F.softplus(neg).sum(1)).mean()
        align = rec.new_zeros(())
        if self.c["alignment_weight"] and self.c["model"] == "sugar":
            # Eq.23 uses W_p without b_p, shared with profile projection.
            anchor = F.normalize(self.items(positive), dim=-1)
            candidates = torch.cat([positive[:, None], alignment_negatives], dim=1)
            projected = F.normalize(F.linear(self.semantics[candidates], (self.alignment_projection.weight if self.alignment_projection is not None else self.projection.weight)), dim=-1)
            logits = (anchor[:, None, :] * projected).sum(-1) / self.c["alignment_temperature"]
            align = F.cross_entropy(logits, torch.zeros(len(positive), dtype=torch.long, device=h.device))
        return rec + self.c["alignment_weight"] * align, rec.detach(), align.detach()
