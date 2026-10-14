"""Independent, configurable baseline implementations under the SUGAR data protocol.
They are research adaptations, not byte-identical copies of original author software.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import SUGAR


def normalized_graph(size, edges, symmetric=True):
    pairs = set((int(a), int(b)) for a, b in edges if a != b)
    if symmetric:
        pairs |= {(b, a) for a, b in pairs}
    pairs = sorted(pairs)
    if not pairs:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (size, size)).coalesce()
    indices = torch.tensor(pairs, dtype=torch.long).T
    degree = torch.bincount(indices[0], minlength=size).float()
    values = (degree[indices[0]].clamp_min(1) * degree[indices[1]].clamp_min(1)).rsqrt()
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()


def social_graph(data):
    return normalized_graph(len(data['users']) + 1,
                            [(int(u), int(w)) for u, neighbors in data['graph'].items() for w in neighbors])


def interaction_graph(data):
    nusers = len(data['users']) + 1
    pairs = [(int(u), nusers + v) for u, seq in data['sequences'].items() for v, _ in seq[:-2]]
    return normalized_graph(nusers + len(data['items']) + 1, pairs)


def propagate(graph, embeddings, layers):
    outputs = [embeddings]
    for _ in range(layers):
        embeddings = torch.sparse.mm(graph, embeddings)
        outputs.append(embeddings)
    return torch.stack(outputs).mean(0)


class RankingModel(nn.Module):
    def __init__(self, nitems, config, output_dim=None):
        super().__init__()
        self.c = config
        self.items = nn.Embedding(nitems + 1, output_dim or config['dim'], padding_idx=0)
        nn.init.normal_(self.items.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()
        self._aux = self.items.weight.new_zeros(())

    def score_targets(self, h, target):
        return (h * self.items(target)).sum(-1)

    def score_catalog(self, h, ids):
        return h @ self.items(ids).T

    def scores(self, h, ids):
        return (h[:, None] * self.items(ids)).sum(-1)

    def diagnostics(self, h):
        zero = h.new_zeros(len(h))
        return {'rho': zero, 'gamma': zero, 'gate': zero}

    def loss(self, h, positive, negatives, alignment_negatives):
        pos, neg = self.score_targets(h, positive), self.scores(h, negatives)
        if self.c.get('loss') == 'bpr':
            rec = F.softplus(neg - pos[:, None]).sum(1).mean()
        elif self.c.get('loss') == 'bpr-max':
            weights = F.softmax(neg, dim=-1)
            rec = -(weights * torch.sigmoid(pos[:, None] - neg)).sum(-1).clamp_min(1e-12).log().mean()
            rec = rec + self.c.get('bpr_max_regularization', 1.0) * (weights * neg.square()).sum(-1).mean()
        else:
            rec = (F.softplus(-pos) + F.softplus(neg).sum(1)).mean()
        return rec + self._aux, rec.detach(), self._aux.detach()

    def prepare_loss(self, positive):
        pass


class GRU4Rec(RankingModel):
    def __init__(self, nitems, config):
        super().__init__(nitems, config)
        d = config['dim']
        self.input_dropout = nn.Dropout(config['dropout'])
        self.gru = nn.GRU(d, d, num_layers=config.get('gru_layers', 1), batch_first=True,
                          dropout=config['dropout'] if config.get('gru_layers', 1) > 1 else 0)
        self.output_dropout = nn.Dropout(config['dropout'])

    def forward(self, sequence, features):
        lengths = sequence.ne(0).sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.input_dropout(self.items(sequence)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        h = self.output_dropout(hidden[-1])
        return h, self.diagnostics(h)


class Caser(RankingModel):
    def __init__(self, nitems, nusers, config):
        d = config['dim']
        super().__init__(nitems, config, 2 * d)
        self.sequence_items = nn.Embedding(nitems + 1, d, padding_idx=0)
        self.users = nn.Embedding(nusers + 1, d, padding_idx=0)
        self.length = config.get('caser_len', 5)
        nh, nv = config.get('caser_horizontal', 16), config.get('caser_vertical', 4)
        self.horizontal = nn.ModuleList([nn.Conv2d(1, nh, (height, d)) for height in range(1, self.length + 1)])
        self.vertical = nn.Conv2d(1, nv, (self.length, 1))
        self.fc = nn.Linear(nh * self.length + nv * d, d)
        self.drop = nn.Dropout(config['dropout'])
        self.item_bias = nn.Embedding(nitems + 1, 1)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.sequence_items.weight, std=0.02)
        nn.init.normal_(self.users.weight, std=0.02)
        with torch.no_grad():
            self.sequence_items.weight[0].zero_()
            self.users.weight[0].zero_()

    def forward(self, sequence, features):
        # Right-padded prefixes converted into the last L actual events, left-padded for CNN.
        length = sequence.ne(0).sum(1)
        offsets = torch.arange(self.length, device=sequence.device)[None] + length[:, None] - self.length
        ids = sequence.gather(1, offsets.clamp_min(0).clamp_max(sequence.shape[1] - 1))
        ids = ids.masked_fill(offsets < 0, 0)
        x = self.sequence_items(ids).unsqueeze(1)
        vertical = self.vertical(x).flatten(1)
        horizontal = [F.relu(conv(x)).squeeze(-1).amax(-1) for conv in self.horizontal]
        encoded = F.relu(self.fc(self.drop(torch.cat([vertical] + horizontal, dim=-1))))
        h = torch.cat([encoded, self.users(features['user_id'])], dim=-1)
        return h, self.diagnostics(h)

    def score_targets(self, h, target):
        return super().score_targets(h, target) + self.item_bias(target).squeeze(-1)

    def score_catalog(self, h, ids):
        return super().score_catalog(h, ids) + self.item_bias(ids).squeeze(-1)[None]

    def scores(self, h, ids):
        return super().scores(h, ids) + self.item_bias(ids).squeeze(-1)


class BERT4Rec(RankingModel):
    def __init__(self, nitems, config):
        super().__init__(nitems, config)
        d = config['dim']
        self.nitems = nitems
        self.mask_id = nitems + 1
        self.mask_embedding = nn.Parameter(torch.randn(d) * 0.02)
        self.position = nn.Embedding(config['max_len'], d)
        self.input_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config['dropout'])
        layer = nn.TransformerEncoderLayer(d, config['heads'], dim_feedforward=4 * d,
                                           dropout=config['dropout'], activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config['blocks'], enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.bias = nn.Parameter(torch.zeros(nitems + 1))

    def contextual(self, ids):
        masks = ids.eq(self.mask_id)
        x = self.items(ids.clamp_max(self.nitems))
        x = torch.where(masks[:, :, None], self.mask_embedding[None, None], x)
        x = self.dropout(self.input_norm(x + self.position(torch.arange(ids.shape[1], device=ids.device))[None]))
        return self.output(self.encoder(x, src_key_padding_mask=ids.eq(0)))

    def forward(self, sequence, features):
        # Validation/test append one MASK; no target information supplied.
        lengths = sequence.ne(0).sum(1)
        rows = []
        for row, length in zip(sequence, lengths):
            observed = row[:int(length)][-(self.c['max_len'] - 1):]
            rows.append(torch.cat([observed, observed.new_tensor([self.mask_id])]))
        ids = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        x = self.contextual(ids)
        last = ids.ne(0).sum(1) - 1
        h = x[torch.arange(len(ids), device=ids.device), last]
        self._training_sequence = sequence
        return h, self.diagnostics(h)

    def masked_loss(self, full_sequence):
        real = full_sequence.ne(0)
        choose = (torch.rand_like(full_sequence.float()) < self.c.get('bert_mask_probability', 0.2)) & real
        # Ensure at least one selected item per row, independently of held-out labels.
        choose[torch.arange(len(full_sequence), device=full_sequence.device), real.sum(1) - 1] = True
        ids = full_sequence.clone()
        random = torch.rand_like(full_sequence.float())
        ids[choose & (random < 0.8)] = self.mask_id
        replacements = torch.randint(1, self.nitems + 1, full_sequence.shape, device=ids.device)
        ids[choose & (random >= 0.8) & (random < 0.9)] = replacements[choose & (random >= 0.8) & (random < 0.9)]
        x = self.contextual(ids)[choose]
        logits = x @ self.items.weight.T + self.bias
        logits[:, 0] = -torch.inf
        return F.cross_entropy(logits, full_sequence[choose])

    def loss(self, h, positive, negatives, alignment_negatives):
        sequence = self._training_sequence
        rows = [torch.cat([row[row.ne(0)], positive[i:i+1]])[-self.c['max_len']:]
                for i, row in enumerate(sequence)]
        full = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        rec = self.masked_loss(full)
        return rec, rec.detach(), rec.new_zeros(())

    def score_targets(self, h, target):
        return super().score_targets(h, target) + self.bias[target]

    def score_catalog(self, h, ids):
        return super().score_catalog(h, ids) + self.bias[ids][None]


class SocialMF(RankingModel):
    def __init__(self, nitems, data, config):
        super().__init__(nitems, config)
        self.users = nn.Embedding(len(data['users']) + 1, config['dim'], padding_idx=0)
        nn.init.normal_(self.users.weight, std=0.02)
        # Directed row-normalized trust weights, exactly the neighborhood-mean penalty.
        pairs = sorted((int(u), int(w)) for u, neighbors in data['graph'].items() for w in neighbors)
        size = len(data['users']) + 1
        indices = torch.tensor(pairs, dtype=torch.long).T if pairs else torch.empty((2, 0), dtype=torch.long)
        degree = torch.bincount(indices[0], minlength=size).float()
        values = degree[indices[0]].reciprocal() if pairs else torch.empty(0)
        self.register_buffer('trust', torch.sparse_coo_tensor(indices, values, (size, size)).coalesce(), persistent=False)
        self.register_buffer('has_neighbors', degree.gt(0), persistent=False)

    def forward(self, sequence, features):
        uid = features['user_id']
        h = self.users(uid)
        self._aux = h.new_zeros(())
        if self.training:
            unique = torch.unique(uid)
            unique = unique[self.has_neighbors[unique]]
            if len(unique):
                mean = torch.sparse.mm(self.trust, self.users.weight)[unique]
                self._aux = self.c.get('social_regularization', 0.1) * (self.users(unique) - mean).square().sum(-1).mean()
            self._aux = self._aux + self.c.get('embedding_regularization', 1e-5) * h.square().sum(-1).mean()
        return h, self.diagnostics(h)


class LightGCNPlus(SUGAR):
    def __init__(self, nitems, semantic_dim, config, semantics, data):
        super().__init__(nitems, semantic_dim, config, semantics)
        self.user_embeddings = nn.Embedding(len(data['users']) + 1, config['dim'], padding_idx=0)
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        self.register_buffer('social_graph', social_graph(data), persistent=False)
        self.late_fusion = nn.Linear(2 * config['dim'], config['dim'])

    def forward(self, sequence, features):
        hseq = self.encode_sequence(sequence)
        socially_propagated = propagate(self.social_graph, self.user_embeddings.weight, self.c.get('graph_layers', 3))
        h = self.late_fusion(torch.cat([hseq, socially_propagated[features['user_id']]], dim=-1))
        zero = h.new_zeros(len(h))
        return h, {'rho': zero, 'gamma': zero, 'gate': zero}


class RLMRec(RankingModel):
    def __init__(self, nitems, data, config, semantics, user_semantics):
        super().__init__(nitems, config)
        self.nusers = len(data['users']) + 1
        self.users = nn.Embedding(self.nusers, config['dim'], padding_idx=0)
        nn.init.normal_(self.users.weight, std=0.02)
        self.register_buffer('graph', interaction_graph(data), persistent=False)
        self.register_buffer('item_semantics', torch.as_tensor(semantics.copy(), dtype=torch.float32), persistent=False)
        self.register_buffer('user_semantics', torch.as_tensor(user_semantics.copy(), dtype=torch.float32), persistent=False)
        de = semantics.shape[1]
        d = config['dim']
        self.projector = nn.Sequential(nn.Linear(de, (de + d) // 2), nn.LeakyReLU(), nn.Linear((de + d) // 2, d))
        self.decoder = nn.Sequential(nn.Linear(d, (de + d) // 2), nn.LeakyReLU(), nn.Linear((de + d) // 2, de)) if config.get('rlm_variant', 'contrastive') == 'generative' else None
        self.mask_token = nn.Parameter(torch.zeros(d)) if self.decoder is not None else None
        nn.init.xavier_uniform_(self.users.weight)
        nn.init.xavier_uniform_(self.items.weight)
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self._eval_embeddings = None

    def provenance(self):
        return {"model": "RLMRec", "variant": self.c.get("rlm_variant", "contrastive"),
                "semantic_input_hashes": self.semantic_input_hashes,
                "source_reference": "HKUDS/RLMRec@22413752246de3dee8ab0d509f7f7a8889080f95"}

    def train(self, mode=True):
        self._eval_embeddings = None
        return super().train(mode)

    def encode(self):
        if not self.training and self._eval_embeddings is not None:
            return self._eval_embeddings
        joined = torch.cat([self.users.weight, self.items.weight])
        self._masked_indices = None
        if self.training and self.decoder is not None:
            # RLMRec-Gen: randomly mask CF representations before neighborhood propagation.
            mask = torch.rand(len(joined), device=joined.device) < self.c.get('rlm_mask_ratio', 0.2)
            mask[0] = False
            mask[self.nusers] = False
            self._masked_indices = torch.where(mask)[0]
            joined = torch.where(mask[:, None], self.mask_token[None], joined)
        encoded = propagate(self.graph, joined, self.c.get('graph_layers', 3)) * (self.c.get('graph_layers', 3) + 1)
        users, items = encoded[:self.nusers], encoded[self.nusers:]
        if not self.training:
            self._eval_embeddings = (users, items)
        return users, items

    def forward(self, sequence, features):
        users, items = self.encode()
        self._current_users, self._current_items = users, items
        self._uids = features['user_id']
        h = users[features['user_id']]
        self._aux = self.c.get('embedding_regularization', 1e-5) * h.square().sum(-1).mean() if self.training else h.new_zeros(())
        return h, self.diagnostics(h)

    def _contrastive(self, anchors, positive, pool):
        anchors = F.normalize(anchors, dim=-1)
        positive = F.normalize(positive, dim=-1)
        pool = F.normalize(pool, dim=-1)
        temperature = self.c.get('rlm_temperature', 0.2)
        correct = (anchors * positive).sum(-1) / temperature
        denominator = torch.logsumexp(anchors @ pool.T / temperature, dim=-1)
        return (denominator - correct).mean()

    def loss(self, h, positive, negatives, alignment_negatives):
        aux = self._aux
        if self.training:
            if self.decoder is None:
                user_profiles = self.projector(self.user_semantics[1:])
                own = user_profiles[self._uids - 1]
                valid = self.user_semantics[self._uids].norm(dim=-1) > 1e-12
                user_loss = self._contrastive(h[valid], own[valid], user_profiles) if valid.any() else h.new_zeros(())
                pos_sem = self.projector(self.item_semantics[positive])
                neg_ids = negatives.flatten()
                neg_sem = self.projector(self.item_semantics[neg_ids])
                aux = aux + self.c.get('rlm_alignment_weight', 0.1) * (
                    user_loss + self._contrastive(self._current_items[positive], pos_sem, pos_sem)
                    + self._contrastive(self._current_items[neg_ids], neg_sem, neg_sem))
            elif self._masked_indices is not None and len(self._masked_indices):
                encoded = torch.cat([self._current_users, self._current_items])[self._masked_indices]
                profiles = torch.cat([self.user_semantics, self.item_semantics])[self._masked_indices]
                valid = profiles.norm(dim=-1) > 1e-12
                if valid.any():
                    predicted = F.normalize(self.decoder(encoded[valid]), dim=-1)
                    actual = F.normalize(profiles[valid], dim=-1)
                    logits = predicted @ actual.T / self.c.get('rlm_temperature', 0.2)
                    aux = aux + self.c.get('rlm_alignment_weight', 0.1) * F.cross_entropy(logits, torch.arange(len(predicted), device=h.device))
        self._aux = aux
        return super().loss(h, positive, negatives, alignment_negatives)

    def score_targets(self, h, target):
        return (h * self._current_items[target]).sum(-1)

    def score_catalog(self, h, ids):
        return h @ self._current_items[ids].T

    def scores(self, h, ids):
        return (h[:, None] * self._current_items[ids]).sum(-1)
