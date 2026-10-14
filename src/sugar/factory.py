"""Single model registry, shared by training, evaluation and checkpoint loading."""
import hashlib
import numpy as np
import torch
from .model import SUGAR
from .baselines import GRU4Rec, Caser, BERT4Rec, SocialMF, LightGCNPlus, RLMRec

MODELS = ['sugar', 'sasrec', 'lm-feature', 'gru4rec', 'caser', 'bert4rec', 'socialmf', 'lightgcn-plus', 'rlmrec', 'llara']


def build_model(data, store, config, device='cpu'):
    name = config['model']
    if name not in MODELS:
        raise ValueError(f'Unknown model {name}; choices: {MODELS}')
    nitems, nusers = len(data['items']), len(data['users'])
    if name in ['sugar', 'sasrec', 'lm-feature']:
        model = SUGAR(nitems, store.dim, config, store.embeddings)
    elif name == 'gru4rec':
        model = GRU4Rec(nitems, config)
    elif name == 'caser':
        model = Caser(nitems, nusers, config)
    elif name == 'bert4rec':
        model = BERT4Rec(nitems, config)
    elif name == 'socialmf':
        model = SocialMF(nitems, data, config)
    elif name == 'lightgcn-plus':
        model = LightGCNPlus(nitems, store.dim, config, store.embeddings, data)
    elif name == 'rlmrec':
        profiles = np.zeros((nusers + 1, store.dim), dtype=np.float32)
        for u, snaps in store.snapshots.items():
            valid = [s for s in snaps if s['profile'] is not None]
            if valid:
                profiles[int(u)] = valid[-1]['vector']
        from pathlib import Path
        static_profiles = Path(store.artifacts) / 'static_user_embeddings.npy'
        if static_profiles.exists():
            profiles = np.load(static_profiles, allow_pickle=False)
        # The static LLM representations can be replaced with original baseline caches.
        if config.get('rlm_user_embeddings'):
            profiles = np.load(config['rlm_user_embeddings'], allow_pickle=False)
        items = store.embeddings
        if config.get('rlm_item_embeddings'):
            items = np.load(config['rlm_item_embeddings'], allow_pickle=False)
        if profiles.shape != (nusers + 1, items.shape[1]) or items.shape[0] != nitems + 1:
            raise ValueError('RLMRec semantic caches must follow current mapped IDs with row 0 padding')
        model = RLMRec(nitems, data, config, items, profiles)
        model.semantic_input_hashes = {"users": hashlib.sha256(profiles.tobytes()).hexdigest(), "items": hashlib.sha256(items.tobytes()).hexdigest()}
    else:
        from .llara import LLaRA
        model = LLaRA(nitems, data, config, store, device)
    return model.to(device)
