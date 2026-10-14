from pathlib import Path
import numpy as np
from .utils import load_json, save_json
from .offline import verify_manifest


def encoder_quality(data_path, artifacts, out, seed=42, clusters=None, sample_limit=None):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    data = load_json(data_path)
    manifest = verify_manifest(data, artifacts)
    x = np.load(Path(artifacts) / 'item_embeddings.npy', mmap_mode='r', allow_pickle=False)
    ids = [i for i in range(1, len(data['items']) + 1) if data['catalog'][str(i)]['category'] != 'unknown']
    if sample_limit is not None and len(ids) > sample_limit:
        ids = sorted(np.random.default_rng(seed).choice(ids, sample_limit, replace=False).tolist())
    labels = [str(data['catalog'][str(i)]['category']) for i in ids]
    categories = sorted(set(labels))
    if len(categories) < 2 or len(ids) < len(categories):
        raise ValueError('At least two independently labelled metadata categories are required')
    lookup = {v: i for i, v in enumerate(categories)}
    y = np.array([lookup[v] for v in labels])
    clusters = clusters or len(categories)
    if not 2 <= clusters <= len(ids):
        raise ValueError('Invalid number of clusters')
    matrix = np.asarray(x[ids], dtype=np.float32)
    # All item content is allowed by M1; no interaction-derived category labels are used.
    algorithm = KMeans if len(ids) <= 5000 else MiniBatchKMeans
    model = algorithm(n_clusters=clusters, random_state=seed, n_init=10)
    predicted = model.fit_predict(matrix)
    report = {'adjusted_rand_index': float(adjusted_rand_score(y, predicted)),
              'normalized_mutual_information': float(normalized_mutual_info_score(y, predicted)),
              'items': len(ids), 'metadata_categories': len(categories), 'clusters': clusters,
              'algorithm': algorithm.__name__, 'seed': seed, 'mock': manifest['mock'],
              'category_definition': 'exact canonical item metadata category string',
              'data_fingerprint': data['fingerprint'], 'reported_paper_ARI': 0.58,
              'paper_value_is_reference_not_target': True}
    save_json(out, report)
    return report


def static_baseline_profiles(data_path, artifacts, config, cache):
    from .semantics import SemanticBackend
    from .offline import verify_manifest
    from .utils import canonical, file_hash
    data = load_json(data_path)
    verify_manifest(data, artifacts)
    backend = SemanticBackend(config, cache)
    descriptors = load_json(Path(artifacts) / 'descriptors.json')
    x = np.load(Path(artifacts) / 'item_embeddings.npy', allow_pickle=False)
    profiles = np.zeros((len(data['users']) + 1, x.shape[1]), dtype=np.float32)
    texts = {}
    for user, sequence in data['sequences'].items():
        profile = backend.generate('build', {'descriptors': [descriptors[str(v)] for v, _ in sequence[:-2]]})
        profiles[int(user)] = backend.profile_vector(profile)
        texts[user] = profile
    np.save(Path(artifacts) / 'static_user_embeddings.npy', profiles)
    save_json(Path(artifacts) / 'static_baseline_profiles.json', texts)
    save_json(Path(artifacts) / 'static_baseline_usage.json', backend.usage)
    # Additional files participate in artifact integrity/provenance.
    manifest = load_json(Path(artifacts) / 'manifest.json')
    for file in ['static_user_embeddings.npy', 'static_baseline_profiles.json', 'static_baseline_usage.json']:
        manifest['files'][file] = file_hash(Path(artifacts) / file)
    save_json(Path(artifacts) / 'manifest.json', manifest)
