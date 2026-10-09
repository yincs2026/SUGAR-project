"""Social graph construction (Section 3.5).

Modes:
  real           : edges from processed social_edges.pkl
  co_interaction : proxy graph built from TRAIN interactions only
                   (users co-interacting on >= threshold items are linked),
                   never touching validation/test behavior
  random         : random graph with same edge count (ablation/upper-bound check)

Direction: edge (u, w) means u follows w; N(u) = out-neighbors = followees.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class SocialGraph:
    def __init__(self, n_users: int):
        self.n_users = n_users
        self.adj = defaultdict(list)  # u -> list of followees w

    def add_edge(self, u: int, w: int):
        if u != w and w not in self.adj[u]:
            self.adj[u].append(w)

    def neighbors(self, u: int) -> list:
        return self.adj.get(u, [])

    @property
    def n_edges(self):
        return sum(len(v) for v in self.adj.values())


def load_real_graph(edges_path: str, n_users: int) -> SocialGraph:
    graph = SocialGraph(n_users)
    path = Path(edges_path)
    if not path.exists():
        return graph  # empty graph: social module disabled at runtime
    with open(path, "rb") as f:
        edges = pickle.load(f)
    for u, w in edges:
        if 0 <= u < n_users and 0 <= w < n_users:
            graph.add_edge(u, w)
    return graph


def build_co_interaction_graph(sequences: dict, n_users: int,
                              threshold: int = 3,
                              user_index_map: dict = None) -> SocialGraph:
    """Link users who share >= threshold items. TRAIN-only sequences only.

    sequences: {user_key: [item_idx, ...]} from the training split.
    user_index_map: maps user_key -> integer index; if None, keys are indices.
    """
    item2users = defaultdict(set)
    for u_key, seq in sequences.items():
        u = user_index_map[u_key] if user_index_map else u_key
        for i in set(seq):
            item2users[i].add(u)

    pair_count = defaultdict(int)
    for users in item2users.values():
        users = sorted(users)
        for a_idx in range(len(users)):
            for b_idx in range(a_idx + 1, len(users)):
                pair_count[(users[a_idx], users[b_idx])] += 1

    graph = SocialGraph(n_users)
    for (a, b), c in pair_count.items():
        if c >= threshold:
            graph.add_edge(a, b)  # bidirectional by construction
            graph.add_edge(b, a)
    return graph


def build_random_graph(n_users: int, n_edges: int, seed: int = 42) -> SocialGraph:
    """Random graph with the same edge count as the real one (control condition)."""
    rng = np.random.default_rng(seed)
    graph = SocialGraph(n_users)
    seen = set()
    while graph.n_edges < n_edges:
        u, w = rng.integers(0, n_users, size=2)
        if (u, w) in seen or u == w:
            continue
        seen.add((u, w))
        graph.add_edge(int(u), int(w))
    return graph


def get_graph(mode: str, n_users: int, real_edges_path: str = None,
              train_sequences: dict = None, threshold: int = 3,
              user_index_map: dict = None, seed: int = 42) -> SocialGraph:
    if mode == "real":
        return load_real_graph(real_edges_path, n_users)
    if mode == "co_interaction":
        assert train_sequences is not None
        return build_co_interaction_graph(train_sequences, n_users,
                                          threshold, user_index_map)
    if mode == "random":
        assert train_sequences is not None
        ref = load_real_graph(real_edges_path, n_users) if real_edges_path else None
        n_edges = ref.n_edges if ref else max(1, n_users // 2)
        return build_random_graph(n_users, n_edges, seed)
    raise ValueError(f"unknown graph mode: {mode}")