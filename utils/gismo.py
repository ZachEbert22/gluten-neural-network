# utils/gismo.py
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import math
import json
import os

from models.bert_embedder import BertEmbedder
import torch

class GISMo:
    """
    Simple graph-based ingredient substitution helper.
    - Build graph from substitutions.json keys and values.
    - Provide nearest substitutions using BFS graph distance + embedding similarity.
    """

    def __init__(self, substitutions: Dict[str, Dict], embedder: BertEmbedder = None):
        self.map = substitutions
        self.nodes = set()
        self.adj = defaultdict(set)
        self._build_graph_from_map()
        self.embedder = embedder if embedder is not None else BertEmbedder()
        # precompute embeddings for all nodes
        self.node_list = sorted(list(self.nodes))
        self.node_vectors = None
        if self.node_list:
            self.node_vectors = self.embedder.embed_texts(self.node_list, batch_size=64)

    def _add_edge(self, a, b):
        self.adj[a].add(b)
        self.adj[b].add(a)
        self.nodes.add(a)
        self.nodes.add(b)

    def _build_graph_from_map(self):
        for g_key, info in self.map.items():
            sub = info.get("substitute")
            if sub:
                a = g_key.lower()
                b = sub.lower()
                self._add_edge(a, b)
                # also connect close variations (split tokens)
                for tok in a.split():
                    self._add_edge(a, tok)
                for tok in b.split():
                    self._add_edge(b, tok)

    def graph_distance(self, source: str, target: str, max_depth=3) -> int:
        """BFS distance (min hops) between two nodes; if unreachable return large."""
        source = source.lower()
        target = target.lower()
        if source == target: return 0
        if source not in self.nodes or target not in self.nodes:
            return 999
        seen = set([source])
        q = deque([(source, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_depth: continue
            for nb in self.adj[node]:
                if nb == target:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return 999

    def rank_substitutes(self, ingredient: str, top_k=3) -> List[Tuple[str, float]]:
        """
        Return list of (candidate, score) combining graph proximity and embedding similarity.
        Score is higher for closer graph distance and higher cosine sim.
        """
        ing = ingredient.lower()
        results = []
        # embedding for query
        qvec = self.embedder.embed(ing)
        if self.node_vectors is None or len(self.node_list) == 0:
            return []
        # compute cosine sims in torch
        sims = torch.nn.functional.cosine_similarity(qvec, self.node_vectors)
        # iterate through candidates and compute combined score
        for idx, node in enumerate(self.node_list):
            sim = float(sims[idx].item())
            # compute distance penalty
            dist = self.graph_distance(ing, node, max_depth=4)
            if dist >= 999:
                dist_penalty = 0.0
            else:
                dist_penalty = 1.0 / (1.0 + dist)  # 1.0 for dist=0, 0.5 for dist=1, etc.
            combined = 0.6 * sim + 0.4 * dist_penalty  # weights tunable
            results.append((node, combined, sim, dist))
        # sort
        results.sort(key=lambda x: x[1], reverse=True)
        top = [(r[0], r[1]) for r in results[:top_k]]
        return top

    def best_substitute(self, ingredient: str, min_score=0.6):
        ranked = self.rank_substitutes(ingredient, top_k=5)
        if not ranked:
            return None, 0.0
        if ranked[0][1] >= min_score:
            return ranked[0]  # (candidate, score)
        return None, ranked[0][1] if ranked else (None, 0.0)

