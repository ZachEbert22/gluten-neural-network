# utils/gismo.py
from collections import defaultdict, deque
from typing import Dict, List
import torch
from models.bert_embedder import BertEmbedder


class GISMo:
    """
    Simple graph-based ingredient substitution helper.
    - Build graph from substitutions.json keys and values.
    - Provide nearest substitutions using BFS + embedding similarity.
    """

    def __init__(self, substitutions: Dict[str, Dict], embedder: BertEmbedder = None):
        self.map = substitutions
        self.nodes = set()
        self.adj = defaultdict(set)

        # Build graph
        self._build_graph_from_map()

        # Embeddings
        self.embedder = embedder if embedder is not None else BertEmbedder()

        # Sorted nodes for repeatable indexing
        self.node_names = sorted(list(self.nodes))

        # Precompute all node vectors on CPU
        self.node_vectors = None
        if self.node_names:
            self.node_vectors = self.embedder.embed_texts(self.node_names, batch_size=64)

        # Device cache
        self._node_vectors_cache = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _add_edge(self, a: str, b: str):
        self.nodes.add(a)
        self.nodes.add(b)
        self.adj[a].add(b)
        self.adj[b].add(a)

    def _build_graph_from_map(self):
        for key, info in self.map.items():
            a = key.lower()

            # Allow multiple possible schema conventions
            if "substitute" in info:
                subs = [info["substitute"]]
            elif "substitutes" in info:
                subs = info["substitutes"]
            else:
                continue

            for sub in subs:
                b = sub.lower()
                self._add_edge(a, b)

                # Token-level fallback edges (keeps graph rich)
                for tok in a.split():
                    self._add_edge(a, tok)
                for tok in b.split():
                    self._add_edge(b, tok)

    # ------------------------------------------------------------------
    # Graph distance (used optionally)
    # ------------------------------------------------------------------
    def graph_distance(self, source: str, target: str, max_depth=3) -> int:
        source = source.lower()
        target = target.lower()
        if source == target:
            return 0
        if source not in self.nodes or target not in self.nodes:
            return 999

        seen = {source}
        q = deque([(source, 0)])

        while q:
            node, d = q.popleft()
            if d >= max_depth:
                continue

            for nb in self.adj[node]:
                if nb == target:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))

        return 999

    # ------------------------------------------------------------------
    # Ranking substitutes
    # ------------------------------------------------------------------
    def rank_substitutes(self, ingredient: str, top_k: int = 10):
        """
        Returns list of (node_name, score) sorted by similarity.
        """

        if self.node_vectors is None or len(self.node_names) == 0:
            return []

        with torch.no_grad():
            qvec = self.embedder.embed(ingredient)

            # Ensure shape: (1, D)
            if qvec.dim() == 1:
                qvec = qvec.unsqueeze(0)

            # Move cached node vectors to same device
            device = str(qvec.device)
            if device not in self._node_vectors_cache:
                self._node_vectors_cache[device] = self.node_vectors.to(qvec.device)

            node_vecs = self._node_vectors_cache[device]

            # Compute cosine similarity
            sims = torch.nn.functional.cosine_similarity(qvec, node_vecs, dim=1)

            top_k = min(top_k, sims.numel())
            vals, idxs = torch.topk(sims, top_k)

            results = []
            for v, i in zip(vals.tolist(), idxs.tolist()):
                name = self.node_names[i]
                results.append((name, float(v)))

            return results

    # ------------------------------------------------------------------
    # Single best substitute
    # ------------------------------------------------------------------
    def best_substitute(self, ingredient: str):
        try:
            ranked = self.rank_substitutes(ingredient, top_k=1)
            if not ranked:
                return None, 0.0
            return ranked[0]
        except Exception as e:
            print("GISMo.best_substitute error:", e)
            return None, 0.0

