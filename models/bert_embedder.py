"""
BERT embedder to convert ingredient text into dense embeddings, and compute cosine similarity.
This module caches candidate substitute embeddings to avoid re-encoding per request.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Tuple

MODEL_NAME = "distilbert-base-uncased"
CACHE_DIR = Path("models/bert_embedder_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class BertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Return embeddings tensor shape (N, D). Using CLS pooling.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64)
                toks = {k: v.to(self.device) for k, v in toks.items()}
                outputs = self.model(**toks)
                # CLS pooling: last_hidden_state[:,0,:]
                cls = outputs.last_hidden_state[:, 0, :].detach().cpu()
                out.append(cls)
        return torch.cat(out, dim=0)  # shape (N, D)

    @staticmethod
    def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        # a: (n, d), b: (m, d)
        a_np = a.numpy()
        b_np = b.numpy()
        an = a_np / np.linalg.norm(a_np, axis=1, keepdims=True)
        bn = b_np / np.linalg.norm(b_np, axis=1, keepdims=True)
        return an.dot(bn.T)  # (n, m) cosine similarities

    def nearest(self, text: str, candidate_texts: List[str], candidate_embs: torch.Tensor, top_k: int = 1) -> List[Tuple[int, float]]:
        t_emb = self.embed_texts([text])  # (1, D)
        sims = self.cosine_sim_matrix(t_emb, candidate_embs)[0]  # (m,)
        idxs = sims.argsort()[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs]

