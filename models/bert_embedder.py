import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class BertEmbedder:
    """
    Provides:
      - embed(text) -> (1, D) tensor on device
      - embed_texts(list[str], batch_size) -> (N, D) tensor on CPU (detached)
      - cosine_sim_matrix(A, B)
      - nearest(query_text, candidate_strings, candidate_vectors, top_k)
    """

    def __init__(self, model_name="distilbert-base-uncased", device: str | None = None):
        # prefer a provided device, otherwise detect cuda
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # move model to device once
        self.model.to(self.device)
        self.model.eval()

    # ---------------------------
    # Basic embedding (single text)
    # ---------------------------
    def embed(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        # move inputs to same device as model
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens).last_hidden_state.mean(dim=1)

        return out  # shape (1, hidden) on device

    # ---------------------------
    # Batch embedding for candidates
    # ---------------------------
    def embed_texts(self, texts, batch_size=32):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                out = self.model(**tokens).last_hidden_state.mean(dim=1)
            # move to CPU for caching/storage to avoid keeping large embeddings on GPU
            all_vecs.append(out.cpu())
        return torch.cat(all_vecs, dim=0)  # (N, hidden) on CPU

    # ---------------------------
    # Cosine similarity matrix (CPU or device)
    # ---------------------------
    def cosine_sim_matrix(self, A: torch.Tensor, B: torch.Tensor):
        # Accepts tensors on any device; normalize on that device
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)
        return A @ B.T  # shape (A_rows, B_rows)

    # ---------------------------
    # Nearest candidate (query_text is str)
    # ---------------------------
    def nearest(self, query_text: str, candidate_strings, candidate_embs: torch.Tensor, top_k=1):
        # candidate_embs is expected on CPU (as produced by embed_texts),
        # but we will move it to the model device for cosine computation to be fast on GPU.
        # If GPU not available, calculation will run on CPU.
        q = self.embed(query_text)  # (1, hidden) on self.device

        # move candidate embeddings to device for fast dot product if possible
        cand = candidate_embs.to(self.device)
        q_norm = F.normalize(q, dim=1)
        cand_norm = F.normalize(cand, dim=1)

        sims = (q_norm @ cand_norm.T)[0]  # (N,) on device
        top_vals, top_idx = torch.topk(sims, k=top_k)

        results = []
        for idx, score in zip(top_idx, top_vals):
            results.append((int(idx.item()), float(score.item())))
        return results

