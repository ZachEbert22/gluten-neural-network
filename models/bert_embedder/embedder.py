import torch
from transformers import AutoTokenizer, AutoModel

class BertEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**tokens)

        # CLS token embedding
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec

