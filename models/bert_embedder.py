import torch
import torch.nn as nn
from transformers import AutoModel

class BertEmbeddingsMLP(nn.Module):
    def __init__(self, pretrained="distilbert-base-uncased", embedding_dim=768, hidden_dim=256, num_subs=21):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        self.pool = lambda x: x.last_hidden_state[:, 0, :]  # [CLS] token pooling
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.flag_head = nn.Linear(hidden_dim, 2)
        self.sub_head = nn.Linear(hidden_dim, num_subs)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = self.pool(out)
        h = self.relu(self.fc1(vec))
        return self.flag_head(h), self.sub_head(h)

