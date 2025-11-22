# models/food_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict

NER_MODEL_PATH = "models/foodner"   # LOCAL MODEL

class FoodNER:
    def __init__(self):
        print(f"Loading local NER model from: {NER_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> List[Dict]:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**tokens)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        tokens_list = self.tokenizer.convert_ids_to_tokens(tokens.input_ids.squeeze())

        results = []
        for token, pred_id in zip(tokens_list, predictions):
            label = self.id2label[int(pred_id)]
            results.append({"token": token, "label": label})

        return results

