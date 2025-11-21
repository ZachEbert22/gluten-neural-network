# models/food_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Tuple

# Recommended pretrained NER model; you can change to another HF checkpoint if preferred
MODEL_NAME = "vblagoje/bert-english-uncased-finetuned-ontonotes"  # placeholder: swap with a recipe/ingredient NER if preferred

class FoodNER:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple", device=device)

    def extract_ingredients(self, text: str) -> List[str]:
        """
        Run NER on `text` and return list of extracted spans that look like ingredient names.
        This is a complement to the rule-based parser.
        """
        ents = self.pipe(text)
        # return top string spans (simple heuristics)
        results = []
        for ent in ents:
            span = ent.get("word") or ent.get("entity_group") or ent.get("entity")
            if span:
                # filter out numeric or measurement tokens - keep alpha spans
                if any(c.isalpha() for c in span):
                    results.append(span.lower())
        # dedupe and return
        out = []
        for r in results:
            if r not in out:
                out.append(r)
        return out

