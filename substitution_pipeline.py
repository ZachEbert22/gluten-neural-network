import re
import torch
from models.bert_embedder import BertEmbedder

# Nutrition lines we should NOT substitute
NUTRITION_RE = re.compile(
    r"^(fat|saturates|carbs|sugars|fibre|fiber|protein|calories)\b",
    re.IGNORECASE
)

QUANTITY_RE = re.compile(r"(\d+(\.\d+)?|\d+\s*\d*\/\d+)")
UNIT_RE = re.compile(
    r"\b(g|kg|cup|cups|tbsp|tsp|ml|l|oz|ounce|ounces|lb|pound)\b",
    re.IGNORECASE
)


class SubstitutionEngine:
    """
    Handles gluten substitution using:
    
    1) Exact matches from substitution.json
    2) Semantic similarity (BERT) among gluten ingredients only
    """

    def __init__(self, substitution_map):
        self.map = substitution_map  # dict: gluten ingredient → gluten-free version
        self.embedder = BertEmbedder()

        # Precompute embeddings of gluten trigger keys
        self.keys = list(self.map.keys())
        self.key_vectors = self.embedder.encode_texts(self.keys)

    def parse_line(self, text):
        """Return (qty+unit, ingredient_name_only)"""
        text = text.strip()

        if NUTRITION_RE.match(text):
            return None, None

        qty_match = QUANTITY_RE.search(text)
        qty = qty_match.group(0) if qty_match else ""

        unit_match = UNIT_RE.search(text)
        unit = unit_match.group(0) if unit_match else ""

        ingredient = text
        if qty:
            ingredient = ingredient.replace(qty, "")
        if unit:
            ingredient = ingredient.replace(unit, "")

        ingredient = ingredient.strip(" ,-").lower()

        return (qty + (" " + unit if unit else "")).strip(), ingredient

    def substitute(self, line):
        """
        Returns: (new_line, changed_flag)
        """

        qty, ingredient = self.parse_line(line)

        if ingredient is None:
            return line, False

        # 1️⃣ Exact match
        if ingredient in self.map:
            return f"{qty} {self.map[ingredient]}".strip(), True

        # 2️⃣ Semantic fallback among gluten keys only
        with torch.no_grad():
            v = self.embedder.embed(ingredient)
            sims = torch.nn.functional.cosine_similarity(v, self.key_vectors)
            idx = torch.argmax(sims).item()
            best_key = self.keys[idx]
            score = sims[idx].item()

        # Too weak → do not substitute
        if score < 0.85:
            return line, False

        sub = self.map[best_key]
        return f"{qty} {sub}".strip(), True

