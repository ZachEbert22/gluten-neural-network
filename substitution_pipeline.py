# substitution_pipeline.py (top)
import re
import torch
from models.bert_embedder import BertEmbedder

# NEW imports
from utils.gismo import GISMo
from models.food_ner import FoodNER
from utils.ingredient_parser import parse_ingredient_line, normalize_ingredient_name
import json
import os
use_gismo = False
use_ner = False

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
        self.key_vectors = self.embedder.embed_texts(self.keys)

        # GISMo graph (optional)
        self.gismo = GISMo(self.map, embedder=self.embedder) if use_gismo else None

        # Food NER (optional)
        self.ner = FoodNER(device=0 if torch.cuda.is_available() else -1) if use_ner else None

    def parse_line(self, text):
        """Return (qty+unit, ingredient_name_only)"""
        text = text.strip()
        if NUTRITION_RE.match(text):
            return None, None

        parsed = parse_ingredient_line(text)
        qty = parsed.get("quantity") or ""
        unit = parsed.get("unit") or ""
        ingredient = parsed.get("ingredient") or ""
        # normalize
        ingredient = normalize_ingredient_name(ingredient)
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
            with torch.no_grad():
                v = self.embedder.embed(ingredient)
                key_vectors = self.key_vectors.to(v.device)  # move key_vectors to same device
                sims = torch.nn.functional.cosine_similarity(v, key_vectors)
                idx = torch.argmax(sims).item()
                best_key = self.keys[idx]
                score = sims[idx].item()

            idx = torch.argmax(sims).item()
            best_key = self.keys[idx]
            score = sims[idx].item()

        # Too weak → try GISMo graph if available
        if score < 0.85 and self.gismo is not None:
            cand, gscore = self.gismo.best_substitute(ingredient)
            if cand and gscore >= 0.6:
                sub = self.map.get(cand, self.map.get(cand, cand))
                return f"{qty} {sub}".strip(), True
            # else give up
            return line, False

        sub = self.map[best_key]
        return f"{qty} {sub}".strip(), True

def load_substitutions():
    """
    Loads substitution.json from the project root or utils/ if present.
    Expected format:
    {
        "wheat flour": "gluten-free flour",
        ...
    }
    """
    possible_paths = [
        "substitutions.json",
        "./substitutions.json",
        "data/substitutions.json",
        "./data/substitutions.json",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        "Could not find substitution.json in root or data/. "
        "Checked: " + ", ".join(possible_paths)
    )

