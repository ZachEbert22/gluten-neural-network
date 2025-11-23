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

use_gismo = True
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
        self.map = substitution_map  # dict: gluten ingredient → { "substitute":..., "ratio":... }
        self.embedder = BertEmbedder()
        # Precompute embeddings on CPU via embed_texts (it returns CPU tensor)
        self.keys = list(self.map.keys())
        # Use embed_texts (returns CPU tensor) and keep it there
        with torch.no_grad():
            self.key_vectors = self.embedder.embed_texts(self.keys, batch_size=64)  # (N, dim) CPU tensor

        # GISMo / NER flags (ensure defined earlier)
        try:
            from utils.gismo import GISMo
            self.gismo = GISMo(self.map, embedder=self.embedder)
        except Exception:
            self.gismo = None

        # Do not initialize heavy NER unless enabled; leave as None unless configured
        self.ner = None

    def parse_line(self, text):
        text = text.strip()
        if NUTRITION_RE.match(text):
            return None, None
        parsed = parse_ingredient_line(text)
        qty = parsed.get("quantity") or ""
        unit = parsed.get("unit") or ""
        ingredient = parsed.get("ingredient") or ""
        ingredient = normalize_ingredient_name(ingredient)
        return (qty + (" " + unit if unit else "")).strip(), ingredient

    def substitute(self, line):
        qty_unit, ingredient = self.parse_line(line)
        if ingredient is None:
            return line, False

        # Exact match (map keys are normalized)
        if ingredient in self.map:
            entry = self.map[ingredient]
            sub_name = entry.get("substitute") if isinstance(entry, dict) else entry
            ratio = float(entry.get("ratio", 1.0)) if isinstance(entry, dict) else 1.0
            # format quantity adjust if numeric
            try:
                qnum = float(qty_unit.split()[0]) if qty_unit and re.match(r'^\d+(\.\d+)?$', qty_unit.split()[0]) else None
            except Exception:
                qnum = None
            if qnum is not None:
                new_q = round(qnum * ratio, 2)
                formatted = f"{new_q} {' '.join(qty_unit.split()[1:])} {sub_name}".strip()
            else:
                formatted = f"{qty_unit} {sub_name}".strip()
            return formatted, True

        # Semantic fallback (ensure device alignment)
        with torch.no_grad():
            v = self.embedder.embed(ingredient)  # single vector (1,dim) on embedder.device
            # move key_vectors to same device as v
            key_vecs = self.key_vectors.to(v.device)
            sims = torch.nn.functional.cosine_similarity(v, key_vecs, dim=1)
            if sims.numel() == 0:
                return line, False
            idx = int(torch.argmax(sims).item())
            score = float(sims[idx].item())
            best_key = self.keys[idx]

        # If similarity low, try GISMo if available
        if score < 0.85 and self.gismo is not None:
            try:
                cand, gscore = self.gismo.best_substitute(ingredient)
            except Exception:
                cand, gscore = None, 0.0
            if cand and gscore >= 0.6:
                sub_info = self.map.get(cand, {})
                sub_name = sub_info.get("substitute") if isinstance(sub_info, dict) else sub_info
                ratio = float(sub_info.get("ratio", 1.0)) if isinstance(sub_info, dict) else 1.0
                # format
                formatted = f"{qty_unit} {sub_name}".strip()
                return formatted, True
            return line, False

        # Use best_key
        sub_info = self.map.get(best_key, {})
        sub_name = sub_info.get("substitute") if isinstance(sub_info, dict) else sub_info
        ratio = float(sub_info.get("ratio", 1.0)) if isinstance(sub_info, dict) else 1.0
        formatted = f"{qty_unit} {sub_name}".strip()
        return formatted, True


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

