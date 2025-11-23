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
        """
        Returns: (new_line, changed_flag)
        Robust pipeline:
         - parse + normalize
         - exact match
         - substring rule-fallback on both normalized ingredient and the raw line
         - semantic embedder fallback (existing)
         - GISMo fallback (existing)
        """
        qty, ingredient = self.parse_line(line)

        if ingredient is None:
            return line, False

        # 1) Exact match (fast)
        if ingredient in self.map:
            return f"{qty} {self.map[ingredient]}".strip(), True

        # 2) Rule/sub-string fallback (robust safety-net)
        # Check normalized ingredient tokens AND the raw original line (lowercased)
        raw_lower = line.lower()
        # Try: any substitution key appears in ingredient OR raw line
        for gkey, info in self.map.items():
            # compare normalized keys and also check substrings in original text
            if (gkey in ingredient) or (gkey in raw_lower) or (ingredient in gkey):
                # use ratio/substitute if available; info may be dict or string
                if isinstance(info, dict):
                    sub = info.get("substitute", None) or info.get("sub", None) or str(info)
                else:
                    sub = info
                return f"{qty} {sub}".strip(), True

        # 3) Semantic fallback (BERT) among keys
        with torch.no_grad():
            v = self.embedder.embed(ingredient)  # shape (1, dim) or (dim,)
            # ensure vector on same device as self.key_vectors
            # self.key_vectors is likely on CPU (from embed_texts), move v to CPU
            if v.device != self.key_vectors.device:
                v_cpu = v.to(self.key_vectors.device)
            else:
                v_cpu = v
            sims = torch.nn.functional.cosine_similarity(v_cpu, self.key_vectors, dim=-1)
            idx = torch.argmax(sims).item()
            best_key = self.keys[idx]
            score = sims[idx].item()

        # If semantic score strong enough, accept
        if score >= 0.85:
            sub = self.map[best_key] if isinstance(self.map[best_key], str) else self.map[best_key].get("substitute", self.map[best_key])
            return f"{qty} {sub}".strip(), True

        # 4) GISMo fallback (if available)
        if hasattr(self, "gismo") and self.gismo is not None:
            try:
                cand, gscore = self.gismo.best_substitute(ingredient)
                if cand and gscore >= 0.55:
                    sub = self.map.get(cand, cand)
                    return f"{qty} {sub}".strip(), True
            except Exception:
                pass

        # nothing found
        return line, False

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

