#!/usr/bin/env python3
"""
Backend API for the Gluten-Free Converter

This FastAPI backend handles:
- FoodNER extraction
- Ingredient parsing (quantity, unit, name)
- Gluten detection via gluten_ingredients.json
- GISMo graph-based substitution lookup
- Rule-based fallback substitutions
- SHARE transformer rewriting of the entire recipe

Streamlit UI (streamlit_app.py) will call this backend.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from utils.normalization import normalize_ingredient
from utils.gluten_check import load_gluten_ingredients, load_substitutions
from utils.ingredient_parser import parse_ingredient_line
from utils.gismo import GISMo
from models.food_ner import FoodNER
from models.share_rewriter import SHARERewriter

app = FastAPI(title="Gluten-Free Conversion API")

# Load data & models once
GLUTEN_LIST = load_gluten_ingredients()
SUBSTITUTIONS = load_substitutions()
food_ner = FoodNER()
gismo = GISMo()
rewriter = SHARERewriter()


# -----------------------------
# Request schema
# -----------------------------
class ConvertRequest(BaseModel):
    ingredients: List[str]
    recipe_text: str = ""  # Optional, for rewriting instructions


# -----------------------------
# Helper: contains gluten
# -----------------------------
def contains_gluten(name: str) -> bool:
    n = name.lower()
    return any(g in n for g in GLUTEN_LIST)


# -----------------------------
# Main conversion endpoint
# -----------------------------
@app.post("/convert_to_gluten_free")
def convert_to_gluten_free(req: ConvertRequest):
    output = []
    parsed_items = []

    for raw_line in req.ingredients:

        # 1. Extract ingredient name using FoodNER
        ner_name = food_ner.extract(raw_line)
        if not ner_name:
            output.append({
                "original": raw_line,
                "status": "not_ingredient",
                "message": "Could not detect ingredient using FoodNER."
            })
            continue

        # 2. Parse ingredient structure
        parsed = parse_ingredient_line(raw_line)
        ingredient_name = parsed.get("ingredient", ner_name)
        ingredient_name_norm = normalize_ingredient(ingredient_name)

        parsed_items.append({
            "original": raw_line,
            "ingredient": ingredient_name_norm,
            "quantity": parsed.get("quantity"),
            "unit": parsed.get("unit"),
        })

        # 3. Check gluten
        if not contains_gluten(ingredient_name_norm):
            output.append({
                "original": raw_line,
                "converted": raw_line,
                "status": "gluten_free",
                "reason": "naturally gluten-free"
            })
            continue

        # 4. GISMo substitution (primary)
        gismo_sub = gismo.find_best_substitute(ingredient_name_norm)

        if gismo_sub:
            output.append({
                "original": raw_line,
                "converted": gismo.apply_quantity_scaling(parsed, gismo_sub),
                "status": "gismo_substitution",
                "substitute": gismo_sub,
            })
            continue

        # 5. Fallback: rule-based json
        backup = None
        for g, info in SUBSTITUTIONS.items():
            if g in ingredient_name_norm:
                backup = info
                break

        if backup:
            output.append({
                "original": raw_line,
                "converted": gismo.apply_quantity_scaling(parsed, backup),
                "status": "rule_fallback",
                "substitute": backup
            })
        else:
            output.append({
                "original": raw_line,
                "status": "no_substitute_found"
            })

    # -----------------------------
    # 6. SHARE Transformer rewriting (full recipe text)
    # -----------------------------
    rewritten = None
    if req.recipe_text.strip():
        rewritten = rewriter.rewrite(req.recipe_text, substitutions=output)

    return {
        "converted": output,
        "rewritten_recipe": rewritten
    }


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)

