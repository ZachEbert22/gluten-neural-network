"""
Unified backend API for Gluten-Free Converter.

Supports:
 - POST /process
 - POST /api/parse_recipe     <-- Streamlit expects this

Run:
    uvicorn unified_api:app --reload --port 8000
"""
from substitution_pipeline import SubstitutionEngine, load_substitutions
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
import traceback

# ---------------------------------------------------------------------
# Safe importer
# ---------------------------------------------------------------------
def safe_import(name: str):
    try:
        mod = __import__(name, fromlist=['*'])
        return mod
    except Exception as e:
        raise ImportError(f"Failed to import {name}: {e}")

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(title="Unified Gluten-Free API")

# ---------------------------------------------------------------------
# INPUT REQUEST SCHEMA
# ---------------------------------------------------------------------
class ProcessRequest(BaseModel):
    ingredients: Optional[List[str]] = None
    raw_text: Optional[str] = None
    url: Optional[str] = None
    instructions: Optional[str] = ""


# ---------------------------------------------------------------------
# URL Extraction
# ---------------------------------------------------------------------
def extract_from_url(url: str):
    """Return (ingredient_lines, instructions_text) extracted heuristically."""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return [], ""

    # Ingredient selectors
    selectors = [
        '[itemprop="recipeIngredient"]',
        '.ingredients-item-name',
        '.recipe-ingredients__list-item',
        'li.ingredient',
        'span.ingredients-item-name',
        '.ingredient'
    ]

    ing_lines = []
    for sel in selectors:
        for tag in soup.select(sel):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                ing_lines.append(txt)

    # Fallback based on <li> & tokens
    if not ing_lines:
        for li in soup.find_all("li"):
            txt = li.get_text(separator=" ", strip=True)
            if re.search(r'\b(cup|tsp|tbsp|ml|g|flour|sugar|salt|butter|egg)\b', txt, re.I):
                ing_lines.append(txt)

    # Instructions selectors
    instr_selectors = [
        '[itemprop="recipeInstructions"]',
        '.instructions-section',
        '.directions',
        '.method-steps',
        '.recipe-directions__list',
        'ol.instructions',
        'ol'
    ]

    instructions_text = ""
    for sel in instr_selectors:
        tags = soup.select(sel)
        if tags:
            texts = [t.get_text(" ", strip=True) for t in tags]
            instructions_text = "\n".join(texts).strip()
            break

    # Fallback based on verbs
    if not instructions_text:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        candidates = [
            p for p in paragraphs
            if re.search(r'\b(mix|bake|stir|cook|whisk|simmer|fold)\b', p, re.I)
        ]
        instructions_text = "\n".join(candidates[:10]).strip()

    # Deduplicate
    seen = set()
    clean = []
    for l in ing_lines:
        if l not in seen:
            clean.append(re.sub(r'\s+', ' ', l).strip())
            seen.add(l)

    return clean, instructions_text


# ---------------------------------------------------------------------
# MODEL LOADING REGISTRY
# ---------------------------------------------------------------------
class Registry:
    loaded = False

    ingredient_parser = None
    foodner = None
    gismo = None
    substitution_engine = None
    share_rewriter = None
    gluten_list = None
    substitutions_json = None


def load_models():
    if Registry.loaded:
        return

    try:
        mod_parser = safe_import("utils.ingredient_parser")
        mod_gismo = safe_import("utils.gismo")
        mod_share = safe_import("models.share_rewriter")
        mod_gluten_check = safe_import("utils.gluten_check")
        mod_subpipeline = safe_import("substitution_pipeline")

        # IngredientParser
        Registry.ingredient_parser = getattr(mod_parser, "IngredientParser")()

        # --- Disable broken FoodNER (no HF model files exist) ---
        class DummyNER:
            def extract_ingredients(self, text):
                #parts = re.split(r'\n|;|\.', text)
                parts = re.split(r'\n|;', text)
                return [p.strip() for p in parts if p.strip()]

        Registry.foodner = DummyNER()

        # Substitution data
        if hasattr(mod_subpipeline, "load_substitution_data"):
            subs = mod_subpipeline.load_substitution_data()
        else:
            subs = mod_subpipeline.load_substitutions()

        Registry.substitutions_json = subs

        # SubstitutionEngine
        if hasattr(mod_subpipeline, "SubstitutionEngine"):
            Registry.substitution_engine = mod_subpipeline.SubstitutionEngine(subs)

        # GISMo
        if hasattr(mod_gismo, "GISMo"):
            Registry.gismo = mod_gismo.GISMo(subs)
        else:
            Registry.gismo = None

        # SHARE rewriter
        if hasattr(mod_share, "SHARERewriter"):
            Registry.share_rewriter = mod_share.SHARERewriter()
        else:
            # fallback rewriter
            Registry.share_rewriter = lambda inst, subs: inst

        # Gluten list
        Registry.gluten_list = mod_gluten_check.load_gluten_ingredients()

        Registry.loaded = True
        print("Unified API: models loaded.")

    except Exception as e:
        print("Model loading error:", e)
        print(traceback.format_exc())
        raise


# ---------------------------------------------------------------------
# Gluten check
# ---------------------------------------------------------------------
def contains_gluten(ingredient_name: str) -> bool:
    name = (ingredient_name or "").lower()
    for g in Registry.gluten_list or []:
        if g in name:
            return True
    return False


# ---------------------------------------------------------------------
# MAIN PROCESS ENDPOINT
# ---------------------------------------------------------------------
@app.post("/process")
def process(req: ProcessRequest):
    load_models()

    # Decide what the user gave us
    if req.ingredients:
        lines = req.ingredients
    elif req.raw_text:
        if Registry.foodner:
            try:
                lines = Registry.foodner.extract_ingredients(req.raw_text)
            except Exception:
                lines = [l.strip() for l in req.raw_text.split("\n") if l.strip()]
        else:
            lines = [l.strip() for l in req.raw_text.split("\n") if l.strip()]
    elif req.url:
        lines, auto_instr = extract_from_url(req.url)
        if auto_instr and not req.instructions:
            req.instructions = auto_instr
    else:
        raise HTTPException(status_code=400, detail="Provide ingredients, raw_text, or url.")

    # Parse each ingredient
    parsed_output = []
    for line in lines:
        parsed = Registry.ingredient_parser.parse(line)
        parsed_output.append({"original": line, "parsed": parsed})

    # Substitutions
    subs_output = []
    for entry in parsed_output:
        orig = entry["original"]
        name = entry["parsed"].get("ingredient", "").lower()

        if not name:
            subs_output.append({"original": orig, "converted": orig, "status": "not_parsed"})
            continue

        if not contains_gluten(name):
            subs_output.append({"original": orig, "converted": orig, "status": "gluten_free"})
            continue

        # Try substitution engine
        conv, changed = Registry.substitution_engine.substitute(orig)
        # If substitution engine returned a dict, convert it to readable text
        if isinstance(conv, dict) and "substitute" in conv:
            qty = entry["parsed"].get("quantity", "")
            unit = entry["parsed"].get("unit", "")
            sub = conv["substitute"]
            ratio = conv.get("ratio", 1)

            # Quantity scaling
            try:
                scaled_qty = float(qty) * float(ratio)
                qty_out = f"{scaled_qty:g}"
            except:
                qty_out = qty

            conv = f"{qty_out} {unit} {sub}".strip()

        # If GISMo exists but SubEngine didn't change anything
        if not changed and Registry.gismo:
            ranked = Registry.gismo.rank_substitutes(name, top_k=3)
            if ranked:
                candidate = ranked[0][0]
                sub_info = Registry.substitutions_json.get(candidate)
                if sub_info:
                    qty = entry["parsed"].get("quantity", "")
                    unit = entry["parsed"].get("unit", "")
                    conv = f"{qty} {unit} {sub_info['substitute']}".strip()
                    changed = True

        subs_output.append({
            "original": orig,
            "converted": conv,
            "status": "substituted" if changed else "no_substitute_found"
        })

    # Rewrite instructions
    rewritten = None
    try:
        rewritten = Registry.share_rewriter.rewrite(req.instructions or "", subs_output)
    except Exception:
        rewritten = req.instructions or ""

    return {
        "parsed": parsed_output,
        "substitutions": subs_output,
        "rewritten": rewritten
    }


# ---------------------------------------------------------------------
# NEW: Streamlit-Compatible Endpoint
# ---------------------------------------------------------------------
@app.post("/api/parse_recipe")
def alias_parse_recipe(req: ProcessRequest):
    return process(req)

def parse_recipe_for_streamlit(payload: dict):
    """
    Streamlit frontend expects:
        { "ingredients": [...], "instructions": "...", "rewritten": "..." }
    """
    try:
        result = process(ProcessRequest(
            ingredients=payload.get("ingredients"),
            raw_text=payload.get("raw_text"),
            url=payload.get("recipe_url"),
            instructions=payload.get("instructions", "")
        ))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    # Convert unified result → Streamlit expected structure
    return {
        "ingredients": result["parsed"],
        "substitutions": result["substitutions"],
        "instructions": payload.get("instructions", ""),
        "rewritten": result["rewritten"]
    }

