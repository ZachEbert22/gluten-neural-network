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
import json
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
# ---------- REPLACEMENT: robust extract_from_url ----------

def extract_from_url(url: str):
    """Return (ingredient_lines, instructions_text) extracted cleanly from a recipe page."""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return [], ""

    # --------- Clean out obviously irrelevant containers ---------
    for junk in soup.select(
        "header, footer, nav, script, style, noscript, .comments, #comments, .comment-list, "
        ".related-posts, .sidebar, .footer, .advertisement, .ad, .promo"
    ):
        junk.decompose()

    # --------- INGREDIENT SELECTORS ---------
    selectors = [
        '[itemprop="recipeIngredient"]',
        "li.ingredient",
        ".ingredient",
        ".ingredients-item-name",
        ".wprm-recipe-ingredient",
        "li.ingredients-item",
        ".recipe-ingredients__list-item",
    ]

    ing_lines = []
    for sel in selectors:
        for tag in soup.select(sel):
            txt = tag.get_text(" ", strip=True)
            if txt:
                ing_lines.append(txt)

    # Fallback: try bullet lists that LOOK like ingredients
    if not ing_lines:
        for li in soup.find_all("li"):
            t = li.get_text(" ", strip=True)
            if re.search(r'\b(cup|tsp|tablespoon|gram|salt|sugar|flour|butter)\b', t, re.I):
                ing_lines.append(t)

    # --------- INSTRUCTIONS SELECTORS ---------
    instr_selectors = [
        '[itemprop="recipeInstructions"]',
        ".wprm-recipe-instruction-text",
        ".instructions",
        ".method-steps",
        ".directions",
        ".recipe-directions__list",
        "ol.instructions",
        "ol.method",
    ]

    instructions_text = ""
    for sel in instr_selectors:
        tags = soup.select(sel)
        if tags:
            texts = [t.get_text(" ", strip=True) for t in tags]
            instructions_text = "\n".join(texts).strip()
            break

    # Fallback: paragraphs with cooking verbs
    if not instructions_text:
        verbs = r"(mix|whisk|stir|bake|fold|heat|cook|combine|pour|spread|preheat|simmer)"
        candidates = []
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if re.search(verbs, text, re.I):
                candidates.append(text)
        instructions_text = "\n".join(candidates[:10])

    # --------- Final clean: remove comments, dates, "Reply ↓" ---------
    bad_patterns = [
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
        r"Reply\s*↓",
        r"^\s*[A-Za-z]+ \d{1,2}, \d{4}",
        r"^\s*\d{1,2}\s*$",
        r"Healthy Muffin Recipes",
        r"Baking Dish",
    ]

    clean_ing = []
    seen = set()

    for l in ing_lines:
        l = re.sub(r"\s+", " ", l).strip()
        if any(re.search(bp, l, re.I) for bp in bad_patterns):
            continue
        if l not in seen:
            clean_ing.append(l)
            seen.add(l)

    return clean_ing, instructions_text.strip()


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

    # 1) Resolve input -> raw_lines
    raw_lines = []
    if req.ingredients:
        raw_lines = [l.strip() for l in req.ingredients if l and l.strip()]
    elif req.raw_text:
        # if FoodNER present, try it (our DummyNER returns lines)
        try:
            raw_lines = Registry.foodner.extract_ingredients(req.raw_text)
        except Exception:
            raw_lines = [l.strip() for l in req.raw_text.split("\n") if l.strip()]
    elif req.url:
        raw_lines, auto_instr = extract_from_url(req.url)
        if auto_instr and not req.instructions:
            req.instructions = auto_instr
    else:
        raise HTTPException(status_code=400, detail="Provide ingredients, raw_text, or url.")

    # 2) Filter lines using classifier (if available) and light heuristic
    filtered = []
    # if you have a classifier model, call it here; otherwise use heuristics
    def looks_like_ingredient(line):
        # quick heuristics: contains measurement or number OR common food words
        if re.search(r'\d', line) or re.search(r'\b(cup|tsp|tbsp|gram|g|oz|ml|stick|slice|teaspoon|tablespoon|kg|pound)\b', line, re.I):
            return True
        # avoid lines that look like comments / long paragraphs
        if len(line) > 250: 
            return False
        if re.search(r'(Reply|Comment|©|Subscribe|Follow)', line, re.I):
            return False
        return True

    for l in raw_lines:
        if Registry.ingredient_parser:
            parsed_tmp = Registry.ingredient_parser.parse(l)
            if parsed_tmp and parsed_tmp.get("ingredient"):
                # the parser found an ingredient field — keep
                filtered.append(l)
                continue
        # fallback heuristic
        if looks_like_ingredient(l):
            filtered.append(l)

    # 3) Parse and run substitution
    parsed_output = []
    subs_output = []
    for l in filtered:
        parsed = Registry.ingredient_parser.parse(l)
        parsed_output.append({"original": l, "parsed": parsed})
        # detect gluten
        name = (parsed.get("ingredient") or "").lower()
        if not name:
            subs_output.append({"original": l, "converted": l, "status": "not_parsed"})
            continue
        if not contains_gluten(name):
            subs_output.append({"original": l, "converted": l, "status": "gluten_free"})
            continue
        try:
            conv, changed = Registry.substitution_engine.substitute(l)
        except Exception:
            conv, changed = l, False

        # GISMo fallback if not changed (optional)
        if not changed and Registry.gismo:
            try:
                ranked = Registry.gismo.rank_substitutes(name, top_k=3)
                if ranked:
                    cand = ranked[0][0]
                    sub_info = Registry.substitutions_json.get(cand, {})
                    conv = f"{parsed.get('quantity') or ''} {parsed.get('unit') or ''} {sub_info.get('substitute')}".strip() if sub_info else conv
                    changed = True
            except Exception:
                pass

        status = "substituted" if changed else "no_substitute_found"
        subs_output.append({"original": l, "converted": conv, "status": status})

    # 4) Rewrite instructions using SHARE
    try:
        rewritten = Registry.share_rewriter.rewrite(req.instructions or "", subs_output)
    except Exception:
        rewritten = req.instructions or ""

    return {"parsed": parsed_output, "substitutions": subs_output, "rewritten": rewritten}


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

