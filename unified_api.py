# unified_api.py
"""
Unified backend API for Gluten-Free Converter.

Supports:
 - POST /process  with JSON { "ingredients": [...], "raw_text": "...", "url": "...", "instructions": "..." }
   - If `ingredients` is provided, treats that as line-by-line input.
   - Else if `raw_text` is provided, uses FoodNER to extract ingredient spans.
   - Else if `url` is provided, fetches the page and extracts ingredients & instructions heuristically.
 - Returns parsed items, normalized ingredient names, gluten flags, substitution results, and rewritten instructions.

Run:
    python -m uvicorn unified_api:app --reload --port 8000
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
import traceback

# Lazy imports / protected load so server can start successfully and report errors
def safe_import(name: str):
    try:
        mod = __import__(name, fromlist=['*'])
        return mod
    except Exception as e:
        raise ImportError(f"Failed to import {name}: {e}")


app = FastAPI(title="Unified Gluten-Free API")

# ---------- Request / Response schemas ----------
class ProcessRequest(BaseModel):
    # provide either ingredients (list of lines) OR raw_text (freeform recipe text) OR url (recipe url)
    ingredients: Optional[List[str]] = None
    raw_text: Optional[str] = None
    url: Optional[str] = None
    instructions: Optional[str] = ""


# ---------- Utilities: webpage extraction ----------
def extract_from_url(url: str):
    """Return (ingredient_lines, instructions_text) extracted heuristically from url content."""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        return [], ""

    # Try common ingredient selectors
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

    # fallback: find <li> with culinary tokens
    if not ing_lines:
        for li in soup.find_all("li"):
            txt = li.get_text(separator=" ", strip=True)
            if re.search(r'\b(cup|tsp|tbsp|ml|g|flour|sugar|salt|butter|egg|banana|chocolate)\b', txt, re.I):
                ing_lines.append(txt)

    # Extract instructions: common selectors
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
            texts = []
            for t in tags:
                texts.append(t.get_text(separator=" ", strip=True))
            instructions_text = "\n".join(texts).strip()
            if instructions_text:
                break

    # As another fallback, grab all <p> that look long and contain cooking verbs
    if not instructions_text:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        candidate = []
        for p in paragraphs:
            if re.search(r'\b(mix|bake|stir|heat|cook|preheat|fold|whisk|simmer|blend)\b', p, re.I):
                candidate.append(p)
        instructions_text = "\n".join(candidate[:10]).strip()

    # dedupe preserve order
    seen = set()
    clean = []
    for l in ing_lines:
        if l not in seen:
            seen.add(l)
            clean.append(re.sub(r'\s+', ' ', l).strip())

    return clean, instructions_text


# ---------- Lazy model loader ----------
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
        # import your modules
        mod_parser = safe_import("utils.ingredient_parser")
        mod_gismo = safe_import("utils.gismo")
        mod_foodner = safe_import("models.food_ner")
        mod_share = safe_import("models.share_rewriter")
        mod_gluten_check = safe_import("utils.gluten_check")
        mod_subpipeline = safe_import("substitution_pipeline")

        # instantiate
        Registry.ingredient_parser = getattr(mod_parser, "IngredientParser")()
        # FoodNER implementation might have different class names — try common names
        if hasattr(mod_foodner, "FoodNER"):
            Registry.foodner = getattr(mod_foodner, "FoodNER")()
        elif hasattr(mod_foodner, "FoodNERModel"):
            Registry.foodner = getattr(mod_foodner, "FoodNERModel")()
        else:
            Registry.foodner = None

        # GISMo class may be named GISMo or GISMoGraph or GISMoGraph — try a few
        if hasattr(mod_gismo, "GISMo"):
            Registry.gismo = getattr(mod_gismo, "GISMo")(mod_subpipeline.load_substitution_data() if hasattr(mod_subpipeline, "load_substitution_data") else mod_subpipeline.load_substitutions())
        elif hasattr(mod_gismo, "GISMoGraph"):
            Registry.gismo = getattr(mod_gismo, "GISMoGraph")(mod_subpipeline.load_substitution_data() if hasattr(mod_subpipeline, "load_substitution_data") else mod_subpipeline.load_substitutions())
        else:
            # fallback: try constructor without args
            try:
                Registry.gismo = mod_gismo.GISMo()
            except Exception:
                Registry.gismo = None

        # SubstitutionEngine
        # Try to load substitutions dict
        subs = None
        if hasattr(mod_subpipeline, "load_substitution_data"):
            subs = mod_subpipeline.load_substitution_data()
        elif hasattr(mod_subpipeline, "load_substitutions"):
            subs = mod_subpipeline.load_substitutions()
        else:
            # fallback to utils.gluten_check.load_substitutions
            subs = getattr(mod_gluten_check, "load_substitutions")()

        # instantiate substitution engine if available
        if hasattr(mod_subpipeline, "SubstitutionEngine"):
            Registry.substitution_engine = getattr(mod_subpipeline, "SubstitutionEngine")(subs)
        else:
            # If not present, create a tiny fallback wrapper
            class SimpleSubEngine:
                def __init__(self, subs): self.subs = subs
                def substitute(self, line):
                    # parse quantity and ingredient using parser
                    parsed = Registry.ingredient_parser.parse(line)
                    name = parsed.get("ingredient","") or ""
                    for g,k in self.subs.items():
                        if g in name:
                            out = {"quantity": parsed.get("quantity") or "", "unit": parsed.get("unit") or "", "ingredient": k.get("substitute", k)}
                            formatted = f"{out['quantity']} {out['unit']} {out['ingredient']}".strip()
                            return formatted, True
                    return line, False
            Registry.substitution_engine = SimpleSubEngine(subs)

        # SHARE rewriter
        if hasattr(mod_share, "SHARERewriter"):
            Registry.share_rewriter = getattr(mod_share, "SHARERewriter")()
        elif hasattr(mod_share, "ShareRewriter"):
            Registry.share_rewriter = getattr(mod_share, "ShareRewriter")()
        else:
            # fallback simple rewriter
            class SimpleRewriter:
                def rewrite(self, original_instructions, substitutions):
                    # naive find/replace based on substitutions list of dicts
                    text = original_instructions or ""
                    for s in substitutions:
                        try:
                            orig = s.get("original")
                            conv = s.get("converted") if isinstance(s.get("converted"), str) else s.get("converted", "")
                            if orig and conv:
                                # replace ingredient names heuristically
                                text = re.sub(re.escape(orig), conv, text, flags=re.I)
                        except Exception:
                            pass
                    return text
            Registry.share_rewriter = SimpleRewriter()

        # load gluten list
        if hasattr(mod_gluten_check, "load_gluten_ingredients"):
            Registry.gluten_list = getattr(mod_gluten_check, "load_gluten_ingredients")()
        else:
            Registry.gluten_list = []

        Registry.substitutions_json = subs

        Registry.loaded = True
        print("Unified API: models loaded.")
    except Exception as e:
        print("Error loading models:\n", e)
        print(traceback.format_exc())
        raise


# ---------- small helper ----------
def contains_gluten(ingredient_name: str) -> bool:
    name = (ingredient_name or "").lower()
    for g in Registry.gluten_list or []:
        if g in name:
            return True
    return False


# ---------- Main endpoint ----------
@app.post("/process")
def process(req: ProcessRequest):
    # Ensure models loaded
    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")

    # Resolve input lines
    lines: List[str] = []
    if req.ingredients:
        lines = [l for l in req.ingredients if l and l.strip()]
    elif req.raw_text:
        # run FoodNER extraction if available
        if Registry.foodner and hasattr(Registry.foodner, "extract_ingredients"):
            try:
                lines = Registry.foodner.extract_ingredients(req.raw_text)
            except Exception:
                # fallback: split on newlines and sentences
                lines = [s.strip() for s in re.split(r'\n|;', req.raw_text) if s.strip()]
        else:
            lines = [s.strip() for s in re.split(r'\n|;|\.', req.raw_text) if s.strip()]
    elif req.url:
        lines, extracted_instructions = extract_from_url(req.url)
        if not req.instructions and extracted_instructions:
            req.instructions = extracted_instructions
    else:
        raise HTTPException(status_code=400, detail="No input provided. Provide 'ingredients' or 'raw_text' or 'url'.")

    # parse each line with IngredientParser
    parsed = []
    parsed_objs = Registry.ingredient_parser.parse_batch(lines) if hasattr(Registry.ingredient_parser, "parse_batch") else [Registry.ingredient_parser.parse(l) for l in lines]
    for original_line, p in zip(lines, parsed_objs):
        parsed.append({"original": original_line, "parsed": p})

    # detect gluten and substitute
    substitutions = []
    for orig_line in lines:
        parsed_line = Registry.ingredient_parser.parse(orig_line)
        ingredient_name = (parsed_line.get("ingredient") or "").lower()

        if not ingredient_name:
            substitutions.append({
                "original": orig_line,
                "converted": orig_line,
                "status": "not_parsed"
            })
            continue

        is_gluten = contains_gluten(ingredient_name)

        if not is_gluten:
            substitutions.append({
                "original": orig_line,
                "converted": orig_line,
                "status": "gluten_free"
            })
            continue

        # Try substitution engine first
        try:
            conv, changed = Registry.substitution_engine.substitute(orig_line)
        except Exception:
            conv, changed = orig_line, False

        # If substitution engine couldn't, try GISMo if available
        if not changed and Registry.gismo is not None:
            try:
                # GISMo best_substitute naming may vary
                if hasattr(Registry.gismo, "best_substitute"):
                    cand = Registry.gismo.best_substitute(ingredient_name)
                    # best_substitute might return tuple (cand,score)
                    if isinstance(cand, tuple) and cand[0]:
                        best_name = cand[0]
                        # find substitute entry in substitutions json
                        sub_info = Registry.substitutions_json.get(best_name, None) if isinstance(Registry.substitutions_json, dict) else None
                        if sub_info:
                            # format
                            parsed_for_qty = parsed_line
                            qty = parsed_for_qty.get("quantity") or ""
                            unit = parsed_for_qty.get("unit") or ""
                            formatted = f"{qty} {unit} {sub_info.get('substitute')}".strip()
                            conv, changed = formatted, True
                elif hasattr(Registry.gismo, "rank_substitutes"):
                    ranked = Registry.gismo.rank_substitutes(ingredient_name, top_k=3)
                    if ranked:
                        cand_name = ranked[0][0]
                        sub_info = Registry.substitutions_json.get(cand_name, None) if isinstance(Registry.substitutions_json, dict) else None
                        if sub_info:
                            parsed_for_qty = parsed_line
                            qty = parsed_for_qty.get("quantity") or ""
                            unit = parsed_for_qty.get("unit") or ""
                            formatted = f"{qty} {unit} {sub_info.get('substitute')}".strip()
                            conv, changed = formatted, True
            except Exception:
                pass

        status = "substituted" if changed else "no_substitute_found"
        substitutions.append({
            "original": orig_line,
            "converted": conv,
            "status": status
        })

    # rewrite instructions with SHARE rewriter (pass substitutions list)
    try:
        rewritten = Registry.share_rewriter.rewrite(req.instructions or "", substitutions)
    except Exception:
        rewritten = None

    return {
        "parsed": parsed,
        "substitutions": substitutions,
        "rewritten": rewritten
    }

