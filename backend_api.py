# backend_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from utils.ingredient_parser import parse_ingredient_line
from models.food_ner import FoodNERTagger
from utils.gismo import GISMoEngine
from models.share_rewriter import SHARERewriter
import requests
from bs4 import BeautifulSoup

app = FastAPI()

ner = FoodNERTagger()
gismo = GISMoEngine()
rewriter = SHARERewriter()

class RecipeRequest(BaseModel):
    raw_text: str | None = None
    recipe_url: str | None = None

class RecipeResponse(BaseModel):
    ingredients: list
    instructions: list
    rewritten: str
    warnings: list

# ---- URL SCRAPER -------------------------------------------------

def scrape_url(url: str) -> str:
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text("\n")
    return text

# ---- NATURAL TEXT PARSER -----------------------------------------

def extract_from_text(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    ingredient_candidates = []
    instruction_candidates = []

    for line in lines:
        # 1) Ingredient Parser (your custom line-by-line)
        parsed = parse_ingredient_line(line)

        if parsed["ingredient"]:
            ingredient_candidates.append(parsed)
            continue

        # 2) FoodNER fallback
        ner_items = ner.extract(line)
        if ner_items:
            ingredient_candidates.extend(ner_items)
            continue

        # Otherwise assume instructions
        instruction_candidates.append(line)

    return ingredient_candidates, instruction_candidates


# ---- UNIFIED ENDPOINT --------------------------------------------

@app.post("/api/parse_recipe", response_model=RecipeResponse)
def unify_recipe_parser(req: RecipeRequest):

    if not req.raw_text and not req.recipe_url:
        return {"error": "No input provided"}

    # --- Get text ---
    if req.recipe_url:
        try:
            raw = scrape_url(req.recipe_url)
        except Exception as e:
            return {"error": f"Failed to scrape URL: {e}"}
    else:
        raw = req.raw_text

    # --- Parse text ---
    ingredients, instructions = extract_from_text(raw)

    # --- Run GISMo substitution graph ---
    enriched = gismo.suggest_substitutions(ingredients)

    # --- Rewrite instructions with SHARE transformer ---
    rewritten = rewriter.rewrite(instructions, enriched)

    return RecipeResponse(
        ingredients=enriched,
        instructions=instructions,
        rewritten=rewritten,
        warnings=[]
    )

