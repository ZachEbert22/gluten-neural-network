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

# ---------------- MODELS ----------------------

class RecipeRequest(BaseModel):
    raw_text: str | None = None
    recipe_url: str | None = None

class RecipeResponse(BaseModel):
    ingredients: list
    instructions: list
    rewritten: str
    warnings: list

# -------------- SCRAPER -----------------------

def scrape_url(url: str) -> str:
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text("\n")

# -------------- PARSER ------------------------

def extract_from_text(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    ingredient_candidates = []
    instruction_candidates = []

    for line in lines:
        # Ingredient parser
        parsed = parse_ingredient_line(line)

        if parsed["ingredient"]:
            ingredient_candidates.append(parsed)
            continue

        # FoodNER fallback
        ner_items = ner.extract(line)
        if ner_items:
            ingredient_candidates.extend(ner_items)
            continue

        # Otherwise it's an instruction
        instruction_candidates.append(line)

    return ingredient_candidates, instruction_candidates

# -------------- UNIFIED ENDPOINT --------------

@app.post("/api/parse_recipe", response_model=RecipeResponse)
def parse_recipe(req: RecipeRequest):

    if not req.raw_text and not req.recipe_url:
        return {"error": "No input provided"}

    # Get text source
    if req.recipe_url:
        raw = scrape_url(req.recipe_url)
    else:
        raw = req.raw_text

    # Extract
    ingredients, instructions = extract_from_text(raw)

    # GISMo substitution engine
    enriched = gismo.suggest_substitutions(ingredients)

    # SHARE transformer rewrite
    rewritten = rewriter.rewrite(instructions, enriched)

    return RecipeResponse(
        ingredients=enriched,
        instructions=instructions,
        rewritten=rewritten,
        warnings=[]
    )

