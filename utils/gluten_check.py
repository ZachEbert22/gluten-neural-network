import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_gluten_ingredients():
    path = DATA_DIR / "ingredients.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_substitutions():
    path = DATA_DIR / "substitutions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

