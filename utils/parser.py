import re
from utils.substitution import substitute_ingredient, substitution_accuracy, substitution_coverage
import unicodedata

# Valid units for parsing
UNITS = [
    "cup", "cups", "tbsp", "tbsp.", "tbsps", "tsp", "tsp.", "g", "kg",
    "ml", "l", "oz", "lb", "pinch", "dash"
]

def clean_text(s: str) -> str:
    """
    Normalize Unicode and fix split characters like 'mil k' → 'milk'
    without merging valid pairs like 'soy sauce'.
    """
    if not isinstance(s, str):
        s = str(s)

    # Normalize and remove invisible unicode
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", s)

    # Fix accidental single-letter splits: only merge consonant + consonant (e.g. 'eggplan t')
    s = re.sub(r"([bcdfghjklmnpqrtvwxyz])\s+([bcdfghjklmnpqrtvwxyz])", r"\1\2", s, flags=re.IGNORECASE)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ---------------------------
# Ingredient Parser
# ---------------------------
def parse_ingredient(ingredient_line):
    ingredient_line = ingredient_line.strip()
    # Allow optional quantity and unit
    pattern = r'(?:(?P<quantity>\d+(\.\d+)?)(?:\s+(?P<unit>[^\d\s]+))?\s*(?:of\s+)?)?(?P<ingredient>.+)'
    match = re.match(pattern, ingredient_line, re.IGNORECASE)
    if not match:
        return {"quantity": "1", "unit": "", "ingredient": ingredient_line.strip()}

    data = match.groupdict()
    quantity = data.get("quantity") or "1"
    unit = data.get("unit") or ""
    ingredient = data.get("ingredient") or ""
    ingredient = ingredient.strip()

    # If unit is invalid, fold it into ingredient
    if unit and unit.lower() not in UNITS:
        ingredient = f"{unit} {ingredient}".strip()
        unit = ""

    ingredient = clean_text(ingredient)
    return {"quantity": quantity, "unit": unit, "ingredient": ingredient}


# ---------------------------
# Formatting
# ---------------------------
def format_ingredient(parsed_ingredient):
    """Convert parsed ingredient dict → readable string."""
    quantity = str(parsed_ingredient.get("quantity", "")).strip()
    unit = str(parsed_ingredient.get("unit", "")).strip()
    ingredient = str(parsed_ingredient.get("ingredient", "")).strip()

    if unit:
        return f"{quantity} {unit} of {ingredient}"
    else:
        return f"{quantity} {ingredient}"
    
    return clean_text(result_string)


# ---------------------------
# Main Processing Pipeline
# ---------------------------
def process_recipe(ingredients, gluten_ingredients, substitutions):
    """
    Apply substitutions to a list of ingredient strings, print accuracy & coverage.
    """
    parsed_list = [parse_ingredient(i) for i in ingredients]
    substituted = [substitute_ingredient(p, substitutions) for p in parsed_list]
    gluten_free = [format_ingredient(i) for i in substituted]

    # Fix: only pass 2 args to substitution_coverage()
    acc = substitution_accuracy(ingredients, gluten_free, gluten_ingredients)
    cov = substitution_coverage(ingredients, gluten_free)

    print("\n--- Ingredient Substitution Results ---")
    for before, after in zip(ingredients, gluten_free):
        print(f"  {before:<40} →  {after}")
    print("----------------------------------------")
    print(f"Substitution Accuracy: {acc * 100:.2f}%")
    print(f"Substitution Coverage: {cov * 100:.2f}%\n")

    return gluten_free

