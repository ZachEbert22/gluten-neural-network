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
    Clean ingredient text without breaking apart valid tokens.
    Ensures spaces between numbers, units, and words.
    """
    if not isinstance(s, str):
        return ""
    # Normalize fractions and symbols
    s = s.replace("⁄", "/").replace("½", "1/2").replace("¼", "1/4").replace("¾", "3/4")
    s = re.sub(r"[\u00A0\u200B\u2009]", " ", s)

    # ✅ Insert missing spaces between numbers and letters (e.g. 200gplain → 200 g plain)
    s = re.sub(r"(?<=\d)([a-zA-Z])", r" \1", s)
    s = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", s)
    s = re.sub(r"([a-zA-Z])([A-Z])", r"\1 \2", s)

    # Remove weird punctuation but keep fractions and dashes
    s = re.sub(r"[^a-zA-Z0-9/\-\s]", "", s)
    # Normalize spacing
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

