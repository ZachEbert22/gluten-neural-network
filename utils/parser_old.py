import re
from utils.substitution import substitute_ingredient, substitution_accuracy, substitution_coverage

# units considered valid
UNITS = [
    "cup", "cups", "tbsp", "tbsp.", "tbsps", "tsp", "tsp.", "g", "kg", "ml", "l", "oz", "lb", "pinch", "dash"
]

def parse_ingredient(ingredient_line):
    ingredient_line = ingredient_line.strip()
    # allow quantity to be missing; make quantity optional
    pattern = r'(?:(?P<quantity>\d+(\.\d+)?)(?:\s+(?P<unit>[^\d\s]+))?\s*(?:of\s+)?)?(?P<ingredient>.+)'
    match = re.match(pattern, ingredient_line, re.IGNORECASE)
    if not match:
        return {"quantity": "1", "unit": "", "ingredient": ingredient_line.strip()}

    data = match.groupdict()
    quantity = data.get("quantity") or "1"
    unit = data.get("unit") or ""
    ingredient = data.get("ingredient") or ""
    ingredient = ingredient.strip()

    # if unit is present but not in UNITS, move it into ingredient
    if unit and unit.lower() not in UNITS:
        ingredient = f"{unit} {ingredient}".strip()
        unit = ""

    return {"quantity": quantity, "unit": unit, "ingredient": ingredient}

def format_ingredient(parsed_ingredient):
    """
    Convert parsed ingredient to string.
    Add 'of' when unit present.
    """
    quantity = str(parsed_ingredient.get("quantity", "")).strip()
    unit = str(parsed_ingredient.get("unit", "")).strip()
    ingredient = str(parsed_ingredient.get("ingredient", "")).strip()

    if unit:
        return f"{quantity} {unit} of {ingredient}"
    else:
        return f"{quantity} {ingredient}"

def process_recipe(ingredients, gluten_ingredients, substitutions):
    """
    ingredients: list of ingredient strings
    returns: list of formatted gluten-free ingredients (applies substitution map)
    """
    parsed_list = [parse_ingredient(i) for i in ingredients]
    substituted = [substitute_ingredient(p, substitutions) for p in parsed_list]
    gluten_free = [format_ingredient(i) for i in substituted]

    acc = substitution_accuracy(ingredients, gluten_free, gluten_ingredients)
    cov = substitution_coverage(ingredients, gluten_free, gluten_ingredients)

    print("\n--- Ingredient Substitution Results ---")
    for before, after in zip(ingredients, gluten_free):
        print(f"  {before:<40} →  {after}")
    print("----------------------------------------")
    print(f"Substitution Accuracy: {acc * 100:.2f}%")
    print(f"Substitution Coverage: {cov * 100:.2f}%\n")

    return gluten_free

