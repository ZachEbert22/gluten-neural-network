def substitute_ingredient(parsed_ingredient, substitutions):
    """
    parsed_ingredient: dict with keys 'quantity','unit','ingredient'
    substitutions: mapping from gluten-key -> {substitute, ratio}
    """
    name = parsed_ingredient["ingredient"].lower()
    try:
        quantity = float(parsed_ingredient.get("quantity") or 1)
    except Exception:
        quantity = 1.0
    unit = parsed_ingredient.get("unit", "")

    for gluten_item, details in substitutions.items():
        if gluten_item in name:
            ratio = details.get("ratio", 1.0)
            new_quantity = round(quantity * ratio, 2)
            return {
                "quantity": str(new_quantity),
                "unit": unit,
                "ingredient": details["substitute"]
            }

    # No substitution
    return {
        "quantity": str(parsed_ingredient.get("quantity", "1")),
        "unit": parsed_ingredient.get("unit", ""),
        "ingredient": parsed_ingredient.get("ingredient", "")
    }


def substitution_accuracy(original, modified, gluten_ingredients):
    """
    Measures fraction of gluten ingredients successfully replaced.
    """
    correct_replacements = 0
    total_gluten_items = 0
    for o, m in zip(original, modified):
        if any(g in o.lower() for g in gluten_ingredients):
            total_gluten_items += 1
            if not any(g in m.lower() for g in gluten_ingredients):
                correct_replacements += 1
    if total_gluten_items == 0:
        return 1.0
    return correct_replacements / total_gluten_items


def substitution_coverage(original, modified):
    """
    Measures fraction of all ingredients that were changed (coverage).
    """
    total = len(original)
    replaced = sum(1 for o, m in zip(original, modified) if o != m)
    return replaced / total if total > 0 else 0

