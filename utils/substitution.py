def substitute_ingredient(parsed_ingredient, substitutions):
    name = parsed_ingredient["ingredient"].lower()
    quantity = float(parsed_ingredient["quantity"] or 1)
    unit = parsed_ingredient["unit"]

    # Normalize flour types (plain, self-raising, etc.)
    if "flour" in name and "gluten-free" not in name:
        # Match to closest key (default to wheat flour)
        key = "wheat flour" if "wheat" in substitutions else list(substitutions.keys())[0]
        details = substitutions.get(key, {"substitute": "almond flour", "ratio": 0.75})
        ratio = details.get("ratio", 1.0)
        new_quantity = round(quantity * ratio, 2)
        return {"quantity": str(new_quantity), "unit": unit, "ingredient": details["substitute"]}

    # Otherwise, try direct matches
    for gluten_item, details in substitutions.items():
        if gluten_item in name:
            ratio = details.get("ratio", 1.0)
            new_quantity = round(quantity * ratio, 2)
            return {
                "quantity": str(new_quantity),
                "unit": unit,
                "ingredient": details["substitute"],
            }

    return parsed_ingredient

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

