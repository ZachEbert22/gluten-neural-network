import re

NORMALIZATION_OVERRIDES = {
    # FLOURS
    "all-purpose flour": "flour",
    "all purpose flour": "flour",
    "plain flour": "flour",
    "ap flour": "flour",
    "wheat flour": "flour",
    "cake flour": "flour",
    "pastry flour": "flour",
    "bread flour": "flour",
    "wheat flour": "flour",
    "wholemeal flour": "flour",
    "white flour": "flour",
    "self rising flour": "flour",
    "self-raising flour": "flour",
    "flour": "flour",
    "wholemeal": "flour",
    "whole wheat": "flour",
    "whole wheat flour": "flour",
    "whole-meal flour": "flour",
    "whole meal": "flour",
    "white ap flour": "flour",
    "ap": "flour",
    "ap flour": "flour",
    "white ap": "flour",
    "whole meal flour": "wholemeal flour",
    "whole wheat meal flour": "wholemeal flour",

    "bread crumb": "bread-crumb",
    "bread-crumb": "bread-crumb",
    "breadcrumb": "bread-crumb",

    # SUGARS
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "brown sugar": "sugar",

    # BUTTER
    "unsalted butter": "butter",
    "salted butter": "butter",
    "butter (unsalted)": "butter",

    # BANANAS
    "ripe banana": "banana",
    "very ripe banana": "banana",
    "well-mashed banana": "banana",

    # CHOCOLATE
    "dark chocolate": "chocolate",
    "milk chocolate": "chocolate",
    "chocolate chunks": "chocolate",
    "chocolate chips": "chocolate",
}


def normalize_ingredient(ingredient: str) -> str:
    ing = ingredient.lower().strip()

    # Exact matches first
    if ing in NORMALIZATION_OVERRIDES:
        return NORMALIZATION_OVERRIDES[ing]

    # Word-boundary fuzzy matching
    for key, val in NORMALIZATION_OVERRIDES.items():
        if re.search(rf"\b{re.escape(key)}\b", ing):
            return val

    # Return base ingredient (last word fallback: "flour")
    if ing.endswith("flour"):
        return "flour"
    if ing.endswith("breadcrumbs") or ing.endswith("breadcrumb"):
        return "bread-crumb"

    return ing

