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

    "wholemeal flour": "flour",
    "whole meal flour": "flour",
    "whole wheat meal flour": "flour",

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
    """Apply override normalization rules to ingredient name."""
    ing = ingredient.lower()
    for key, val in NORMALIZATION_OVERRIDES.items():
        if key in ing:
            return val
    return ing

