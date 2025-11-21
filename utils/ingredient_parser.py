# utils/ingredient_parser.py
import re
from typing import Dict, Optional

# Common units (expandable)
UNITS = [
    "cup", "cups", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons", "tsp",
    "gram", "grams", "g", "kg", "kilogram", "kilograms", "ounce", "ounces", "oz",
    "pound", "pounds", "lb", "lbs", "ml", "l", "liter", "liters", "stick", "sticks",
    "clove", "cloves", "slice", "slices", "can", "cans"
]
UNIT_RE = re.compile(r"\b(" + r"|".join([re.escape(u) for u in UNITS]) + r")\b", re.I)

FRACTION_RE = re.compile(r"(?P<num>\d+)\s*(?P<frac>\d+/\d+)|(?P<simple>\d+/\d+)|(?P<float>\d+\.\d+)|(?P<int>\d+)")
PAREN_RE = re.compile(r"\(.*?\)")

PREP_SPLIT_RE = re.compile(r",|\s+-\s+|\s+or\s+|\s+and\s+")

def parse_ingredient_line(line: str) -> Dict[str, Optional[str]]:
    """
    Returns a dict:
      { "quantity": str or None,
        "unit": str or None,
        "ingredient": str or None,
        "preparation": str or None }
    """
    s = line.strip()
    # remove parenthetical weights at the end and keep them as note
    paren = PAREN_RE.findall(s)
    s_no_paren = PAREN_RE.sub("", s).strip()

    qty = None
    unit = None
    preparation = None
    ingredient = s_no_paren

    # find quantity (first fraction/number)
    m = FRACTION_RE.search(s_no_paren)
    if m:
        qty = m.group(0)
        ingredient = s_no_paren.replace(qty, "", 1).strip()

    # find unit
    m2 = UNIT_RE.search(ingredient)
    if m2:
        unit = m2.group(0)
        # remove the first unit occurrence
        ingredient = (ingredient[:m2.start()] + ingredient[m2.end():]).strip()

    # If ingredient contains commas, last part often prep
    parts = [p.strip() for p in PREP_SPLIT_RE.split(ingredient) if p.strip()]
    if len(parts) >= 2:
        # Heuristic: last segment is prep if it contains words like "chopped", "beaten", "mashed"
        last = parts[-1].lower()
        if any(k in last for k in ["chop", "minc", "beat", "mashed", "room temperature", "soften", "slice", "grate", "melt"]):
            preparation = parts[-1]
            ingredient = " ".join(parts[:-1])
        else:
            ingredient = " ".join(parts)

    # cleanup words like 'of' or connectors
    ingredient = re.sub(r"^\b(of|about|a|an)\b\s*", "", ingredient.strip(), flags=re.I)
    ingredient = re.sub(r"\s+", " ", ingredient).strip()

    # lower-case normalized ingredient
    ingredient_norm = ingredient.lower() if ingredient else None

    # include paren note if present
    paren_note = paren[0] if paren else None

    return {
        "quantity": qty,
        "unit": unit,
        "ingredient": ingredient_norm,
        "preparation": preparation,
        "note": paren_note
    }

# small helper to normalize ingredient tokens (strip extraneous punctuation)
def normalize_ingredient_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\-\s/']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # collapse "all purpose" to "all-purpose" for matching
    name = name.replace("all purpose", "all-purpose")
    return name

