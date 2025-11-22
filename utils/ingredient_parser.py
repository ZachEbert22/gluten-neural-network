# utils/ingredient_parser.py
import re
from typing import Dict, Optional, List


# -----------------------------
# UNIT DEFINITIONS
# -----------------------------
UNITS = [
    "cup", "cups", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons", "tsp",
    "gram", "grams", "g", "kg", "kilogram", "kilograms", "ounce", "ounces", "oz",
    "pound", "pounds", "lb", "lbs", "ml", "l", "liter", "liters",
    "stick", "sticks", "clove", "cloves", "slice", "slices", "can", "cans"
]

UNIT_RE = re.compile(r"\b(" + r"|".join([re.escape(u) for u in UNITS]) + r")\b", re.I)
FRACTION_RE = re.compile(
    r"(?P<mixed>\d+\s+\d+/\d+)|"      # 1 1/2
    r"(?P<fraction>\d+/\d+)|"         # 1/2
    r"(?P<float>\d*\.\d+)|"           # .75 or 0.75 or 1.25
    r"(?P<int>\d+)"                   # 1 or 2 or 3
)
PAREN_RE = re.compile(r"\(.*?\)")
PREP_SPLIT_RE = re.compile(r",|\s+-\s+|\s+or\s+|\s+and\s+")


# -----------------------------
# FUNCTIONS
# -----------------------------
def parse_ingredient_line(line: str) -> Dict[str, Optional[str]]:
    """
    Parse a single ingredient line:
    - quantity
    - unit
    - ingredient name
    - preparation note ("chopped", "beaten", etc.)
    """

    s = line.strip()

    # capture parenthetical notes (ex: "(247 grams)")
    paren = PAREN_RE.findall(s)
    s_no_paren = PAREN_RE.sub("", s).strip()

    qty = None
    unit = None
    preparation = None
    ingredient = s_no_paren

    # detect quantity
    m = FRACTION_RE.search(s_no_paren)
    if m:
        qty = m.group(0)
        start, end = m.span()
        ingredient = s_no_paren[end:].strip()

    # detect unit
    m2 = UNIT_RE.search(ingredient)
    if m2:
        unit = m2.group(0)
        ingredient = (ingredient[:m2.start()] + ingredient[m2.end():]).strip()

    # detect preparation ("chopped", "beaten", "mashed")
    parts = [p.strip() for p in PREP_SPLIT_RE.split(ingredient) if p.strip()]
    if len(parts) >= 2:
        last = parts[-1].lower()
        if any(k in last for k in [
            "chop", "minc", "beat", "mashed",
            "room temperature", "soften", "slice", "grate", "melt"
        ]):
            preparation = parts[-1]
            ingredient = " ".join(parts[:-1])
        else:
            ingredient = " ".join(parts)

    # clean ingredient
    ingredient = re.sub(r"^\b(of|about|a|an)\b\s*", "", ingredient.strip(), flags=re.I)
    ingredient = re.sub(r"\s+", " ", ingredient).strip()

    # normalize form
    ingredient_norm = ingredient.lower() if ingredient else None
    ingredient_norm = normalize_ingredient_name(ingredient_norm)

    return {
        "quantity": qty,
        "unit": unit,
        "ingredient": ingredient_norm,
        "preparation": preparation,
        "note": paren[0] if paren else None
    }


def normalize_ingredient_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\-\s/']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # collapse "all purpose" to "all-purpose"
    name = name.replace("all purpose", "all-purpose")
    return name


# -----------------------------
# CLASS WRAPPER (Fixes Your Backend Import)
# -----------------------------
class IngredientParser:
    """
    Clean wrapper around the parsing functions.
    Backend imports: IngredientParser()
    """

    def parse(self, line: str) -> Dict[str, Optional[str]]:
        return parse_ingredient_line(line)

    def parse_batch(self, lines: List[str]) -> List[Dict[str, Optional[str]]]:
        return [parse_ingredient_line(l) for l in lines]

