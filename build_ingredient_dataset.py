#!/usr/bin/env python3
"""
build_ingredient_dataset_kaggle.py

Robust dataset builder for the Food.com Kaggle dataset
(irkaal/foodcom-recipes-and-reviews).

Outputs:
  - data/ingredient_dataset.csv
  - data/ingredient_classifier_dataset.csv

Both files have columns: text,label
label: 1 => ingredient, 0 => non-ingredient (review text)

Notes:
  - Set MAX_POS_SAMPLES and MAX_NEG_SAMPLES environment variables to limit sizes
    (useful when testing).
  - If you want to inspect the downloaded files, set DEBUG_LIST_DIR=True.
"""

import os
import re
import ast
import random
import argparse
from typing import List, Optional

import pandas as pd

try:
    import kagglehub
except Exception:
    kagglehub = None

# Output paths (kept compatible with existing code)
OUT_PATH_1 = "data/ingredient_dataset.csv"
OUT_PATH_2 = "data/ingredient_classifier_dataset.csv"
os.makedirs("data", exist_ok=True)

# Measurement tokens and common ingredient words used to bias detection
MEASUREMENT_TOKENS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "g", "gram", "grams", "kg", "kilogram", "ounce", "oz", "ml", "liter", "litre",
    "pound", "lb", "slice", "slices", "pinch", "dash"
}
SPOONY = {"salt", "sugar", "butter", "flour", "egg", "eggs", "oil", "milk", "vanilla", "yeast", "baking", "soda", "powder"}

# Simple regex to detect fractions and quantities (e.g. "1/2", "2 1/2", "2.5")
QUANTITY_RE = re.compile(r"(^|\s)(\d+([.,]\d+)?)(\s+\d+/\d+)?(\s|$)|\d+/\d+")

# Acceptable max/min word counts
MIN_WORDS = 1
MAX_WORDS = 12


def looks_like_ingredient(x: Optional[str]) -> bool:
    """
    Heuristic for ingredient-like strings.
    - Accepts short lines (1..MAX_WORDS words)
    - Rejects lines with multiple sentences (contain ".")
    - Accepts lines with numeric quantities or measurement tokens or common ingredient words
    - Accepts lines that look like ingredient phrases (no long sentences)
    """
    if x is None:
        return False
    if not isinstance(x, str):
        return False

    s = x.strip()
    if len(s) == 0:
        return False

    # Reject if appears to be a sentence (many words AND period)
    if "." in s:
        # sometimes ingredients have '.' (rare) but mostly not
        return False

    words = s.split()
    wcount = len(words)

    if wcount < MIN_WORDS or wcount > MAX_WORDS:
        return False

    low = s.lower()

    # If contains quantity or fraction => ingredient
    if QUANTITY_RE.search(s):
        return True

    # If contains a measurement token or common ingredient word => ingredient
    for token in MEASUREMENT_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", low):
            return True

    for token in SPOONY:
        if re.search(rf"\b{re.escape(token)}\b", low):
            return True

    # Many ingredient lines have parentheses, commas (e.g. "1 cup flour (sifted)"), commas alone are OK.
    # If there's a comma but the text is short, treat as ingredient too
    if "," in s and wcount <= 8:
        return True

    # If the whole string is short (1-3 words) and looks like a food (heuristic):
    # Accept if contains only letters/hyphens and no sentence punctuation.
    if 1 <= wcount <= 3:
        # ensure not a multi-word sentence fragment (no verbs heuristic is hard),
        # but allow common food tokens: approximate by letters, hyphens, slashes.
        if re.fullmatch(r"[A-Za-z0-9\s\-'/&()]+", s):
            return True

    # Otherwise, treat as non-ingredient
    return False


def normalize_part(p) -> Optional[str]:
    """Return a cleaned string for a recipe 'part' if possible, else None."""
    if p is None:
        return None
    if isinstance(p, str):
        s = p.strip()
        if s == "":
            return None
        return s
    # if it's numeric or other type, convert to str
    return str(p).strip()


def load_foodcom(download: bool = True, path_override: Optional[str] = None):
    """
    Downloads (if needed) and loads recipes.csv and reviews.csv.

    Returns: (df_recipes, df_reviews, dataset_dir)
    """
    print("Loading Food.com dataset via kagglehub…")
    if path_override:
        path = path_override
    else:
        if kagglehub is None:
            raise RuntimeError(
                "kagglehub is not installed or could not be imported. "
                "Install it or provide path_override to an existing dataset location."
            )
        path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    print("Dataset path:", path)

    recipes_path = os.path.join(path, "recipes.csv")
    reviews_path = os.path.join(path, "reviews.csv")

    # Show files for debugging if requested
    if os.environ.get("DEBUG_LIST_DIR", "").lower() in ("1", "true", "yes"):
        print("Files in dataset dir:", os.listdir(path))

    if not os.path.exists(recipes_path):
        raise RuntimeError(f"recipes.csv not found. Found: {os.listdir(path)}")

    if not os.path.exists(reviews_path):
        raise RuntimeError(f"reviews.csv not found. Found: {os.listdir(path)}")

    df_recipes = pd.read_csv(recipes_path)
    df_reviews = pd.read_csv(reviews_path)

    print("Loaded:", df_recipes.shape, df_reviews.shape)
    return df_recipes, df_reviews, path


def build_dataset(
    max_pos: Optional[int] = None,
    max_neg: Optional[int] = None,
    shuffle_seed: int = 42
):
    df_recipes, df_reviews, dataset_dir = load_foodcom()

    # Ensure column exists for recipe ingredient parts
    # In the repository's older code the column was 'RecipeIngredientParts'
    possible_recipe_cols = ["RecipeIngredientParts", "Ingredients", "ingredients", "recipeIngredientParts"]
    recipe_col = None
    for c in possible_recipe_cols:
        if c in df_recipes.columns:
            recipe_col = c
            break
    if recipe_col is None:
        raise RuntimeError(
            "Could not find recipe ingredient column in recipes.csv. "
            f"Available columns: {df_recipes.columns.tolist()}"
        )
    print(f"Using recipe parts column: '{recipe_col}'")

    # Determine review column (case-insensitive)
    possible_review_cols = ["review", "Review", "review_text", "text", "comment"]
    review_col = None
    for c in possible_review_cols:
        if c in df_reviews.columns:
            review_col = c
            break
    if review_col is None:
        raise RuntimeError(
            "Could not find review text column in reviews.csv. "
            f"Available columns: {df_reviews.columns.tolist()}"
        )
    print(f"Using review column: '{review_col}'")

    # -------------------------
    # Extract positive samples
    # -------------------------
    pos_samples: List[str] = []
    for raw in df_recipes[recipe_col].dropna():
        # raw can be a Python list (object), or a stringified list "['a','b']"
        parts = None
        if isinstance(raw, str):
            # Try literal_eval (stringified list) first
            try:
                parts_candidate = ast.literal_eval(raw)
                # sometimes literal_eval returns a string if it's not a list; guard
                if isinstance(parts_candidate, (list, tuple)):
                    parts = parts_candidate
                else:
                    # maybe it's a single ingredient stored as a string
                    parts = [parts_candidate]
            except Exception:
                # Fallback: assume it's a newline/comma separated list or single line
                # Try splitting on "||" or newlines or commas between items if it looks like multiple parts
                if "\n" in raw:
                    parts = [p.strip() for p in raw.splitlines() if p.strip()]
                elif "||" in raw:
                    parts = [p.strip() for p in raw.split("||") if p.strip()]
                elif "," in raw and len(raw.split(",")) <= 40:
                    # avoid splitting long descriptive fields
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                else:
                    parts = [raw.strip()]
        else:
            # if it's already a list/tuple, use directly
            if isinstance(raw, (list, tuple)):
                parts = raw
            else:
                parts = [str(raw).strip()]

        # Ensure parts is iterable list-like
        if not isinstance(parts, (list, tuple)):
            continue

        for part in parts:
            s = normalize_part(part)
            if s is None:
                continue
            if looks_like_ingredient(s):
                pos_samples.append(s)

    print("Positive samples collected:", len(pos_samples))

    # -------------------------
    # Extract negative samples (reviews)
    # -------------------------
    neg_samples: List[str] = []
    # ensure review column is str
    df_reviews = df_reviews.copy()
    df_reviews[review_col] = df_reviews[review_col].astype(str)
    for txt in df_reviews[review_col].dropna().tolist():
        t = txt.strip()
        if t == "":
            continue
        # skip if it looks like an ingredient line (we want non-ingredient review sentences)
        if looks_like_ingredient(t):
            continue
        # also skip very short fragments that are noise (<3 chars)
        if len(t) < 3:
            continue
        neg_samples.append(t)

    print("Negative candidate samples found:", len(neg_samples))

    # If sizes are huge and environment variables set, trim for practical use
    env_max_pos = os.environ.get("MAX_POS_SAMPLES")
    env_max_neg = os.environ.get("MAX_NEG_SAMPLES")
    if env_max_pos:
        try:
            max_pos = int(env_max_pos)
        except:
            pass
    if env_max_neg:
        try:
            max_neg = int(env_max_neg)
        except:
            pass

    # If not specified, cap to sane default for first runs to avoid memory explosions.
    # If you want the full dataset, set MAX_POS_SAMPLES="" (unset) in env.
    if max_pos is None:
        # default: use all positives (could be large)
        max_pos = None
    if max_neg is None:
        max_neg = None

    # Shuffle and trim negatives to match positive size (balanced)
    random.seed(shuffle_seed)
    random.shuffle(neg_samples)

    if max_pos is not None:
        pos_samples = pos_samples[:max_pos]
    if max_neg is not None:
        neg_samples = neg_samples[:max_neg]

    # Balance: make negatives equal to positives (trim if needed)
    if len(pos_samples) == 0:
        print("Warning: 0 positive samples collected. Check ingredient parsing heuristics.")
    target_neg = len(pos_samples)
    if target_neg == 0:
        neg_samples = []
    else:
        if len(neg_samples) >= target_neg:
            neg_samples = neg_samples[:target_neg]
        else:
            # If not enough negatives, allow duplicates (but try to sample from whole set)
            # This prevents empty outputs but warns the user
            print("Warning: not enough negative samples to fully balance dataset; "
                  "will allow repeats to balance.")
            # expand by sampling with replacement
            while len(neg_samples) < target_neg:
                neg_samples.extend(random.choices(neg_samples, k=min(len(neg_samples), target_neg - len(neg_samples))))
            neg_samples = neg_samples[:target_neg]

    print("Final positives:", len(pos_samples), "Final negatives:", len(neg_samples))

    # Build dataframe and save (shuffled)
    df_out = pd.DataFrame({
        "text": list(pos_samples) + list(neg_samples),
        "label": [1] * len(pos_samples) + [0] * len(neg_samples)
    }).sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    # Save to both filenames for compatibility
    df_out.to_csv(OUT_PATH_1, index=False)
    df_out.to_csv(OUT_PATH_2, index=False)
    print(f"Saved dataset → {OUT_PATH_1} and {OUT_PATH_2}")
    print(df_out.head(10))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ingredient dataset from Food.com Kaggle dataset")
    parser.add_argument("--path", type=str, default=None, help="If provided, skip kagglehub download and use this local dataset path")
    parser.add_argument("--max-pos", type=int, default=None, help="Limit number of positive samples (for quick tests)")
    parser.add_argument("--max-neg", type=int, default=None, help="Limit number of negative samples (for quick tests)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")

    args = parser.parse_args()
    build_dataset(max_pos=args.max_pos, max_neg=args.max_neg, shuffle_seed=args.seed)

