import kagglehub
import pandas as pd
import re
import random
import os
import ast

OUT_PATH = "data/ingredient_dataset.csv"
os.makedirs("data", exist_ok=True)

# Ingredient detection regex
ING_PATTERN = re.compile(
    r"(cup|tbsp|tsp|g|kg|oz|ml|salt|flour|sugar|oil|butter|egg|spice|milk|yeast|baking|soda|powder|vanilla|onion|garlic|pepper|water)",
    re.I
)

def looks_like_ingredient(x: str) -> bool:
    if not isinstance(x, str):
        return False
    if len(x.split()) < 2:
        return False
    return ING_PATTERN.search(x) is not None


def load_foodcom():
    """
    Loads the REAL filenames from Food.com Kaggle dataset.
    """
    print("Downloading Food.com dataset via kagglehub…")
    path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    print("Dataset path:", path)

    recipes_path = f"{path}/recipes.csv"
    reviews_path = f"{path}/reviews.csv"

    if not os.path.exists(recipes_path):
        raise RuntimeError(f"recipes.csv not found. Found: {os.listdir(path)}")

    df_recipes = pd.read_csv(recipes_path)
    df_reviews = pd.read_csv(reviews_path)

    print("Loaded:", df_recipes.shape, df_reviews.shape)
    return df_recipes, df_reviews


def build_dataset():
    df_recipes, df_reviews = load_foodcom()

    # -------------------------------
    # CHECK FOR REAL COLUMNS
    # -------------------------------
    if "RecipeIngredientParts" not in df_recipes.columns:
        raise RuntimeError(
            "ERROR: Expected 'RecipeIngredientParts' in recipes.csv\n"
            f"Columns found: {df_recipes.columns}"
        )

    print("Extracting ingredient lines…")

    pos_samples = []

    # ------------------------------------
    # PARSE LIST-LIKE STRING FIELDS
    # ------------------------------------
    for raw in df_recipes["RecipeIngredientParts"].dropna():
        try:
            parts = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except:
            continue

        if not isinstance(parts, (list, tuple)):
            continue

        for ing in parts:
            if looks_like_ingredient(ing):
                pos_samples.append(ing)

    print("Positive samples collected:", len(pos_samples))

    # --------------------------
    # NEGATIVE SAMPLES (reviews)
    # --------------------------
    neg_samples = [
        txt for txt in df_reviews["review"].dropna().tolist()
        if not looks_like_ingredient(txt)
    ]
    random.shuffle(neg_samples)
    neg_samples = neg_samples[:len(pos_samples)]  # balance dataset

    print("Negative samples used:", len(neg_samples))

    # --------------------------
    # BUILD BALANCED DATASET
    # --------------------------
    df_out = pd.DataFrame({
        "text": pos_samples + neg_samples,
        "label": [1]*len(pos_samples) + [0]*len(neg_samples)
    }).sample(frac=1).reset_index(drop=True)

    df_out.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset → {OUT_PATH}")
    print(df_out.head())


if __name__ == "__main__":
    build_dataset()

