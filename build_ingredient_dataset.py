import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import re
import random

# ------------------------------
# LOAD KAGGLE DATASET DIRECTLY
# ------------------------------
def load_kaggle_dataset(handle: str, file_name="recipes.csv"):
    api = KaggleApi()
    api.authenticate()

    out_dir = Path("data/kaggle_cache") / handle.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (out_dir / file_name).exists():
        api.dataset_download_file(handle, file_name, path=str(out_dir))
        zip_file = out_dir / f"{file_name}.zip"

        # Extract
        import zipfile
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(out_dir)
        zip_file.unlink()

    return pd.read_csv(out_dir / file_name)


# ------------------------------
# CLEAN INGREDIENT STRING
# ------------------------------
def clean_ing_text(text: str):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s/.-]", " ", text)  # remove symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# BUILD DATASET
# ------------------------------
def build_dataset(output_csv="data/bert_ingredient_dataset.csv", sample_size=15000):
    df = load_kaggle_dataset("irkaal/foodcom-recipes")   # <--- THIS IS WHAT YOU ASKED FOR

    print(f"Loaded {len(df)} recipe rows from Kaggle.")

    all_lines = []

    # Food.com dataset stores ingredients in R-style:
    # c("1 cup", "all-purpose flour")
    for idx, row in df.iterrows():
        parts_raw = row.get("RecipeIngredientParts", "[]")
        quants_raw = row.get("RecipeIngredientQuantities", "[]")

        try:
            parts = eval(parts_raw) if isinstance(parts_raw, str) else []
            quants = eval(quants_raw) if isinstance(quants_raw, str) else []
        except:
            continue

        for p, q in zip(parts, quants):
            p = clean_ing_text(str(p))
            q = clean_ing_text(str(q))
            line = f"{q} {p}".strip()
            if len(line.split()) >= 2:
                all_lines.append(line)

    # Deduplicate
    all_lines = list(sorted(set(all_lines)))

    # Keep only sample_size
    if len(all_lines) > sample_size:
        all_lines = random.sample(all_lines, sample_size)

    # Build training set:
    df_out = pd.DataFrame({
        "text": all_lines,
        "label": [1] * len(all_lines)   # Ingredient = 1
    })

    # Add negative examples from instructions/descriptions
    neg = df["Instructions"].dropna().sample(len(df_out)//2, replace=True)
    neg_clean = neg.apply(clean_ing_text)
    neg_clean = neg_clean[neg_clean.apply(lambda x: len(x.split()) >= 4)]

    df_neg = pd.DataFrame({
        "text": neg_clean,
        "label": [0] * len(neg_clean)
    })

    final = pd.concat([df_out, df_neg], ignore_index=True)
    final.to_csv(output_csv, index=False)

    print(f"Saved ingredient dataset: {output_csv} ({len(final)} lines)")


if __name__ == "__main__":
    build_dataset()

