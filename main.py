import re
from pathlib import Path
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch, os, pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from utils.gluten_check import load_gluten_ingredients, load_substitutions
from models.gluten_model import GlutenSubstitutionNet
from utils.train_utils import train_model, test_model
from utils.parser import process_recipe

# -------------------------
# Helpers for handling R-style lists and pairing parts+quantities
# -------------------------
def de_r_list(raw):
    """
    Convert an R-style c("a","b","c") or Python-list-like string into a Python list of strings.
    If input already a list, return as-is.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    # If it looks like an R vector: c("a","b")
    if s.startswith("c(") and s.endswith(")"):
        # remove leading c( and trailing )
        inner = s[2:-1]
        # split on commas that separate quoted tokens
        # remove quotes and whitespace
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        return [p for p in parts if p and p.lower() != "na"]
    # If looks like Python list string
    if s.startswith("[") and s.endswith("]"):
        try:
            vals = eval(s)
            if isinstance(vals, list):
                return [str(v).strip() for v in vals if v is not None]
        except Exception:
            pass
    # If newline separated
    if "\n" in s:
        parts = [p.strip().strip('"').strip("'") for p in s.splitlines()]
        return [p for p in parts if p and p.lower() != "na"]
    # fallback: split on commas
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p and p.lower() != "na"]


def is_quantity_token(tok: str):
    """Return True if token looks like a quantity (4, 1/2, 1 1/2, 0.5)."""
    if not isinstance(tok, str):
        return False
    tok = tok.strip()
    # pure numeric, fractions, mixed numbers like "1 1/2"
    return bool(re.fullmatch(r'[\d]+(?:\s+[\d/]+)?|[\d/]+(?:\s*[\d/]+)?|[\d]+\.[\d]+', tok))

def clean_text(s: str) -> str:
    """
    Clean ingredient text without breaking apart valid tokens.
    Ensures spaces between numbers, units, and words.
    """
    if not isinstance(s, str):
        return ""
    # Normalize fractions and symbols
    s = s.replace("⁄", "/").replace("½", "1/2").replace("¼", "1/4").replace("¾", "3/4")
    s = re.sub(r"[\u00A0\u200B\u2009]", " ", s)

    # ✅ Insert missing spaces between numbers and letters (e.g. 200gplain → 200 g plain)
    s = re.sub(r"(?<=\d)([a-zA-Z])", r" \1", s)
    s = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", s)
    s = re.sub(r"([a-zA-Z])([A-Z])", r"\1 \2", s)

    # Remove weird punctuation but keep fractions and dashes
    s = re.sub(r"[^a-zA-Z0-9/\-\s]", "", s)
    # Normalize spacing
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pair_parts_and_quantities(parts_raw, quants_raw):
    """
    Input: raw values from RecipeIngredientParts and RecipeIngredientQuantities columns
    Output: list of cleaned ingredient strings like "1 cup flour" or "flour"
    """
    parts = de_r_list(parts_raw)
    quants = de_r_list(quants_raw)

    # Handle list-like strings
    if len(parts) == 1 and "," in parts[0]:
        parts = [p.strip() for p in parts[0].split(",") if p.strip()]
    if len(quants) == 1 and "," in quants[0]:
        quants = [p.strip() for p in quants[0].split(",") if p.strip()]

    paired = []
    maxlen = max(len(parts), len(quants))
    for i in range(maxlen):
        part = parts[i] if i < len(parts) else ""
        quant = quants[i] if i < len(quants) else ""

        part = str(part).strip().strip('"').strip("'")
        quant = str(quant).strip().strip('"').strip("'")

        if part and is_quantity_token(part):
            if i < len(quants) and not is_quantity_token(quants[i]):
                part, quant = quants[i], part

        if (not part or part.lower() in ["na", "nan", ""]) and quant and not is_quantity_token(quant):
            part, quant = quant, ""

        if is_quantity_token(part) and not quant:
            continue

        if quant and quant.lower() not in ["na", "nan", ""]:
            combined = f"{quant} {part}".strip()
        else:
            combined = part.strip()

        combined = re.sub(r'(^c\(|\)$)', '', combined).strip(' ,;')
        if combined:
            paired.append(combined)

    # ✅ Apply cleaning to final strings
    paired = [clean_text(x) for x in paired if x]
    return paired


# -------------------------
# Reuse earlier dataset helpers (find csv)
# -------------------------
def find_csv_in_path(path):
    p = Path(path)
    csvs = list(p.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    # prefer recipes.csv or raw_recipes
    for candidate in csvs:
        name = candidate.name.lower()
        if "recipes" in name or "recipe" in name:
            return candidate
    return csvs[0]


# -------------------------
# Dataset wrapper
# -------------------------
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, X, flags, subs):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.flags = torch.tensor(flags, dtype=torch.long)
        self.subs = torch.tensor(subs, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.flags[idx], self.subs[idx]


# -------------------------
# Label creation
# -------------------------
def make_labels(ingredient_lists, gluten_keys, subs_keys):
    flags = []
    sub_classes = []
    for lst in ingredient_lists:
        joined = " ".join(lst).lower()
        flag = int(any(g.lower() in joined for g in gluten_keys))
        flags.append(flag)
        idx = 0
        for i, key in enumerate(subs_keys, start=1):
            if key.lower() in joined:
                idx = i
                break
        sub_classes.append(idx)
    return flags, sub_classes


# -------------------------
# Kaggle dataset loader
# -------------------------
def load_kaggle_dataset(dataset_handle: str):
    print("Downloading dataset via kagglehub:", dataset_handle)
    path = kagglehub.dataset_download(dataset_handle)
    print("Downloaded to:", path)
    csv_path = find_csv_in_path(path)
    print("Using CSV:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    return df


# -------------------------
# Prepare features & paired ingredient_list
# -------------------------
def prepare_features_labels(df, gluten_ingredients, substitutions, sample_limit=None):
    # First try to find explicit parts+quantities columns
    col_parts = None
    col_quants = None
    for c in df.columns:
        low = c.lower()
        if "ingredientpart" in low or "ingredient_part" in low or "recipeingredientparts" in low:
            col_parts = c
        if "ingredientquant" in low or "ingredient_quantity" in low or "recipeingredientquantities" in low:
            col_quants = c

    # fallback: find any 'ingredient' column
    if col_parts is None:
        possible = [c for c in df.columns if "ingredient" in c.lower()]
        if not possible:
            raise ValueError("No ingredient-like columns found in dataset.")
        col_parts = possible[0]

    print("Using parts column:", col_parts, "quantities column:", col_quants)

    # build ingredient_list by pairing parts & quantities when possible
    def build_pair(row):
        parts_raw = row.get(col_parts, None)
        quants_raw = row.get(col_quants, None) if col_quants else None
        paired = pair_parts_and_quantities(parts_raw, quants_raw)
        # if pair result empty, fallback to parse from single column string
        if not paired:
            # try to parse the single cell as a comma-separated list
            single = row.get(col_parts, "")
            if pd.isna(single):
                return []
            if isinstance(single, list):
                return [str(x).strip() for x in single if x]
            s = str(single)
            # try eval
            if s.strip().startswith("[") and s.strip().endswith("]"):
                try:
                    vals = eval(s)
                    return [str(v).strip() for v in vals if v]
                except Exception:
                    pass
            # split on commas/newlines
            if "\n" in s:
                return [p.strip().strip('"').strip("'") for p in s.splitlines() if p.strip()]
            return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]

        return paired

    df = df.copy()
    df["ingredient_list"] = df.apply(build_pair, axis=1)
    if sample_limit:
        df = df.head(sample_limit)

    # Vectorize full-ingredient text for model
    texts = [" ".join(lst) for lst in df["ingredient_list"]]
    vectorizer = CountVectorizer(max_features=1024, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    subs_keys = list(substitutions.keys())
    flags, sub_classes = make_labels(df["ingredient_list"].tolist(), gluten_ingredients, subs_keys)

    return X, flags, sub_classes, df, vectorizer


# -------------------------
# Main
# -------------------------
def main():
    gluten_ingredients = load_gluten_ingredients()
    substitutions = load_substitutions()
    subs_keys = list(substitutions.keys())

    # try dataset handles
    df = None
    for handle in [
        "irkaal/foodcom-recipes-and-reviews",
        "irkaal/foodcom-recipes-with-search-terms-and-tags",
        "irkaal/foodcom-recipes",
    ]:
        try:
            df = load_kaggle_dataset(handle)
            break
        except Exception as e:
            print(f"Failed to load {handle}: {e}")
    if df is None:
        raise RuntimeError("Could not load any Kaggle dataset successfully.")

    X, flags, subs, df_proc, vectorizer = prepare_features_labels(df, gluten_ingredients, substitutions, sample_limit=2000)

    # train/test split
    X_train, X_test, y_flag_train, y_flag_test, y_sub_train, y_sub_test = train_test_split(
        X, flags, subs, test_size=0.2, random_state=42
    )

    train_ds = RecipeDataset(X_train, y_flag_train, y_sub_train)
    test_ds = RecipeDataset(X_test, y_flag_test, y_sub_test)
    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=64)

    # build model
    input_dim = X.shape[1]
    num_substitutes = len(subs_keys) + 1
    model = GlutenSubstitutionNet(input_dim=input_dim, hidden_dim=128, num_substitutes=num_substitutes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_flag = nn.CrossEntropyLoss()
    criterion_sub = nn.CrossEntropyLoss()

    model = train_model(model, trainloader, optimizer, criterion_flag, criterion_sub, epochs=6)
    test_model(model, testloader)

    # Run rule-based substitution sample output (first 5)
    print("\nRunning rule-based substitution sample output (first 5 recipes):")
    for lst in df_proc["ingredient_list"].head(5).tolist():
        # lst should now be a list of strings like "1 cup flour" or "flour"
        if not lst:
            continue
        process_recipe(lst, gluten_ingredients, substitutions)

    import os, pickle, torch

    # === SAVE TRAINED MODEL + VECTORIZER ===
    SAVE_DIR = "models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = os.path.join(SAVE_DIR, "model.pth")
    vec_path = os.path.join(SAVE_DIR, "vectorizer.pkl")

    # Save the model weights
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved trained model → {model_path}")

    # Save the fitted vectorizer
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Saved fitted vectorizer → {vec_path}")


if __name__ == "__main__":
    main()

