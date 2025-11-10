import os
from pathlib import Path
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import re

from utils.gluten_check import load_gluten_ingredients, load_substitutions
from models.gluten_model import GlutenSubstitutionNet
from utils.train_utils import train_model, test_model
from utils.parser import process_recipe


# -------------------------
# Dataset helper
# -------------------------
def find_csv_in_path(path):
    """Find a CSV file that looks like the recipe data."""
    p = Path(path)
    csvs = list(p.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    for candidate in csvs:
        name = candidate.name.lower()
        if "raw" in name and "recipe" in name:
            return candidate
    for candidate in csvs:
        if "recipe" in candidate.name.lower() or "recipes" in candidate.name.lower():
            return candidate
    return csvs[0]


# -------------------------
# Dataset class for PyTorch
# -------------------------
class RecipeDataset(Dataset):
    def __init__(self, X, flags, subs):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.flags = torch.tensor(flags, dtype=torch.long)
        self.subs = torch.tensor(subs, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.flags[idx], self.subs[idx]


# -------------------------
# Label helper
# -------------------------
def make_labels(ingredient_lists, gluten_keys, subs_keys):
    flags, sub_classes = [], []
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
    """Download dataset and find a CSV inside."""
    print("Downloading dataset via kagglehub:", dataset_handle)
    path = kagglehub.dataset_download(dataset_handle)
    print("Downloaded to:", path)
    csv_path = find_csv_in_path(path)
    print("Using CSV:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def clean_ingredient_list(raw_entry):
    """
    Clean R-style c("...") ingredient strings from the Food.com dataset.
    Returns a list of cleaned ingredient strings.
    """
    if raw_entry is None:
        return []

    # Ensure it's a string
    if not isinstance(raw_entry, str):
        try:
            raw_entry = str(raw_entry)
        except Exception:
            return []

    s = raw_entry.strip()
    # Remove leading/trailing R list syntax
    s = re.sub(r'^c\(|\)$', '', s)
    # Remove all double quotes
    s = s.replace('"', '')
    # Split on commas
    parts = [p.strip() for p in s.split(',') if p.strip()]

    cleaned = []
    for part in parts:
        # Drop NA, empty, or purely numeric tokens like "4", "1/2"
        if part.lower() == "na":
            continue
        if re.fullmatch(r'[\d\s\/\.]+', part):
            continue
        cleaned.append(part)
    return cleaned

# -------------------------
# Feature & label builder
# -------------------------
def prepare_features_labels(df, gluten_ingredients, substitutions, sample_limit=None):
    """Extract ingredient lists and build bag-of-words + labels."""

    possible_cols = [c for c in df.columns if "ingredient" in c.lower()]
    if not possible_cols:
        raise ValueError(f"No ingredient-like column found in dataset. Columns: {list(df.columns)}")
    col = possible_cols[0]
    print("Using ingredient column:", col)

    def to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        s = str(x)
        if s.strip().startswith("[") and s.strip().endswith("]"):
            try:
                return eval(s)
            except Exception:
                pass
        if "\n" in s:
            return [line.strip() for line in s.splitlines() if line.strip()]
        return [p.strip() for p in s.split(",") if p.strip()]

    df = df.dropna(subset=[col]).copy()
    df["ingredient_list"] = df[col].apply(to_list)
    if sample_limit:
        df = df.head(sample_limit)

    texts = [" ".join(lst) for lst in df["ingredient_list"]]
    vectorizer = CountVectorizer(max_features=1024, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    subs_keys = list(substitutions.keys())
    flags, sub_classes = make_labels(df["ingredient_list"].tolist(), gluten_ingredients, subs_keys)

    return X, flags, sub_classes, df, vectorizer


# -------------------------
# MAIN
# -------------------------
def main():
    gluten_ingredients = load_gluten_ingredients()
    substitutions = load_substitutions()
    subs_keys = list(substitutions.keys())

    # Try two known Kaggle datasets
    df = None
    for handle in [
        "irkaal/foodcom-recipes-and-reviews",
        "irkaal/foodcom-recipes-with-search-terms-and-tags",
    ]:
        try:
            df = load_kaggle_dataset(handle)
            break
        except Exception as e:
            print(f"Failed to load {handle}: {e}")
    if df is None:
        raise RuntimeError("Could not load any Kaggle dataset successfully.")

    # Prepare data
    try:
        X, flags, subs, df_proc, vectorizer = prepare_features_labels(
            df, gluten_ingredients, substitutions, sample_limit=2000
        )
    except Exception as e:
        print("⚠️ Could not prepare features properly:", e)
        return

    # Split data
    X_train, X_test, y_flag_train, y_flag_test, y_sub_train, y_sub_test = train_test_split(
        X, flags, subs, test_size=0.2, random_state=42
    )

    train_ds = RecipeDataset(X_train, y_flag_train, y_sub_train)
    test_ds = RecipeDataset(X_test, y_flag_test, y_sub_test)
    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=64)

    # Model setup
    input_dim = X.shape[1]
    num_substitutes = len(subs_keys) + 1
    model = GlutenSubstitutionNet(input_dim, hidden_dim=128, num_substitutes=num_substitutes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_flag = nn.CrossEntropyLoss()
    criterion_sub = nn.CrossEntropyLoss()

    # Train and test
    model = train_model(model, trainloader, optimizer, criterion_flag, criterion_sub, epochs=6)
    test_model(model, testloader)

    # Print substitutions using processed dataframe
    # Run sample substitutions
    print("\nRunning rule-based substitution sample output (first 5 recipes):\n")
    sample_lists_raw = df_proc['ingredient_list'].head(5).tolist()
    sample_lists = [clean_ingredient_list(lst) for lst in sample_lists_raw]

    for lst in sample_lists:
        if lst:  # skip empty
            process_recipe(lst, gluten_ingredients, substitutions)


if __name__ == "__main__":
    main()

