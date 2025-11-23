#!/usr/bin/env python3
"""
tools/statistics_core.py

Generates graphs (1-6) that do NOT require the backend server:
1) Ingredient Coverage Graph
2) Detection Accuracy by Ingredient Category  (needs labels/preds if present)
3) Confusion Matrix Heatmap for Ingredient Normalization (if preds exist)
4) Frequency Graph of Training Dataset Ingredients
5) Tokenizer Length Distribution
6) Loss vs Dataset Size Graph (simulated if no real logs)

Outputs:
 - PNGs → reports/plots/
 - JSON summaries → reports/stats/
Usage:
    python tools/statistics_core.py
"""

import os
import json
import math
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import tokenizer from your model dir
from transformers import AutoTokenizer

# Local utilities from your repo
try:
    from utils.ingredient_parser import parse_ingredient_line
    from utils.normalization import normalize_ingredient
except Exception:
    # fallback to a tiny parser inside this file if imports fail
    def parse_ingredient_line(line):
        # minimal fallback: try to extract a leading number and unit
        import re
        qty = None
        unit = None
        ingredient = line
        m = re.search(r"(?P<qty>\d+(\.\d+)?|\.\d+|\d+\s*\d+/\d+)", line)
        if m:
            qty = m.group(0)
            ingredient = line.replace(qty, "").strip()
        return {"quantity": qty, "unit": None, "ingredient": ingredient}
    def normalize_ingredient(x): return (x or "").lower()

# Paths (tweak if you moved files)
ROOT = Path(".")
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / ".." / "data"
MODEL_DIR = ROOT / "models" / "ingredient_classifier"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
STATS_DIR = REPORTS_DIR / "stats"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Helpful regexes / lookups
DEFAULT_ING_TOKENS = ["cup", "tsp", "tbsp", "flour", "sugar", "salt", "egg", "butter", "banana", "chocolate", "bread", "crumb"]

# -------------------------
# Helpers
# -------------------------
def load_csv_samples(path: Path, nrows=None):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=nrows)
        return df
    except Exception as e:
        print("Could not read CSV", path, e)
        return None

def guess_ingredient_column(df: pd.DataFrame):
    # Try common column names
    for c in ["ingredient", "text", "line", "raw", "ingredient_text"]:
        if c in df.columns:
            return c
    # fallback: first text-like column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None

def extract_ingredients_from_series(series):
    # returns list of strings (raw lines)
    lines = []
    for v in series.dropna().astype(str).tolist():
        # if cell contains JSON-like list, try to parse
        v2 = v.strip()
        if v2.startswith("[") and v2.endswith("]"):
            try:
                arr = json.loads(v2)
                if isinstance(arr, list):
                    lines.extend([str(x) for x in arr])
                    continue
            except Exception:
                pass
        # split multi-line into lines
        if "\n" in v2:
            lines.extend([l.strip() for l in v2.splitlines() if l.strip()])
        else:
            lines.append(v2)
    return lines

# -------------------------
# 1) Ingredient Coverage Graph
# -------------------------
def ingredient_coverage_graph(lines):
    """
    Compute:
     - percent recipes (samples) with >=1 detected ingredient
     - percent with all ingredients detected (we cannot compute 'all' w/out ground truth; so approximate by lines that contain tokens)
     - substitution-eligible ingredients (contain gluten keywords; use utils.normalization keys)
    """
    # We'll treat each input line as one sample (since we don't have multi-sample files)
    total = len(lines)
    has_any = sum(1 for l in lines if any(tok in l.lower() for tok in DEFAULT_ING_TOKENS))
    # "all ingredients detected" approximated by lines that contain a number or unit (likely ingredient)
    import re
    has_qty = sum(1 for l in lines if re.search(r"\d", l))
    # substitution-eligible: lines that mention flour, bread, wheat, breadcrumbs, breadcrumb
    subs_eligible = sum(1 for l in lines if any(x in l.lower() for x in ["flour", "bread", "wheat", "breadcrumb", "breadcrumbs"]))
    coverage = {
        "total_lines": total,
        "has_any_pct": 100.0 * has_any / total if total else 0.0,
        "has_qty_pct": 100.0 * has_qty / total if total else 0.0,
        "subs_eligible_pct": 100.0 * subs_eligible / total if total else 0.0,
    }
    # Plot
    labels = ["Has ingredient token", "Has numeric qty (proxy)", "Substitution-eligible"]
    vals = [coverage["has_any_pct"], coverage["has_qty_pct"], coverage["subs_eligible_pct"]]
    plt.figure(figsize=(7,4))
    plt.bar(labels, vals)
    plt.ylim(0, 100)
    plt.title("Ingredient coverage (approx.)")
    plt.ylabel("% of lines")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_ingredient_coverage.png", dpi=150)
    plt.close()
    with open(STATS_DIR / "01_ingredient_coverage.json","w") as f:
        json.dump(coverage, f, indent=2)
    print("Saved 01_ingredient_coverage.png and JSON")
    return coverage

# -------------------------
# 2) Detection Accuracy by Ingredient Category
# -------------------------
def detection_accuracy_by_category(df, ingredient_col, pred_col=None, label_col=None):
    """
    If ground truth labels exist (label_col) and predictions exist (pred_col),
    compute category F1 scores. Otherwise attempt a lightweight category labeling
    by keyword mapping and compute a rough self-consistency measure.
    """
    from sklearn.metrics import precision_recall_fscore_support
    # build categories mapping using normalization overrides if present
    def simple_cat(name):
        name = (name or "").lower()
        if any(x in name for x in ["flour","bread","wheat","ap flour","plain flour"]): return "flour"
        if any(x in name for x in ["sugar","caster","granulated","brown sugar"]): return "sugar"
        if any(x in name for x in ["butter","oil","margarin"]): return "fat_oil"
        if any(x in name for x in ["banana","apple","pear","fruit"]): return "fruit"
        if any(x in name for x in ["milk","buttermilk","cream","cheese","yogurt"]): return "dairy"
        if any(x in name for x in ["spice","cinnamon","ginger","nutmeg","mixed spice"]): return "spice"
        return "other"

    # prefer explicit columns if present otherwise derive
    raws = extract_ingredients_from_series(df[ingredient_col])
    cats = [simple_cat(r) for r in raws]
    # if labels exist compute F1 vs preds; otherwise show counts
    if label_col and pred_col and label_col in df.columns and pred_col in df.columns:
        y_true = df[label_col].astype(str).tolist()
        y_pred = df[pred_col].astype(str).tolist()
        p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(set(y_true+y_pred)))
        out = {"labels": list(set(y_true+y_pred)), "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}
        # plot bar chart of F1 per label
        plt.figure(figsize=(8,4))
        plt.bar(out["labels"], out["f1"])
        plt.ylim(0,1)
        plt.title("Per-label F1 (if preds & labels exist)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_detection_accuracy_by_category.png", dpi=150)
        plt.close()
        with open(STATS_DIR / "02_detection_accuracy_by_category.json","w") as f:
            json.dump(out, f, indent=2)
        print("Saved 02_detection_accuracy_by_category.png and JSON")
        return out
    else:
        # fallback: show counts by our simple categories
        c = Counter(cats)
        labels = list(c.keys())
        vals = [c[k] for k in labels]
        plt.figure(figsize=(8,4))
        plt.bar(labels, vals)
        plt.title("Detection counts by heuristic category")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_detection_by_heuristic_category.png", dpi=150)
        plt.close()
        out = {"counts": dict(c)}
        with open(STATS_DIR / "02_detection_by_heuristic_category.json","w") as f:
            json.dump(out, f, indent=2)
        print("Saved 02_detection_by_heuristic_category.png and JSON")
        return out

# -------------------------
# 3) Confusion Matrix for Normalization
# -------------------------
def confusion_matrix_normalization(df, ingredient_col, pred_col=None, label_col=None):
    """
    Attempt confusion matrix between 'normalized' ground truth and predicted normalized labels.
    If none exist, we try to normalize using normalize_ingredient and pretend predictions = same (no file).
    """
    from sklearn.metrics import confusion_matrix
    raws = extract_ingredients_from_series(df[ingredient_col])
    normalized = [normalize_ingredient(r) for r in raws]
    # need preds: fallback attempt to load a predictions file
    preds = None
    preds_path = DATA_DIR / "predictions.csv"
    if preds_path.exists():
        try:
            d = pd.read_csv(preds_path)
            if "predicted" in d.columns:
                preds = d["predicted"].astype(str).tolist()
        except Exception:
            preds = None

    if preds is None:
        # no preds available — produce a small similarity/confusion between manual tokens
        # build a co-occurrence confusion of top N normalized tokens vs itself (identity)
        cnt = Counter(normalized)
        top = [k for k,_ in cnt.most_common(12)]
        mat = np.zeros((len(top),len(top)), dtype=int)
        label_to_idx = {k:i for i,k in enumerate(top)}
        # mark diagonal counts
        for n in normalized:
            if n in label_to_idx:
                mat[label_to_idx[n], label_to_idx[n]] += 1
        plt.figure(figsize=(8,6))
        plt.imshow(mat, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(top)), top, rotation=45, ha='right')
        plt.yticks(range(len(top)), top)
        plt.title("Pseudo-Confusion (no preds available) — diagonal counts")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "03_confusion_matrix_pseudo.png", dpi=150)
        plt.close()
        with open(STATS_DIR / "03_confusion_matrix_pseudo.json","w") as f:
            json.dump({"top_labels": top, "matrix_shape": list(mat.shape)}, f, indent=2)
        print("Saved 03_confusion_matrix_pseudo.png and JSON")
        return {"status":"no_preds", "top_labels": top}
    else:
        # compute real confusion matrix
        labs = list(sorted(set(normalized + preds)))
        cm = confusion_matrix(normalized, preds, labels=labs)
        plt.figure(figsize=(9,7))
        plt.imshow(cm, cmap="Reds")
        plt.colorbar()
        plt.xticks(range(len(labs)), labs, rotation=45, ha='right')
        plt.yticks(range(len(labs)), labs)
        plt.title("Confusion matrix: normalized (true) vs preds")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "03_confusion_matrix_real.png", dpi=150)
        plt.close()
        with open(STATS_DIR / "03_confusion_matrix_real.json","w") as f:
            json.dump({"labels": labs, "matrix_shape": list(cm.shape)}, f, indent=2)
        print("Saved 03_confusion_matrix_real.png and JSON")
        return {"status":"ok", "labels": labs}

# -------------------------
# 4) Frequency Graph of Training Dataset Ingredients
# -------------------------
def dataset_frequency_graph(df, ingredient_col):
    raws = extract_ingredients_from_series(df[ingredient_col])
    normalized = [normalize_ingredient(r) for r in raws]
    c = Counter(normalized)
    top = c.most_common(20)
    labels = [l for l,_ in top]
    vals = [v for _,v in top]
    plt.figure(figsize=(10,5))
    plt.barh(labels[::-1], vals[::-1])
    plt.title("Top 20 ingredients in dataset (normalized)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_dataset_frequency.png", dpi=150)
    plt.close()
    with open(STATS_DIR / "04_dataset_frequency.json","w") as f:
        json.dump({"top_20": top}, f, indent=2)
    print("Saved 04_dataset_frequency.png and JSON")
    return top

# -------------------------
# 5) Tokenizer Length Distribution
# -------------------------
def tokenizer_length_distribution(texts, model_dir=MODEL_DIR):
    """
    Use transformers AutoTokenizer from your model dir.
    If not available, tokenizes by whitespace.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        tokenized_lens = [len(tok(t)["input_ids"]) for t in texts]
        # also compute ingredient length (split)
        word_lengths = [len(t.split()) for t in texts]
        plt.figure(figsize=(8,4))
        plt.hist(tokenized_lens, bins=40)
        plt.title("Tokenizer token length distribution (input samples)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_tokenizer_length_tokens.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8,4))
        plt.hist(word_lengths, bins=40)
        plt.title("Ingredient word-length distribution")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_tokenizer_length_words.png", dpi=150)
        plt.close()

        with open(STATS_DIR / "05_tokenizer_length_summary.json","w") as f:
            json.dump({
                "median_tokens": float(np.median(tokenized_lens)),
                "median_words": float(np.median(word_lengths))
            }, f, indent=2)
        print("Saved tokenizer length histograms")
        return {"median_tokens": np.median(tokenized_lens), "median_words": np.median(word_lengths)}
    except Exception as e:
        print("Tokenizer load failed:", e)
        # fallback whitespace lengths
        lens = [len(t.split()) for t in texts]
        plt.figure(figsize=(8,4))
        plt.hist(lens, bins=40)
        plt.title("Token length (whitespace) fallback")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_tokenizer_length_words_fallback.png", dpi=150)
        plt.close()
        with open(STATS_DIR / "05_tokenizer_length_summary.json","w") as f:
            json.dump({"median_words": float(np.median(lens))}, f, indent=2)
        print("Saved fallback tokenizer length")
        return {"median_words": np.median(lens)}

# -------------------------
# 6) Loss vs Dataset Size Graph (simulate if no logs)
# -------------------------
def loss_vs_dataset_size(simulate=True, outfile_name="06_loss_vs_dataset_size.png"):
    """
    If you have recorded losses for different dataset sizes put them into:
        data/loss_curve.csv  (columns: dataset_size, train_loss, val_loss)
    Otherwise, we simulate a decaying loss curve.
    """
    csvp = DATA_DIR / "loss_curve.csv"
    if csvp.exists():
        df = pd.read_csv(csvp)
        plt.figure(figsize=(8,4))
        plt.plot(df["dataset_size"], df["train_loss"], label="train")
        if "val_loss" in df.columns:
            plt.plot(df["dataset_size"], df["val_loss"], label="val")
        plt.xlabel("dataset size")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Recorded loss vs dataset size")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / outfile_name, dpi=150)
        plt.close()
        print("Saved recorded loss curve")
        with open(STATS_DIR / "06_loss_vs_dataset_size.json","w") as f:
            json.dump({"source":"recorded", "rows": len(df)}, f, indent=2)
        return True
    else:
        # simulate
        sizes = np.array([5e3, 1e4, 2e4, 4e4, 8e4, 1.6e5])
        # simulate decaying loss: L = a / (1 + b * log(N)) + noise
        a = 2.2; b = 0.35
        train_loss = a / (1 + b * np.log(sizes))
        val_loss = a / (1 + b * np.log(sizes)) + 0.1/np.sqrt(np.log1p(sizes))
        plt.figure(figsize=(8,4))
        plt.plot(sizes, train_loss, label="train (sim)")
        plt.plot(sizes, val_loss, label="val (sim)")
        plt.xscale("log")
        plt.xlabel("dataset size (log scale)")
        plt.ylabel("loss (simulated)")
        plt.legend()
        plt.title("Simulated loss vs dataset size")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / outfile_name, dpi=150)
        plt.close()
        with open(STATS_DIR / "06_loss_vs_dataset_size.json","w") as f:
            json.dump({"source":"simulated","sizes": sizes.tolist()}, f, indent=2)
        print("Saved simulated loss curve")
        return True

# -------------------------
# Main runner
# -------------------------
def main():
    print("===== Running statistics_core.py =====")
    # Attempt to load dataset file (prefer ingredient_classifier_dataset.csv)
    df1 = load_csv_samples(DATA_DIR / "ingredient_classifier_dataset.csv")
    df2 = load_csv_samples(DATA_DIR / "ingredient_dataset.csv")
    df = df1 if df1 is not None else df2

    if df is None:
        print("No dataset CSV found in data/. Nothing to plot.")
        return

    # Figure out text column
    ing_col = guess_ingredient_column(df)
    if ing_col is None:
        print("No text column found in CSV; aborting.")
        return

    # Expand into lines
    lines = extract_ingredients_from_series(df[ing_col])
    print(f"Loaded {len(lines)} ingredient lines from column '{ing_col}'")

    # 1
    cov = ingredient_coverage_graph(lines)

    # 2
    dcat = detection_accuracy_by_category(df, ing_col, pred_col=None, label_col=None)

    # 3
    conf = confusion_matrix_normalization(df, ing_col)

    # 4
    freq = dataset_frequency_graph(df, ing_col)

    # 5
    tok = tokenizer_length_distribution(lines)

    # 6
    loss_vs_dataset_size(simulate=True)

    print("All core stats generated. Check reports/plots/ and reports/stats/")

if __name__ == "__main__":
    main()

