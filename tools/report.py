import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import json

OUTPUT_DIR = "report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- NEW: Robust CSV Loader ----
DATASET_CANDIDATES = [
    "ingredient_classifier_dataset.csv",
    "../data/ingredient_classifier_dataset.csv",
    "data/ingredient_classifier_dataset.csv"
]

df = None
for path in DATASET_CANDIDATES:
    if os.path.exists(path):
        print(f"Loading dataset from: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    raise FileNotFoundError("Could not find ingredient_classifier_dataset.csv in any known location.")


# ------------------------------------------------------
# 1. DATASET STATISTICS + GRAPHS
# ------------------------------------------------------
label_counts = df["label"].value_counts()
label_percent = (label_counts / len(df)) * 100

print("\n===== LABEL DISTRIBUTION =====\n")
print(label_counts)
print("\nPercentages:")
print(label_percent.round(2))

# Bar chart
plt.figure(figsize=(7,5))
label_counts.plot(kind="bar")
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
plt.close()

# Token length histogram
token_lengths = df["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(7,5))
plt.hist(token_lengths, bins=30)
plt.title("Token Length Distribution")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/token_length_histogram.png")
plt.close()

# ------------------------------------------------------
# 2. MODEL STATISTICS
# ------------------------------------------------------
TOKENIZER_CANDIDATES = [
    "models/ingredient_classifier",
    "models/ingredient_classifier/"
]

tokenizer_path = None
for p in TOKENIZER_CANDIDATES:
    if os.path.exists(p):
        tokenizer_path = p
        break

if tokenizer_path is None:
    raise FileNotFoundError("Tokenizer folder not found.")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
vocab_size = len(tokenizer.get_vocab())

print("\n===== MODEL STATISTICS =====\n")
print(f"Vocabulary size: {vocab_size}")

MODEL_DIR = "models/ingredient_classifier/"
model_size_bytes = sum(
    os.path.getsize(os.path.join(MODEL_DIR, f)) 
    for f in os.listdir(MODEL_DIR)
    if os.path.isfile(os.path.join(MODEL_DIR, f))
)
model_size_mb = model_size_bytes / (1024**2)

print(f"Model size: {model_size_mb:.2f} MB")

with open("stats_model.json", "w") as f:
    json.dump({
        "vocab_size": vocab_size,
        "model_size_mb": model_size_mb
    }, f, indent=4)

plt.figure(figsize=(7,5))
plt.bar(["Vocabulary Size", "Model Size (MB)"], [vocab_size, model_size_mb])
plt.title("Model Summary")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_summary.png")
plt.close()

print("\nSaved all charts to → report_figures/")

