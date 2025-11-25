import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import re

LOG_PATH = Path("data/prediction_log.jsonl")

# -----------------------------
# Load gluten list exactly the way backend does
# -----------------------------
from utils.gluten_check import load_gluten_ingredients
gluten_list = load_gluten_ingredients()


def contains_gluten(name: str) -> bool:
    """Same rule used by backend Unified API."""
    name = (name or "").lower()
    for g in gluten_list:
        if g in name:
            return True
    return False


# -----------------------------
# Extract true/pred labels from JSONL logs
# -----------------------------
def load_labels_from_log():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"{LOG_PATH} does not exist.")

    y_true = []
    y_pred = []
    all_ing = []

    with open(LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)

            # entry["substitutions"] is a list of ingredient outputs
            for sub in entry.get("substitutions", []):
                original = sub.get("original") or ""
                parsed_ing = original.lower()

                # ---------------- TRUE LABEL ----------------
                true_label = 1 if contains_gluten(parsed_ing) else 0

                # ---------------- PRED LABEL ----------------
                status = sub.get("status", "")
                pred_label = 1 if status == "substituted" else 0

                y_true.append(true_label)
                y_pred.append(pred_label)
                all_ing.append(original)

    return np.array(y_true), np.array(y_pred), all_ing


# -----------------------------
# Build and plot the 2×2 matrix
# -----------------------------
def build_confusion_matrix():
    y_true, y_pred, ingredients = load_labels_from_log()

    # Only 2 classes: {0, 1}
    labels = [0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ---- Plot ----
    plt.figure(figsize=(5, 5))
    plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0\n(gluten-free)", "Pred 1\n(gluten)"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Normalized Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()

    print("\nRAW CONFUSION MATRIX")
    print(cm)

    return {
        "confusion_matrix": cm_norm.tolist(),
        "raw_confusion_matrix": cm.tolist(),
        "ingredients": ingredients,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }


if __name__ == "__main__":
    result = build_confusion_matrix()
    print("\nDone.")

