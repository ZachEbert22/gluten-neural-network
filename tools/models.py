# statistics_model.py
import os
import json

MODEL_DIR = "../models/ingredient_classifier"
VOCAB_FILE = f"{MODEL_DIR}/vocab.txt"
TOKENIZER = f"{MODEL_DIR}/tokenizer.json"
MODEL_WEIGHTS = f"{MODEL_DIR}/model.safetensors"

def count_vocab():
    try:
        with open(VOCAB_FILE, "r") as fp:
            return sum(1 for _ in fp)
    except:
        return 0

def main():
    print("\n===== MODEL STATISTICS =====\n")

    vocab_size = count_vocab()
    tokenizer_size = os.path.getsize(TOKENIZER) if os.path.exists(TOKENIZER) else 0
    model_size = os.path.getsize(MODEL_WEIGHTS) if os.path.exists(MODEL_WEIGHTS) else 0

    print("Vocabulary size:", vocab_size)
    print("Tokenizer file size:", f"{tokenizer_size/1024/1024:.2f} MB")
    print("Model size:", f"{model_size/1024/1024:.2f} MB")

    checkpoints = [
        d for d in os.listdir(MODEL_DIR)
        if d.startswith("checkpoint") and os.path.isdir(f"{MODEL_DIR}/{d}")
    ]

    print("\nAvailable checkpoints:")
    for c in checkpoints:
        print(" -", c)

    stats = {
        "vocab_size": vocab_size,
        "tokenizer_mb": tokenizer_size / 1024 / 1024,
        "model_mb": model_size / 1024 / 1024,
        "checkpoints": checkpoints,
    }

    with open("stats_model.json", "w") as fp:
        json.dump(stats, fp, indent=4)

    print("\nSaved → stats_model.json")

if __name__ == "__main__":
    main()

