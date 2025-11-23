import pandas as pd
import numpy as np

print("\n===== DATASET STATISTICS =====\n")

def describe_dataset(path):
    print(f"\n--- {path} ---")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return

    print(f"Samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Label statistics if present
    if "label" in df.columns:
        print("\nLabel Distribution:")
        print(df["label"].value_counts())

        print("\nLabel Percentages:")
        print((df["label"].value_counts(normalize=True) * 100).round(2))

    # Text statistics
    if "text" in df.columns:
        df["char_len"] = df["text"].astype(str).apply(len)
        df["token_len"] = df["text"].astype(str).apply(lambda x: len(x.split()))

        print(f"\nAverage Characters Per Sample: {df['char_len'].mean():.2f}")
        print(f"Average Tokens Per Sample: {df['token_len'].mean():.2f}")
        print(f"Max Tokens: {df['token_len'].max()}")

    print("\n--- END ---\n")


describe_dataset("../data/ingredient_classifier_dataset.csv")
describe_dataset("../data/ingredient_dataset.csv")

