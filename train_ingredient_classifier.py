import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from datasets import Dataset

DATASET_PATH = "data/ingredient_dataset.csv"
MODEL_OUT = "models/ingredient_classifier"

os.makedirs(MODEL_OUT, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# -------------------------------------------------------
# Main training routine
# -------------------------------------------------------
def main():
    global tokenizer

    print("Loading dataset…")
    train_ds, val_ds = load_dataset()

    print("Loading tokenizer/model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    print("Tokenizing…")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir=MODEL_OUT,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Training…")
    trainer.train()

    print("Saving model…")
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    print(f"Model saved to: {MODEL_OUT}")


# -------------------------------------------------------
if __name__ == "__main__":
    main()

