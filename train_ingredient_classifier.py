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
import transformers, sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Transformers version:", transformers.__version__)
print("Python:", sys.executable)

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
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="longest",
        max_length=True,        # MUCH FASTER
    )


# -------------------------------------------------------
# Main training routine
# -------------------------------------------------------
def main():
    global tokenizer

    print("Loading dataset…")
    train_ds, val_ds = load_dataset()

    print("Loading tokenizer/model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # BIG speedup, lower memory
    model.gradient_checkpointing_enable()

    print("Tokenizing…")
    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=4)
    val_ds = val_ds.map(tokenize_fn, batched=True, num_proc=4)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    # Critical: speed + stability
    train_ds = train_ds.with_format("torch")
    val_ds = val_ds.with_format("torch")

    # TRAINING SETTINGS — optimized for speed + safety
    training_args = TrainingArguments(
        output_dir=MODEL_OUT,

        per_device_train_batch_size=16,       # up from 4
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,       # effective batch size = 16

        fp16=True,                           # huge speedup
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,

        logging_steps=200,
        num_train_epochs=2,

        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,

        torch_compile=True,                  # PyTorch 2.0 speed boost
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

    print(f"Model saved to → {MODEL_OUT}")


if __name__ == "__main__":
    main()

