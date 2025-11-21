import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch


DATASET_PATH = "data/ingredient_dataset.csv"
MODEL_OUT = "models/ingredient_classifier"
os.makedirs(MODEL_OUT, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"


# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # enforce correct dtypes
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # train/val split
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)

    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


# -------------------------------------------------------
# TOKENIZATION FUNCTION
# -------------------------------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,     # dynamic padding (safer)
        max_length=128
    )


# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
def main():
    global tokenizer

    print("Loading dataset…")
    train_ds, val_ds = load_dataset()

    print("Loading tokenizer/model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # GPU memory optimization
    model.gradient_checkpointing_enable()

    print("Tokenizing…")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # rename label → labels
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    # remove non-model columns
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    # convert to torch tensors
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # dynamic padding collator
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_OUT,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="epoch",
        save_strategy="epoch",

        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=200,

        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("Training…")
    trainer.train()

    print("Saving model…")
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    print(f"Model saved to → {MODEL_OUT}")


if __name__ == "__main__":
    main()

