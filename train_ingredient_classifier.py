#!/usr/bin/env python3
"""
Train a DistilBERT binary classifier to detect ingredient lines.

Key fixes included:
- Sets TOKENIZERS_PARALLELISM=false to avoid fork warnings.
- Ensures dataset contains only text,label (casts dtypes).
- Uses DataCollatorWithPadding for dynamic padding (no variable-length tensor crash).
- Removes stray columns after tokenization.
- Respects environment vars MAX_POS_SAMPLES/MAX_NEG_SAMPLES if dataset was built with them.
- Uses fp16 when available and GPU if present.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import argparse
import random

DATASET_PATH = "data/ingredient_dataset.csv"
MODEL_OUT = "models/ingredient_classifier"
os.makedirs(MODEL_OUT, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"


def load_dataset(max_pos: int | None = None, max_neg: int | None = None):
    # Load CSV and keep only text,label (robust cleanup)
    df = pd.read_csv(DATASET_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"Dataset missing required columns. Found: {df.columns.tolist()}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # optional trimming if env var set
    env_pos = os.environ.get("MAX_POS_SAMPLES")
    env_neg = os.environ.get("MAX_NEG_SAMPLES")
    if env_pos and env_pos.isdigit():
        max_pos = int(env_pos)
    if env_neg and env_neg.isdigit():
        max_neg = int(env_neg)

    # If user requested trimming by function args, do that before shuffling
    if max_pos is not None or max_neg is not None:
        pos = df[df.label == 1]
        neg = df[df.label == 0]
        if max_pos is not None:
            pos = pos.sample(n=min(max_pos, len(pos)), random_state=42)
        if max_neg is not None:
            neg = neg.sample(n=min(max_neg, len(neg)), random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Loaded dataset rows:", len(df))
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
    return Dataset.from_pandas(train_df.reset_index(drop=True)), Dataset.from_pandas(val_df.reset_index(drop=True))


def tokenize_fn(batch, tokenizer, max_length=128):
    # dynamic padding; we will use DataCollatorWithPadding when training
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-pos", type=int, default=None)
    parser.add_argument("--max-neg", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds, val_ds = load_dataset(max_pos=args.max_pos, max_neg=args.max_neg)

    print("Instantiating tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # memory optimizations
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model.to(device)

    print("Tokenizing train/val (batched)...")
    train_ds = train_ds.map(lambda b: tokenize_fn(b, tokenizer, max_length=args.max_length), batched=True, remove_columns=["text"])
    val_ds = val_ds.map(lambda b: tokenize_fn(b, tokenizer, max_length=args.max_length), batched=True, remove_columns=["text"])

    # rename label -> labels
    if "label" in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in val_ds.column_names:
        val_ds = val_ds.rename_column("label", "labels")

    # keep only model columns
    keep = {"input_ids", "attention_mask", "labels"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    # Data collator: dynamic padding per batch
    data_collator = DataCollatorWithPadding(tokenizer)

    per_device = args.batch_size
    training_args = TrainingArguments(
        output_dir=MODEL_OUT,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # FIXED: evaluation_strategy -> eval_strategy
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,

        logging_steps=200,
        num_train_epochs=2,

        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Saving model & tokenizer...")
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print("Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    main()

