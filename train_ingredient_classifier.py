import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

DATA_PATH = "data/ingredient_classifier_dataset.csv"
MODEL_DIR = "models/ingredient_classifier"
PRETRAIN = "distilbert-base-uncased"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    assert "text" in df.columns and "label" in df.columns

    # Shuffle and split
    ds = Dataset.from_pandas(df.sample(frac=1, random_state=42).reset_index(drop=True))
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAIN, num_labels=2
    )

    # FIX FOR Older Transformers — fallback argument names
    try:
        args = TrainingArguments(
            output_dir=MODEL_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
        )
    except TypeError:
        # Old versions use "eval_strategy"
        args = TrainingArguments(
            output_dir=MODEL_DIR,
            eval_strategy="epoch",   # fallback
            save_strategy="epoch",
            num_train_epochs=3,
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = (preds == labels).astype(float).mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    tokenizer.save_pretrained(MODEL_DIR)
    trainer.save_model(MODEL_DIR)

    print("Saved model →", MODEL_DIR)


if __name__ == "__main__":
    main()

