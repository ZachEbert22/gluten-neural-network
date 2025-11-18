import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "data/ingredient_classifier_dataset.csv"
MODEL_DIR = "models/ingredient_classifier"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# Load tokenizer/model
# ---------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ---------------------------------------------------------
# Tokenization function
# ---------------------------------------------------------
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_enc = train_df.apply(lambda x: tokenize({"text": x["text"]}), axis=1)
eval_enc  = eval_df.apply(lambda x: tokenize({"text": x["text"]}), axis=1)

def build_dataset(encodings, labels):
    dataset = []
    for enc, label in zip(encodings, labels):
        dataset.append({
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(label)
        })
    return dataset

train_dataset = build_dataset(train_enc, train_df["label"].tolist())
eval_dataset  = build_dataset(eval_enc,  eval_df["label"].tolist())

# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=20,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
trainer.train()

# ---------------------------------------------------------
# Save model & tokenizer
# ---------------------------------------------------------
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("\n=== DONE ===")
print("Saved Transformer ingredient classifier:")
print(f"> {MODEL_DIR}")

