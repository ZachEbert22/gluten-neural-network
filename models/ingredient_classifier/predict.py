import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("models/ingredient_classifier")


def load_transformer_classifier():
    """
    Loads the DistilBERT ingredient classifier trained earlier.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def is_ingredient_line_transformer(text: str, model=None, tokenizer=None, threshold: float = 0.5) -> bool:
    """
    Uses the BERT classifier to determine whether a text line is an ingredient line.
    Returns True if probability(ingredient) >= threshold.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_transformer_classifier()

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0][1].item()   # label 1 = ingredient

    return probs >= threshold

