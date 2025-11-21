import streamlit as st
import torch
import pickle
import requests
import re
from bs4 import BeautifulSoup

from models.gluten_model import GlutenSubstitutionNet
from models.ingredient_classifier.predict import (
    load_transformer_classifier,
    is_ingredient_line_transformer
)
from models.bert_embedder.embedder import BertEmbedder

from utils.parser import parse_ingredient, format_ingredient
from utils.gluten_check import load_gluten_ingredients, load_substitutions
from utils.substitution import substitute_ingredient

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

ING_MODEL_PATH = "models/ingredient_classifier"

@st.cache_resource
def load_transformer_classifier():
    tokenizer = AutoTokenizer.from_pretrained(ING_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(ING_MODEL_PATH)
    model.eval()
    return tokenizer, model

tok_ing, clf_ing = load_transformer_classifier()

def is_ingredient_line_transformer(text: str) -> bool:
    encoded = tok_ing(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = clf_ing(**encoded).logits
    pred = logits.softmax(dim=-1).argmax().item()
    return pred == 1   # label 1 = ingredient


# ------------------------------------------------
# Load MLP substitution model
# ------------------------------------------------
@st.cache_resource
def load_mlp_model():
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    checkpoint = torch.load("models/model.pth", map_location="cpu")
    num_subs = checkpoint["out_substitute.weight"].shape[0]

    model = GlutenSubstitutionNet(
        input_dim=len(vectorizer.get_feature_names_out()),
        hidden_dim=128,
        num_substitutes=num_subs
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model, vectorizer


mlp_model, vectorizer = load_mlp_model()
bert_embedder = BertEmbedder()
clf_model, clf_tokenizer = load_transformer_classifier()

gluten_ingredients = load_gluten_ingredients()
substitutions = load_substitutions()


# ------------------------------------------------
# Better URL extraction using BERT ingredient classifier
# ------------------------------------------------
def extract_recipe_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        candidates = []

        selectors = [
            '[itemprop="recipeIngredient"]',
            '.ingredients-item-name',
            '.recipe-ingredients__list-item',
            'li.ingredient',
        ]

        # direct ingredient selectors
        for sel in selectors:
            for tag in soup.select(sel):
                txt = tag.get_text(strip=True)
                candidates.append(txt)

        # fallback: any LI with ingredient-like patterns
        if not candidates:
            for li in soup.find_all("li"):
                txt = li.get_text(strip=True)
                if re.search(r"(cup|tsp|tbsp|g|kg|flour|sugar|butter)", txt, re.I):
                    candidates.append(txt)

        # FILTER USING THE TRANSFORMER CLASSIFIER
        cleaned = []
        for line in candidates:
            if is_ingredient_line_transformer(line, clf_model, clf_tokenizer):
                cleaned.append(line)

        cleaned = list(dict.fromkeys(cleaned))
        return cleaned[:20]

    except Exception as e:
        st.error(f"Error extracting recipe: {e}")
        return []

# ------------------------------
# UI SECTION
# ------------------------------
st.title("🥣 AI Gluten-Free Recipe Converter")
st.markdown("Upload a recipe, or paste a URL from Food.com / AllRecipes / Epicurious to get a gluten-free version!")

mode = st.radio("Choose input type:", ["Paste Recipe Text", "Recipe URL"])

if mode == "Paste Recipe Text":
    recipe_input = st.text_area("Enter recipe ingredients (one per line):")
    if st.button("Convert to Gluten-Free"):
        ingredients = [i.strip() for i in recipe_input.split("\n") if i.strip()]
        if ingredients:
            st.subheader("Converted Ingredients:")
            for ing in ingredients:
                parsed = parse_ingredient(ing)
                sub = substitute_ingredient(parsed, substitutions)
                formatted = format_ingredient(sub)
                st.write(f"✅ {ing} → {formatted}")
        else:
            st.warning("Please enter some ingredients first.")

ingredients = extract_recipe_from_url(recipe_url)
ingredients = [ing for ing in ingredients if is_ingredient_line_transformer(ing)]

if ingredients:
    st.subheader("Extracted Ingredients:")
    for ing in ingredients:
        st.write(f"- {ing}")
    st.subheader("Converted to Gluten-Free:")
    for ing in ingredients:
        parsed = parse_ingredient(ing)
        sub = substitute_ingredient(parsed, substitutions)
        formatted = format_ingredient(sub)
        st.write(f"✅ {ing} → {formatted}")
else:
    st.warning("Could not extract ingredients from that URL.")


