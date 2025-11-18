import streamlit as st
import torch
import pickle
import requests
import re
from bs4 import BeautifulSoup

from models.gluten_model import GlutenSubstitutionNet
from utils.parser import parse_ingredient, format_ingredient, clean_text
from utils.gluten_check import load_gluten_ingredients, load_substitutions
from utils.substitution import substitute_ingredient

# transformer imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# LOAD MAIN MODEL + VECTORIZER
# ============================================================
@st.cache_resource
def load_model_and_vectorizer(
    model_path="models/model.pth",
    vec_path="models/vectorizer.pkl",
):
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    checkpoint = torch.load(model_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    out_key = next((k for k in state.keys() if "out_substitute.weight" in k), None)
    num_subs = state[out_key].shape[0] if out_key else 5

    model = GlutenSubstitutionNet(
        input_dim=len(vectorizer.get_feature_names_out()),
        hidden_dim=128,
        num_substitutes=num_subs,
    )

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, vectorizer


# ============================================================
# LOAD TRANSFORMER INGREDIENT CLASSIFIER
# ============================================================
@st.cache_resource
def load_transformer_classifier(model_dir="models/ingredient_classifier"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        clf = AutoModelForSequenceClassification.from_pretrained(model_dir)
        clf.eval()
        return tokenizer, clf
    except Exception as e:
        st.warning(f"Ingredient classifier not available: {e}")
        return None, None


model, vectorizer = load_model_and_vectorizer()
tokenizer, clf_model = load_transformer_classifier()
gluten_ingredients = load_gluten_ingredients()
substitutions = load_substitutions()


# ============================================================
# TRANSFORMER-BASED INGREDIENT PREDICTOR
# ============================================================
def is_ingredient_line_transformer(text):
    if tokenizer is None or clf_model is None:
        return None  # fallback to heuristics
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = clf_model(**tokens).logits
    return bool(torch.argmax(logits, dim=-1).item())


# ============================================================
# URL INGREDIENT EXTRACTION
# ============================================================
def extract_recipe_from_url(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        selectors = [
            '[itemprop="recipeIngredient"]',
            '.recipe-ingredients__list-item',
            '.ingredients-item-name',
            'li.ingredient',
            'li.recipe-ingredients__list-item',
            'span.ingredients-item-name',
        ]

        candidates = []
        for sel in selectors:
            for tag in soup.select(sel):
                txt = clean_text(tag.get_text(" ", strip=True))
                if txt and len(txt.split()) > 1:
                    candidates.append(txt)

        # fallback heuristic
        if not candidates:
            for li in soup.find_all("li"):
                txt = clean_text(li.get_text(" ", strip=True))
                if re.search(r"(cup|tbsp|g|kg|ml|flour|sugar|butter|milk|egg)", txt, re.I):
                    candidates.append(txt)

        candidates = list(dict.fromkeys(candidates))

        # Transformer filtering
        filtered = []
        for c in candidates:
            pred = is_ingredient_line_transformer(c)
            if pred is None:
                # no transformer -> heuristic
                if re.search(r"(cup|g|kg|ml|flour|sugar|butter)", c, re.I):
                    filtered.append(c)
            else:
                if pred:
                    filtered.append(c)

        return filtered[:30]

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

elif mode == "Recipe URL":
    recipe_url = st.text_input("Paste a recipe URL:")
    if st.button("Fetch & Convert"):
        ingredients = extract_recipe_from_url(recipe_url)
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


