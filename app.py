import streamlit as st
import torch
import pickle
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from models.gluten_model import GlutenSubstitutionNet
from utils.parser import parse_ingredient, format_ingredient
from utils.gluten_check import load_gluten_ingredients, load_substitutions
from utils.substitution import substitute_ingredient

# ------------------------------
# LOAD MODEL + VECTORIZER
# ------------------------------
@st.cache_resource
def load_model_and_vectorizer(model_path="models/model.pth", vec_path="models/vectorizer.pkl"):
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Dynamically detect num_substitutes from checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    num_subs = checkpoint["out_substitute.weight"].shape[0] if "out_substitute.weight" in checkpoint else 5

    model = GlutenSubstitutionNet(
        input_dim=len(vectorizer.get_feature_names_out()),
        hidden_dim=128,
        num_substitutes=num_subs
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
gluten_ingredients = load_gluten_ingredients()
substitutions = load_substitutions()

# ------------------------------
# HELPER: EXTRACT RECIPE FROM URL
# ------------------------------
def extract_recipe_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        ingredients = []

        # Common patterns for sites like AllRecipes, Food.com, Epicurious
        possible_tags = soup.find_all(["li", "span"], string=re.compile(r"cup|tbsp|tsp|flour|sugar|salt|oil|egg", re.I))
        for tag in possible_tags:
            text = tag.get_text(strip=True)
            if len(text.split()) > 1 and not re.search(r"reviews?|comments?", text, re.I):
                ingredients.append(text)
        ingredients = list(dict.fromkeys(ingredients))  # deduplicate
        return ingredients[:15] if ingredients else []
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


