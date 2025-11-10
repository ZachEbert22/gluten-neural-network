import streamlit as st
import torch
import pandas as pd
from utils.parser import parse_ingredient, format_ingredient, process_recipe
from utils.gluten_check import load_gluten_ingredients, load_substitutions
from utils.model_loader import load_model_and_vectorizer

st.set_page_config(page_title="Gismo – Gluten-Free Recipe Converter", layout="centered")

st.title("🥖 Gismo: Gluten-Free Recipe Converter")
st.markdown("Enter or upload a recipe, and Gismo will suggest gluten-free alternatives.")

# Load assets
gluten_ingredients = load_gluten_ingredients()
substitutions = load_substitutions()
model, vectorizer = load_model_and_vectorizer()

uploaded_file = st.file_uploader("Upload a recipe (.txt or .csv):", type=["txt", "csv"])
user_text = st.text_area("Or paste your recipe ingredients here (one per line):")

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        ingredients = df.iloc[:, 0].dropna().tolist()
    else:
        ingredients = [line.strip() for line in uploaded_file.readlines() if line.strip()]
elif user_text:
    ingredients = [line.strip() for line in user_text.split("\n") if line.strip()]
else:
    ingredients = []

if st.button("Convert to Gluten-Free") and ingredients:
    st.subheader("Original Ingredients")
    st.write(ingredients)

    # Vectorize for prediction
    X = vectorizer.transform(ingredients).toarray()
    with torch.no_grad():
        flag_logits, sub_logits = model(torch.tensor(X, dtype=torch.float32))
        gluten_pred = flag_logits.argmax(dim=1).numpy()
        sub_pred = sub_logits.argmax(dim=1).numpy()

    # Rule-based refinement
    gluten_free = process_recipe(ingredients, gluten_ingredients, substitutions)

    st.subheader("Gluten-Free Version")
    for before, after in zip(ingredients, gluten_free):
        if before != after:
            st.markdown(f"✅ **{before} → {after}**")
        else:
            st.markdown(f"• {before}")

    st.success("Conversion complete! Review substitutions above.")

