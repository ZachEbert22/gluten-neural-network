import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/process"

# -------------- BACKEND CALLER -----------------

def call_backend(raw_text=None, recipe_url=None):
    payload = {
        "raw_text": raw_text,
        "recipe_url": recipe_url
    }
    r = requests.post(API_URL, json=payload)
    r.raise_for_status()
    return r.json()

# ----------------- UI --------------------------

st.set_page_config(page_title="Gluten-Free AI", layout="centered")

st.title("🥖 Gluten-Free AI Ingredient Engine")

st.markdown("""
Paste **ANY** of the following:

- Ingredient list  
- Full recipe text  
- Blog post  
- Recipe URL  
- Anything else — the AI will parse it automatically
""")
recipe_url = st.text_input("🔗 Paste Recipe URL (optional)")
raw_text = st.text_area("📋 Paste ingredient list or recipe text (optional)", height=200)

if st.button("Process"):
    if not recipe_url and not raw_text.strip():
        st.error("Please paste a recipe URL or ingredient text.")
    else:
        try:
            payload = {}
            if recipe_url:
                payload["url"] = recipe_url
            if raw_text.strip():
                payload["raw_text"] = raw_text.strip()

            result = requests.post(API_URL, json=payload)
            result.raise_for_status()
            result = result.json()

            # ---------------- RESULTS ----------------
            st.subheader("🧾 Parsed Ingredients")
            st.json(result.get("parsed", []))

            st.subheader("✨ Ingredient Substitutions")
            st.json(result.get("substitutions", []))

            st.subheader("👩‍🍳 Rewritten Gluten-Free Instructions")
            st.write(result.get("rewritten", ""))

        except Exception as e:
            st.error(f"Backend Error: {str(e)}")
