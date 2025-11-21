import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/parse_recipe"

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

user_input = st.text_area("Paste recipe text or a URL", height=250)

if st.button("Process"):
    if not user_input.strip():
        st.error("Please paste something first.")
    else:

        try:
            # If the user pasted a URL
            if user_input.strip().startswith("http"):
                result = call_backend(recipe_url=user_input.strip())
            else:
                # Otherwise process raw text
                result = call_backend(raw_text=user_input)

            # ------------ DISPLAY RESULTS ------------

            st.subheader("🧾 Parsed Ingredients")
            st.json(result["ingredients"])

            st.subheader("👩‍🍳 Instructions Found")
            st.json(result["instructions"])

            st.subheader("✨ Gluten-Free Rewritten Recipe")
            st.write(result["rewritten"])

        except Exception as e:
            st.error(f"Backend Error: {str(e)}")

