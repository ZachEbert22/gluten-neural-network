import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/process"

st.set_page_config(
    page_title="Gluten-Free Recipe AI",
    layout="centered",
)

st.title("🍞 Gluten-Free Recipe Converter")
st.write("Convert any recipe into a gluten-free version with smart ingredient substitutions.")

# ----------------------------------------------------------
# User Input Section
# ----------------------------------------------------------

st.subheader("Recipe Input")
recipe_text = st.text_area(
    "Paste your recipe (ingredients + method).",
    height=260,
    placeholder="e.g. 2 cups all-purpose flour...\n1 tbsp soy sauce...\n\nMix all ingredients...",
)

st.divider()

st.subheader("Options")
user_mode = st.checkbox("Developer Mode (show raw API output)", value=False)

if st.button("Convert Recipe", type="primary"):
    if not recipe_text.strip():
        st.error("Please paste a recipe first.")
        st.stop()

    with st.spinner("Processing recipe..."):
        try:
            payload = {"text": recipe_text}
            r = requests.post(API_URL, json=payload)
        except Exception as e:
            st.error(f"Could not contact backend: {e}")
            st.stop()

        if r.status_code != 200:
            st.error(f"Backend Error {r.status_code}: {r.text}")
            st.stop()

        data = r.json()

    # ----------------------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------------------
    st.success("Recipe processed successfully!")

    # -------------------------
    # 1) Pretty Ingredient List
    # -------------------------
    st.subheader("🧾 Parsed Ingredients")

    ing_list = data.get("ingredients", [])

    for ing in ing_list:
        name = ing.get("name", "")
        qty = ing.get("quantity", "")
        unit = ing.get("unit", "")
        txt = f"**{name}** — {qty} {unit}".strip()
        st.markdown(f"- {txt}")

    st.divider()

    # -------------------------
    # 2) Substitution Table
    # -------------------------
    st.subheader("🔄 Smart Gluten-Free Substitutions")

    subs = data.get("substitutions", [])

    if not subs:
        st.info("No substitutions needed!")
    else:
        import pandas as pd
        df = pd.DataFrame(subs)
        # Expected schema: ingredient, substitute, reason, score
        if "score" in df.columns:
            df["score"] = df["score"].round(3)

        st.dataframe(df, use_container_width=True)

    st.divider()

    # -------------------------
    # 3) Clean Method Section
    # -------------------------
    st.subheader("👨‍🍳 Cooking Instructions")
    method = data.get("steps", data.get("method", ""))

    if method:
        st.write(method)
    else:
        st.info("No method section detected in recipe.")

    # -------------------------
    # 4) Developer Mode
    # -------------------------
    if user_mode:
        st.divider()
        st.subheader("🔧 Developer Mode")

        st.write("### Raw JSON Response")
        st.json(data)

        if "debug" in data:
            st.write("### Debug Info")
            st.json(data["debug"])


