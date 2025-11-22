import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/api/parse_recipe"

st.set_page_config(page_title="Gluten-Free Converter", layout="wide")

st.title("🧁 Gluten-Free Recipe Converter")

st.write("Paste a recipe OR enter a URL. The AI will parse ingredients, detect gluten, "
         "apply substitutions, and rewrite the instructions.")

# -------------------------------------------------------------------------
# INPUT MODE SELECTION
# -------------------------------------------------------------------------
mode = st.radio(
    "Choose input type:",
    ["Paste recipe text", "Provide recipe URL"],
    horizontal=True
)

# -------------------------------------------------------------------------
# INPUTS
# -------------------------------------------------------------------------
raw_text = None
recipe_url = None

if mode == "Paste recipe text":
    raw_text = st.text_area(
        "Recipe (one ingredient per line):",
        height=200,
        placeholder=(
            "2.75 cups of all-purpose flour\n"
            "2 teaspoons cornstarch\n"
            "1.25 teaspoons kosher salt\n"
            "1 teaspoon baking soda\n"
            "0.75 cups butter\n"
            "1.25 cups sugar\n"
            "1 egg\n"
            "2 tsp vanilla\n"
            "1 cup mashed bananas\n"
            "1.75 cups chocolate chunks"
        )
    )
else:
    recipe_url = st.text_input(
        "Recipe URL:",
        placeholder="https://www.example.com/my-recipe"
    )

instructions_input = st.text_area(
    "Optional instructions text (leave blank to auto-extract or auto-rewrite):",
    height=120
)

submitted = st.button("Convert to Gluten-Free")

# -------------------------------------------------------------------------
# SEND TO BACKEND
# -------------------------------------------------------------------------
if submitted:
    if not raw_text and not recipe_url:
        st.error("Please provide either recipe text or a URL.")
    else:
        payload = {
            "raw_text": raw_text if raw_text else None,
            "url": recipe_url if recipe_url else None,
            "ingredients": None,       # never needed in UI
            "instructions": instructions_input or ""
        }

        try:
            r = requests.post(API_URL, json=payload)
            if r.status_code != 200:
                st.error(f"Backend error {r.status_code}: {r.text}")
            else:
                result = r.json()

                # -----------------------------------------------------------------
                # DISPLAY PARSED INGREDIENTS
                # -----------------------------------------------------------------
                st.subheader("🧾 Parsed Ingredients")
                ing_list = result.get("ingredients", [])

                pretty = [
                    f"- **{i['parsed'].get('quantity','')} {i['parsed'].get('unit','')} "
                    f"{i['parsed'].get('ingredient','')}**"
                    for i in ing_list
                ]

                st.markdown("\n".join(pretty))

                # -----------------------------------------------------------------
                # SUBSTITUTION TABLE
                # -----------------------------------------------------------------
                st.subheader("🔄 Substitutions")

                subs = result.get("substitutions", [])
                st.table([
                    {
                        "Original": s["original"],
                        "Converted": s["converted"],
                        "Status": s["status"]
                    }
                    for s in subs
                ])

                # -----------------------------------------------------------------
                # INSTRUCTIONS
                # -----------------------------------------------------------------
                st.subheader("📘 Rewritten Gluten-Free Instructions")
                st.write(result.get("rewritten", ""))

                # -----------------------------------------------------------------
                # DEV MODE
                # -----------------------------------------------------------------
                with st.expander("🐍 Developer Mode (Raw JSON)"):
                    st.json(result)

        except Exception as e:
            st.error(f"Failed to connect: {e}")

