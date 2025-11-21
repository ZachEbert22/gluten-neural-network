# streamlit_app.py
import streamlit as st
import requests
from pathlib import Path
import re

BACKEND_URL = "http://127.0.0.1:8000/process"

st.set_page_config(page_title="AI Gluten-Free Converter", layout="centered")
st.title("🥣 AI Gluten-Free Converter — Unified Backend")
st.markdown("Paste ingredients or a recipe URL. Backend does NER, parsing, GISMo substitutions and recipe rewriting.")

mode = st.radio("Input mode", ["Paste Recipe Text", "Recipe URL"])

if mode == "Paste Recipe Text":
    st.markdown("**Enter ingredients (one per line)** OR paste entire recipe text. The backend will auto-detect.")
    text = st.text_area("Enter ingredients or recipe text:", height=240)
    instr = st.text_area("Recipe instructions (optional):", height=160)
    if st.button("Convert"):
        if not text.strip():
            st.error("Please paste recipe text or ingredient lines.")
        else:
            payload = {"raw_text": text, "instructions": instr}
            with st.spinner("Sending to backend..."):
                try:
                    r = requests.post(BACKEND_URL, json=payload, timeout=60)
                    r.raise_for_status()
                    resp = r.json()
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    resp = None

            if resp:
                st.subheader("Parsed Items (detected)")
                for item in resp.get("parsed", []):
                    st.write("-", item["original"], "→", item["parsed"])

                st.subheader("Substitutions")
                for s in resp.get("substitutions", []):
                    st.write(f"- {s['original']} → {s['converted']}  ({s['status']})")

                st.subheader("Rewritten Instructions")
                if resp.get("rewritten"):
                    st.write(resp["rewritten"])
                else:
                    st.info("No rewritten instructions returned.")

elif mode == "Recipe URL":
    st.markdown("Paste a recipe URL; backend will try to extract ingredients & instructions.")
    url = st.text_input("Recipe URL:")
    if st.button("Fetch & Convert"):
        if not url.strip():
            st.error("Please paste a URL.")
        else:
            payload = {"url": url}
            with st.spinner("Fetching & processing..."):
                try:
                    r = requests.post(BACKEND_URL, json=payload, timeout=60)
                    r.raise_for_status()
                    resp = r.json()
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    resp = None

            if resp:
                st.subheader("Extracted & Parsed Items")
                for item in resp.get("parsed", []):
                    st.write("-", item["original"], "→", item["parsed"])

                st.subheader("Substitutions")
                for s in resp.get("substitutions", []):
                    st.write(f"- {s['original']} → {s['converted']}  ({s['status']})")

                st.subheader("Rewritten Instructions")
                if resp.get("rewritten"):
                    st.write(resp["rewritten"])
                else:
                    st.info("No rewritten instructions returned.")

