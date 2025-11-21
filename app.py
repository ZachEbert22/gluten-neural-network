#!/usr/bin/env python3
"""
Streamlit UI:
- Loads trained ingredient classifier at models/ingredient_classifier
- Loads BERT embedder and precomputes embeddings for substitution candidates
- Extracts ingredients from URL or pasted text
- Filters lines with the ingredient classifier
- For each ingredient, finds best substitute via cosine similarity
- Falls back to rule-based substitutions.json if similarity is below threshold
"""
import streamlit as st
import torch
import json
import requests
import re
from bs4 import BeautifulSoup
from pathlib import Path
from models.bert_embedder import BertEmbedder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.normalization import normalize_ingredient
from utils.parser import parse_ingredient, format_ingredient
from utils.gluten_check import load_gluten_ingredients, load_substitutions

# Load gluten keyword list (used to avoid substituting gluten-free items)
GLUTEN_LIST = load_gluten_ingredients()

def contains_gluten(ingredient_name: str) -> bool:
    """Return True if ingredient contains any gluten keyword."""
    ing = ingredient_name.lower()
    return any(g in ing for g in GLUTEN_LIST)

# ---------------------------
# Config
# ---------------------------
ING_CLASS_MODEL_DIR = "models/ingredient_classifier"  # output of training
SUBS_JSON = Path("data/substitutions.json")
TOP_K = 1
SIM_THRESHOLD = 0.88   # cosine similarity threshold for accepting a semantic match

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_ingredient_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ING_CLASS_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(ING_CLASS_MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource
def load_embedder_and_candidates():
    # load substitutions json
    substitutions = load_substitutions()
    # candidate strings = substitute names (unique)
    candidates = []
    # Keep mapping from candidate index -> (substitute_string, ratio)
    cand_meta = []
    seen = set()
    for orig, d in substitutions.items():
        sub = d.get("substitute")
        ratio = float(d.get("ratio", 1.0))
        if sub and sub not in seen:
            seen.add(sub)
            candidates.append(sub)
            cand_meta.append({"orig": orig, "sub": sub, "ratio": ratio})
    embedder = BertEmbedder()
    # compute candidate embeddings (these will be returned on CPU by embed_texts)
    candidate_embs = embedder.embed_texts(candidates, batch_size=64)  # CPU tensor
    # keep metadata and return device used by embedder
    return embedder, candidates, candidate_embs, cand_meta, substitutions, embedder.device

tok_ing, clf_ing, clf_device = load_ingredient_classifier()
embedder, candidates, candidate_embs, cand_meta, substitutions, embedder_device = load_embedder_and_candidates()

# ---------------------------
# Helper: ingredient classifier inference
# ---------------------------
def is_ingredient_line_transformer(text: str, threshold: float = 0.5) -> bool:
    enc = tok_ing(text, return_tensors="pt", truncation=True, max_length=128)
    # move inputs to classifier device
    enc = {k: v.to(clf_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = clf_ing(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    # label 1 = ingredient
    return float(probs[1]) >= threshold

# ---------------------------
# Helper: semantic substitution using BERT embedder
# ---------------------------
def semantic_substitute(ingredient_text: str):
    matches = embedder.nearest(ingredient_text, candidates, candidate_embs, top_k=TOP_K)
    if not matches:
        return None, 0.0
    idx, score = matches[0]
    if score >= SIM_THRESHOLD:
        meta = cand_meta[idx]
        return meta, score
    return None, score

# ---------------------------
# Extract ingredients from URL (heuristic + structural)
# ---------------------------
def extract_recipe_from_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return []

    selectors = [
        '[itemprop="recipeIngredient"]',
        '.ingredients-item-name',
        '.recipe-ingredients__list-item',
        'li.ingredient',
        'span.ingredients-item-name',
    ]
    candidates = []
    for sel in selectors:
        for tag in soup.select(sel):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                candidates.append(txt)

    # fallback: any <li> that looks culinary
    if not candidates:
        for li in soup.find_all("li"):
            txt = li.get_text(separator=" ", strip=True)
            if re.search(r'\b(cup|tsp|tbsp|ml|g|flour|sugar|salt|oil|butter|milk|egg|spice|bread)\b', txt, re.I):
                candidates.append(txt)

    # dedupe/order-preserve
    seen = set()
    clean = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            clean.append(re.sub(r'\s+', ' ', c).strip())
    return clean

# ---------------------------
# UI
# ---------------------------
st.title("🥣 AI Gluten-Free Converter — BERT Substitution (Option 1)")
st.markdown("Paste ingredients or a recipe URL. Uses BERT to detect ingredient lines and find semantic substitutes.")

mode = st.radio("Input mode", ["Paste Recipe Text", "Recipe URL"])

if mode == "Paste Recipe Text":
    text = st.text_area("Enter ingredients (one per line):")
    if st.button("Convert"):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        st.subheader("Converted:")
        for orig in lines:
            # ingredient detection
            if not is_ingredient_line_transformer(orig):
                if not re.search(r"\d|\b(cup|cups|tsp|tbsp|teaspoon|tablespoon|gram|g|ml|ounce|oz|stick|egg|banana)\b", orig, re.I):
                    st.write(f"⚠ Not recognized as ingredient: {orig}")
                    continue

            parsed = parse_ingredient(orig)
            raw_ing = parsed.get("ingredient", "").lower()
            ingredient_name = normalize_ingredient(raw_ing)

            if len(ingredient_name.split()) > 2:
                st.write(f"🟦 {orig} → (dish, not substituted)")
                continue

            # Skip ingredients that do not contain gluten
            if not contains_gluten(ingredient_name):
                st.write(f"🟦 {orig} → (naturally gluten-free, unchanged)")
                continue

            # try semantic substitute
            meta, score = semantic_substitute(ingredient_name)
            if meta:
                # Use ratio from meta, adjust quantity if parsed quantity available
                qty = parsed.get("quantity") or "1"
                try:
                    qf = float(qty)
                except:
                    qf = 1.0
                new_qty = round(qf * meta["ratio"], 2)
                out = {"quantity": str(new_qty), "unit": parsed.get("unit",""), "ingredient": meta["sub"]}
                formatted = format_ingredient(out)
                st.write(f"✅ {orig} → {formatted}  (semantic score {score:.2f})")
            else:
                # fallback rule-based substring match in substitutions.json
                substituted = None
                name = ingredient_name
                for g, info in substitutions.items():
                    if g in name:
                        new_qty = parsed.get("quantity") or "1"
                        try:
                            qf = float(new_qty)
                        except:
                            qf = 1.0
                        out = {"quantity": str(round(qf * info.get("ratio",1.0),2)),
                               "unit": parsed.get("unit",""),
                               "ingredient": info.get("substitute")}
                        formatted = format_ingredient(out)
                        substituted = formatted
                        break
                if substituted:
                    st.write(f"✅ {orig} → {substituted}  (rule fallback)")
                else:
                    st.write(f"❌ {orig} → No substitute found (try extending substitutions.json)")

elif mode == "Recipe URL":
    url = st.text_input("Paste recipe URL:")
    if st.button("Fetch & Convert"):
        candidates = extract_recipe_from_url(url)
        if not candidates:
            st.warning("Could not extract ingredients from the URL.")
        else:
            st.subheader("Extracted (raw):")
            for c in candidates[:40]:
                st.write("-", c)

            st.subheader("Filtered & Converted:")
            for c in candidates:
                if re.match(r"^(fat|saturates|carbs|sugars|fibre|fiber|protein|calories)\b", c, re.I):
                    continue
                if not is_ingredient_line_transformer(c):
                    # skip non-ingredient lines
                    continue
                parsed = parse_ingredient(c)
                ingredient_name = parsed.get("ingredient", "").lower()

                # Skip ingredients that are already gluten-free
                if not contains_gluten(ingredient_name):
                    st.write(f"🟦 {c} → (naturally gluten-free, unchanged)")
                    continue

                meta, score = semantic_substitute(c)
                if meta:
                    qty = parsed.get("quantity") or "1"
                    try:
                        qf = float(qty)
                    except:
                        qf = 1.0
                    new_qty = round(qf * meta["ratio"], 2)
                    out = {"quantity": str(new_qty), "unit": parsed.get("unit",""), "ingredient": meta["sub"]}
                    formatted = format_ingredient(out)
                    st.write(f"✅ {c} → {formatted}  (semantic score {score:.2f})")
                else:
                    # rule fallback
                    name = parsed.get("ingredient","").lower()
                    substituted = None
                    for g, info in substitutions.items():
                        if g in name:
                            qty = parsed.get("quantity") or "1"
                            try:
                                qf = float(qty)
                            except:
                                qf = 1.0
                            out = {"quantity": str(round(qf * info.get("ratio",1.0),2)),
                                   "unit": parsed.get("unit",""),
                                   "ingredient": info.get("substitute")}
                            formatted = format_ingredient(out)
                            substituted = formatted
                            break
                    if substituted:
                        st.write(f"✅ {c} → {substituted}  (rule fallback)")
                    else:
                        st.write(f"❌ {c} → No substitute found")

# ---------------------------
# Debugging area (optional)
# ---------------------------
st.markdown("---")
if st.checkbox("Show substitution candidates (debug)"):
    st.write("Candidates:", candidates)
    st.write("Candidate meta:", cand_meta)
    st.write("Substitutions.json:", substitutions)

