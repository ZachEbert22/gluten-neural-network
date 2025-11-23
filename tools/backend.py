#!/usr/bin/env python3
"""
tools/statistics_backend.py

Generates backend-dependent graphs (7 & 8):
7) Substitution Success Rate (calls local backend or substitution engine directly)
8) Confidence Drop-off Graph for a target category (flour) vs name length and adjectives

Outputs saved in reports/plots and JSON stats.

Usage:
    python tools/statistics_backend.py --host http://127.0.0.1:8000
If backend not running, will try to import local SubstitutionEngine and run directly.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

ROOT = Path(".")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
STATS_DIR = REPORTS_DIR / "stats"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
import os


def load_prediction_log():
    log_file = os.path.join(DATA_DIR, "prediction_log.jsonl")

    if not os.path.isfile(log_file):
        print(f"Could not find prediction_log.jsonl at: {log_file}")
        return []

    lines = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except:
                pass
    return lines

# For fallback embedding-based confidence
def cosine(a,b):
    import numpy as np
    a = np.array(a); b = np.array(b)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def call_api_process(host, payload):
    url = host.rstrip("/") + "/process"
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# Try to import substitution engine classes if backend not available
LocalSubEngine = None
Embedder = None
try:
    from substitution_pipeline import SubstitutionEngine, load_substitutions
    from models.bert_embedder import BertEmbedder
    LocalSubEngine = SubstitutionEngine
    Embedder = BertEmbedder
except Exception as e:
    # if import fails, we still can run an embedding-based fallback using models/bert_embedder if present
    try:
        from models.bert_embedder import BertEmbedder
        Embedder = BertEmbedder
    except Exception:
        Embedder = None

# Read dataset lines (sample)
def load_sample_lines(limit=1000):
    df1 = None
    try:
        df1 = pd.read_csv(DATA_DIR / "ingredient_dataset.csv")
    except Exception:
        try:
            df1 = pd.read_csv(DATA_DIR / "ingredient_classifier_dataset.csv")
        except Exception:
            df1 = None
    if df1 is None:
        return []
    # guess text column
    if "ingredient" in df1.columns:
        col = "ingredient"
    elif "text" in df1.columns:
        col = "text"
    else:
        # pick first object column
        col = [c for c in df1.columns if df1[c].dtype == object]
        col = col[0] if col else df1.columns[0]
    lines = df1[col].dropna().astype(str).tolist()
    # flatten multi-line cells
    out = []
    for l in lines:
        if "\n" in l:
            out.extend([x.strip() for x in l.splitlines() if x.strip()])
        else:
            out.append(l.strip())
    return out[:limit]

# -------------------------
# 7) Substitution Success Rate
# -------------------------
def substitution_success_rate(lines, host=None):
    """
    For each line, call backend /process or local substitution engine.
    Compute substitution success metrics and store details.
    """
    results = []
    use_api = False
    if host:
        try:
            # quick health check
            resp = requests.get(host.rstrip("/") + "/docs", timeout=3)
            use_api = True
        except Exception:
            use_api = False

    subs_json_path = DATA_DIR / "substitutions.json"
    if subs_json_path.exists():
        with open(subs_json_path, "r") as f:
            subs_json = json.load(f)
    else:
        subs_json = None

    # prepare local engine if available
    engine = None
    if not use_api and LocalSubEngine and subs_json:
        try:
            engine = LocalSubEngine(subs_json)
        except Exception as e:
            engine = None

    success_total = 0
    detectable = 0
    can_sub = 0
    total_lines = len(lines)
    details = []
    for line in tqdm(lines, desc="Processing lines"):
        payload = {"ingredients": [line]}
        converted_line = None
        status = "error"
        try:
            if use_api:
                r = call_api_process(host, payload)
                # r contains 'substitutions' list
                subs = r.get("substitutions", [])
                if subs:
                    out = subs[0]
                    converted_line = out.get("converted")
                    status = out.get("status")
            else:
                if engine:
                    converted_line, changed = engine.substitute(line)
                    status = "substituted" if changed else "gluten_free"
                else:
                    # no engine — fallback: try simple substring lookup
                    if subs_json:
                        norm = line.lower()
                        matched = None
                        for k,info in subs_json.items():
                            if k in norm:
                                matched = info.get("substitute")
                                break
                        if matched:
                            converted_line = matched
                            status = "substituted"
                        else:
                            converted_line = line
                            status = "gluten_free"
                    else:
                        converted_line = line
                        status = "unknown"
        except Exception as e:
            converted_line = None
            status = "error"

        if status in ("substituted", "substituted_by_gismo", "substituted_by_rule"):
            can_sub += 1
        if status in ("gluten_free","not_parsed"):
            # not a substitution candidate
            pass
        # attempt detection: if the converted differs from original we call success_total++ for now
        if converted_line and converted_line != line:
            success_total += 1

        # store details
        details.append({
            "original": line,
            "converted": converted_line,
            "status": status
        })
    # compute metrics
    detectable = sum(1 for d in details if d["status"] != "not_parsed" and d["status"] != "error")
    can_sub = sum(1 for d in details if d["status"].startswith("substit"))
    substitution_rate = (can_sub / detectable * 100.0) if detectable else 0.0
    coverage = {"total": total_lines, "detectable": detectable, "can_sub": can_sub, "substitution_rate_pct": substitution_rate}
    # save CSV & JSON
    outp = STATS_DIR / "07_substitution_details.json"
    with open(outp, "w") as f:
        json.dump({"coverage": coverage, "details_sample": details[:200]}, f, indent=2)
    # bar chart
    plt.figure(figsize=(6,4))
    plt.bar(["detectable", "can_sub", "not_detectable"], [detectable, can_sub, total_lines-detectable])
    plt.title("Substitution success / coverage")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_substitution_success_rate.png", dpi=150)
    plt.close()
    print("Saved 07_substitution_success_rate.png and JSON")
    return coverage, details

# -------------------------
# 8) Confidence Drop-off Graph (for 'flour')
# -------------------------
def confidence_dropoff(lines, host=None):
    """
    For each flour-like line, measure the 'confidence' of substitution:
     - If backend returns a confidence, use it
     - Else compute embedding-based cosine score between query and best candidate (if embedder available)
    Plot confidence vs ingredient length and vs presence of adjectives.
    """
    flour_lines = [l for l in lines if re.search(r"\b(flour|bread|wheat|wholemeal|all-purpose|all purpose|plain)\b", l, re.I)]
    if not flour_lines:
        print("No flour-like lines found in sample.")
        return {}

    # try using API to get semantic scores (if API returns them)
    host_api = None
    try:
        if host and requests.get(host.rstrip("/") + "/docs", timeout=3).ok:
            host_api = host
    except Exception:
        host_api = None

    # prepare embedder fallback
    emb = None
    candidate_embs = None
    candidates = None
    if Embedder:
        try:
            emb = Embedder()
            # load substitutions.json as candidates
            sj = DATA_DIR / "substitutions.json"
            if sj.exists():
                import json
                with open(sj, "r") as f:
                    subsd = json.load(f)
                # candidates = [k for k in subsd.keys()]
                candidates = [v.get("substitute") if isinstance(v,dict) and v.get("substitute") else v for v in subsd.values()]
                # compute embeddings
                candidate_embs = emb.embed_texts(candidates, batch_size=32) if hasattr(emb, "embed_texts") else None
        except Exception:
            emb = None

    rows = []
    for l in flour_lines:
        conf = None
        source = None
        # try API
        if host_api:
            try:
                r = requests.post(host_api.rstrip("/") + "/process", json={"ingredients":[l]})
                if r.ok:
                    resp = r.json()
                    subs = resp.get("substitutions", [])
                    if subs:
                        first = subs[0]
                        # sometimes backend returns score in different key names; check common ones
                        conf = first.get("score") or first.get("confidence") or None
                        source = "api"
            except Exception:
                pass
        # embedder fallback: compute best cosine of query -> candidate_embs
        if conf is None and emb and candidate_embs is not None:
            try:
                with __import__("torch").no_grad():
                    qv = emb.embed(l) if hasattr(emb, "embed") else None
                if qv is not None:
                    # move node vectors to qv device if necessary
                    import torch
                    if isinstance(candidate_embs, torch.Tensor):
                        # ensure devices match
                        if qv.device != candidate_embs.device:
                            node = candidate_embs.to(qv.device)
                        else:
                            node = candidate_embs
                        sims = torch.nn.functional.cosine_similarity(qv, node)
                        best = float(torch.max(sims).item())
                        conf = float(best)
                        source = "embed_cos"
                    else:
                        # numpy fallback
                        conf = 0.0
                        source = "none"
            except Exception:
                conf = None

        # heuristics for length & adjectives
        length = len(l.split())
        adjectives = bool(re.search(r"\b(white|wholemeal|whole|plain|all-purpose|self-|corn|gluten-free|dark|light)\b", l.lower()))
        rows.append({"original": l, "length": length, "has_adj": adjectives, "confidence": conf, "source": source})

    # convert to dataframe
    rdf = pd.DataFrame(rows)
    rdf["confidence_filled"] = rdf["confidence"].fillna(0.0)
    # plot: confidence vs length
    plt.figure(figsize=(8,4))
    plt.scatter(rdf["length"], rdf["confidence_filled"])
    plt.xlabel("Ingredient token length")
    plt.ylabel("Confidence (0-1)")
    plt.title("Confidence vs ingredient length (flour-like lines)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_confidence_vs_length.png", dpi=150)
    plt.close()

    # boxplot: confidence by adjective presence
    plt.figure(figsize=(6,4))
    rdf.boxplot(column="confidence_filled", by="has_adj")
    plt.title("Confidence by presence of adjectives (flour lines)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_confidence_by_adj.png", dpi=150)
    plt.close()

    with open(STATS_DIR / "08_confidence_dropoff.json", "w") as f:
        json.dump({"rows": len(rows)}, f, indent=2)
    print("Saved 08_confidence* plots and JSON")
    return rows

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Backend host URL (optional)")
    parser.add_argument("--sample", type=int, default=500, help="Number of sample lines to test")
    args = parser.parse_args()

    print("Loading sample lines from data/...")
    lines = load_sample_lines(limit=args.sample)
    if not lines:
        print("No lines found in data/ - aborting")
        return

    print(f"Loaded {len(lines)} lines; running substitution success test (may take time)...")
    cov, details = substitution_success_rate(lines, host=args.host)

    print("Running confidence drop-off analysis (flour)...")
    conf = confidence_dropoff(lines, host=args.host)

    print("Backend stats complete. Check reports/plots and reports/stats.")

if __name__ == "__main__":
    main()

