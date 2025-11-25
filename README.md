# AI-Powered Gluten-Free Recipe Transformation  
An end-to-end AI system that converts any recipe — URL or text — into a fully gluten-free version.  
Built using ingredient parsing, BERT classification, semantic substitution, GISMo graph reasoning, and a unified FastAPI + Streamlit interface.

# Overview

This project provides a complete AI pipeline that:

1. **Parses raw recipe text** OR **scrapes ingredients from recipe URLs**
2. **Normalizes units, quantities, and ingredient names**
3. **Classifies ingredients** using a **fine-tuned BERT model**
4. **Detects gluten-containing items**
5. **Generates gluten-free substitutions** using:
   - Direct substitutions (`substitutions.json`)
   - BERT embedding similarity (semantic matching)
   - **GISMo graph model** (optional)
   - Optional **FoodNER** ingredient tagging
6. **Rewrites the recipe fluently** using a SHARE-inspired rewriting model
7. **Outputs the final cleaned, gluten-free ingredient list**

Includes:
- **Unified backend** (FastAPI)
- **Frontend UI** (Streamlit)
- **Training scripts**
- **Ingredient dataset builder**
- **Evaluation and graphing tools**

# Locations

root/
    - build_ingredient_dataset.py
    - frontend.py
    - mlp_model.py
    - substitution_pipeline.py
    - train_ingredient_classifier.py
    - unfied_api.py

data/
    - ingredients.json
    - substitutions.json
    - ingredient_dataset/          <- auto-created
    - ingredient_classifier/       <- auto-created

models/
    - bert_embedder.py
    - food_ner.py
    - share_rewriter.py
    - gluten_model.py              <- older model
    - ingredient_classifier/       <- config, tokenizer, vocab, weights

utils/
    - ingredient_parser.py
    - gismo.py
    - normalization.py
    - model_loader.py
    - gluten_check.py
    - parser.py
    - substitution.py

tools/
    - backend.py      <- creates training graphs
    - core.py
    - confusion_matrix.py
    - models.py
    - datasets.py
    - reports/        <- output
    - system_info.py

# Each Script Description 

## Top-Level Scripts

### **`build_ingredient_dataset.py`**
**Purpose:** Creates the ingredient dataset needed to train the ingredient classifier.

**Responsibilities:**
- Reads raw ingredient lists.
- Normalizes via `utils/ingredient_parser.py`.
- Generates training/validation splits.
- Outputs structured files into:
  - `data/ingredient_dataset/`
  - `data/ingredient_classifier/`


### **`train_ingredient_classifier.py`**
**Purpose:** Trains the BERT ingredient classifier.

**Uses:**
- HuggingFace Transformers
- Dataset created from `build_ingredient_dataset.py`

**Outputs Model Folder:**
- models/ingredient_classifier/
- config.json
- pytorch_model.bin
- vocab.txt
- tokenizer.json
- tokenizer_config.json

### **`unified_api.py`**
**Purpose:**  
The **core FastAPI backend** used by the frontend.

**Loads & Manages:**
- Ingredient classifier
- BERT embedder (`models/bert_embedder.py`)
- FoodNER (`models/food_ner.py`)
- GISMo graph model (`utils/gismo.py`)
- Substitution engine (`substitution_pipeline.py`)
- Ingredient parser (`utils/ingredient_parser.py`)
- SHARE rewrite model (`models/share_rewriter.py`)
- Normalization overrides (`utils/normalization.py`)

**Endpoints:**
- `POST /process` → Unified inference pipeline  
  (ingredients → classification → substitution → rewrite → output)

## **`frontend.py`**
**Purpose:**  
A Streamlit UI for interacting with the backend.

**Supports:**
- Paste recipe URL  
- Paste recipe text (multi-line)  
- Line-by-line ingredient parsing  
- Displays:
  - normalized ingredients
  - gluten-free substitutions
  - rewritten SHARE output  
  - optional backend debug logs

### **`mlp_model.py`** *(Legacy Model)*
A fully connected neural network used in the earliest prototypes.

**Notes:**  
- **Not part of the current system.**
- Kept for historical and comparison purposes.

### **`substitution_pipeline.py`**
**Purpose:**  
Implements the gluten substitution logic.

**Key Components:**
- Loads `data/substitutions.json`
- Performs:
  1. Exact match substitutions  
  2. Semantic similarity (BERT) match  
  3. GISMo graph-based fallback  
  4. Optional FoodNER enhancement  
- Parses ingredient strings  
  (`utils/ingredient_parser.py`)

**Main Class:**  
`SubstitutionEngine`

## Data Directory

### **`data/ingredient_classifier/`**
Generated after dataset building and training.

Contains:
- tokenizer
- vocab
- BERT config
- model weights

### **`data/ingredient_dataset`** + **`data/ingredient_classifier_dataset`**
Generated dataset used during training.

### **`data/ingredients.json`**
Human-curated ingredient dictionary.

### **`data/substitutions.json`**
Maps *gluten ingredients → gluten-free equivalents*.

### **`data/prediction_log.jsonl`**
Backend logging file used by tools to generate:
- performance metrics
- confusion matrices
- accuracy stats

## Documentation (docs/)

### **`AI-Guided-Gluten-Free-Recipe-Transformation.docx`**
Main project paper.

### **`AI-Gluten-Free-Recipe-Transformation-Models.docx`**
Extended technical model descriptions.

### **`Checkpoint1.md`, `Checkpoint2.md`**
Project milestone documents.

### **`requirements.txt`**
List of all required Python packages.

### **`system_requirements.txt`**
Exact versions tested on GPU and CPU systems.

### **`PowerPoint.docx'**
Content that will go Into the Poster and Powerpoint (Code Demo)

## Models (models/)

### **`models/bert_embedder.py`**
Provides:
- BERT embeddings  
- Mean pooling encoder  
- Used by semantic substitution.

Methods:
- `embed(text)`
- `embed_texts(list_of_texts)`

### **`models/food_ner.py`**
Named entity recognition for ingredient segmentation.

Tags:
- QUANTITY  
- UNIT  
- INGREDIENT  
- DESCRIPTOR  

Improves parsing robustness.

### **`models/gluten_model.py`** *(Legacy)*
Old model used by `mlp_model.py`.

### **`models/ingredient_classifier/`**
Trained BERT model folder containing:
- tokenizer
- vocabulary
- HF config
- pytorch model

Loaded automatically by `unified_api.py`.

### **`models/share_rewriter.py`**
Implements the SHARE-style rewriting model.

Purpose:
- Takes substituted text  
- Rewrites for naturalness and readability  

Used as the *final step* of the pipeline.

## Utilities (utils/)

### **`utils/gismo.py`**
Graph-based semantic substitution engine.

Used when:
- BERT similarity is too low  
- Ingredient is ambiguous

### **`utils/gluten_check.py`**
Legacy MLP gluten-detection utility.

Mostly deprecated.

### **`utils/ingredient_parser.py`**
One of the most important files.

Performs:
- quantity extraction  
- unit normalization  
- FoodNER optional tagging  
- ingredient normalization  
- removal of nutritional lines  
- fuzzy cleaning  

Used by both:
- dataset generation  
- inference pipeline  

### **`utils/model_loader.py`**
Generic loader for:
- BERT models
- tokenizer directories
- fallback paths

### **`utils/normalization.py`**
Overrides for:
- ingredient names
- unit correction
- special-case mappings

### **`utils/parser.py`**
Legacy parser from early prototypes.

Kept for reference.

### **`utils/substitution.py`**
Older substitution helper file.

Some functions still used indirectly.

### **`utils/train_utils.py`**
Training helper functions used by the legacy MLP model.

## Tools (tools/)

### **`backend.py`**
Used for generating backend training graphs:
- training loss curves
- validation performance
- embedding visualizations

### **`core.py`**
Frontend graph utilities.

### **`confusion_matrix.py`**
Generates:
- confusion matrices  
- precision/recall charts  

### **`datasets.py`**
DataLoader creation utilities for training.

### **`models.py`**
Legacy model instantiation utilities.

### **`system_info.py`**
Prints:
- CUDA availability  
- GPU model  
- Python environment info  
- Installed libraries  

Used for debugging user environments.

### **`tools/reports/`**
Where graphs are output (PNG, PDF, reports).

# Recommended Environment Setup (Conda)

Conda is strongly recommended for this project.

`conda create -n gluten python=3.10`
`conda activate gluten`

## Install Dependencies

All Required dependencies are provided in:
docs/requirements.txt
docs/system_requirements.txt

Install with
`pip install -r docs/requirements.txt`

Major libraries used:
    - torch (with CUDA recommended)
    - transformers
    - fastapi
    - uvicorn
    - streamlit
    - beautifulsoup4
    - pandas, numpy, scikit-learn
    - matplotlib, seaborn
    - validators
    - requests

# GPU Installation for Pytorch
    - `pip install torch --index-url https://download.pytorch.org/whl/cu121`
    - Use CUDA if possible — training and inference are 10–20× faster

## How to Run the Code

python build_ingredient_dataset.py --max-pos 50000 --max-neg 50000

    - datasize:
        the datasize is 50000 in the example: Would Recommend between 20000 and 50000 For optional Training Loss
        ex: python build_ingredient_dataset.py --max-pos 50000 --max-neg 50000
    - generates: 
        data/ingredient_dataset
         data/ingredient_classifier/ (train/test splits for BERT)

_run Concurrently_
python -m uvicorn unified_api:app --reload --port 8000
streamlit run frontend.py

    - Two Commands runs unified_api.py and frontend.py
    - Outputs Data Information
        - models/ingredient_classifier/
            - Remembers user information for datasets
            - data/prediction_log.jsonl
        - Recommend using GPU, CPU can take hours/days

# Testing the System
    sample 1: 
        2 cups all-purpose flour
        1.25 tsp kosher salt
        1 cup mashed banana
        1.75 cups dark chocolate chunks
    sample 2:
        https://www.loveandlemons.com/brownies-recipe/

# Running the Backend Scripts
    Generate Training Curves
        `python3 tools/backend.py`
    Generate Confusion Matrix   
        `python3 tools/confusion_matrix.py`
    Check System GPU
        `python3 tools/system_info.py`
    Will Be Stored in tools/reports

# Summary 

Ingredient Parsing

    - Regex, NER, quantity detection, unit normalization.

Ingredient Classification

    - Custom BERT classifier.

Gluten Detection

    - Combines ML + rule-based filtering + curated ingredient lists.

Substitution Pipeline

    - direct mapping (substitutions.json)

    - semantic embedding similarity

    - GISMo graph model support

    - fallback heuristics

Recipe Rewriting

    - SHARE-style rewriting for clarity and fluency.

Full Web Interface

    - Clean UX with Streamlit + FastAPI backend. 
