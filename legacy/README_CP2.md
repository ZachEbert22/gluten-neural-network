## Slide 1: Problem Statement

- **Goal:** Develop an AI system to convert recipes containing gluten ingredients into gluten-free versions.
- **Checkpoint 1 Summary:**
  - Used static JSON files (`ingredients.json`, `substitutions.json`) to map replacements manually.
  - Rule-based system worked for simple substitutions but lacked adaptability.
- **Problem Identified:**
  - No ability to generalize beyond explicitly listed gluten ingredients.
  - Could not handle novel recipes or ingredient variations.
- **Checkpoint 2 Objective:**
  - Introduce a **neural network (MLP)** to learn substitution behavior from a real recipe dataset (Food.com Kaggle data).

## Slide 2: Updated Methodology

**System Architecture Overview:**

1. **Dataset Integration**
   - Paired columns `RecipeIngredientParts` and `RecipeIngredientQuantities`.
   - Generated normalized text like `"1 cup wheat flour"`.
2. **Text Feature Extraction**
   - Bag-of-Words (CountVectorizer, 1024 features).
3. **Neural Network (MLP)**
   - Predicts gluten presence and substitute class.
4. **Hybrid Refinement**
   - Neural output validated with existing rule-based substitutions for transparency.
5. **Evaluation**
   - Tracks model accuracy and substitution coverage across 2,000 sample recipes.

**Key Upgrade vs. Checkpoint 1:**
- Transitioned from **rule-based lookups** to a **learned, data-driven model** that adapts to unseen text patterns.

## Slide 3: Code Snippet 1 – Data Pairing and Parsing

```{python}
def pair_parts_and_quantities(parts_raw, quants_raw):
    parts = de_r_list(parts_raw)
    quants = de_r_list(quants_raw)
    paired = []
    for i in range(max(len(parts), len(quants))):
        p = parts[i] if i < len(parts) else ""
        q = quants[i] if i < len(quants) else ""
        if q and not is_quantity_token(q):
            paired.append(f"{q} {p}".strip())
        elif q:
            paired.append(f"{q} {p}".strip())
        else:
            paired.append(p.strip())
    return [clean_text(x) for x in paired if x]
```
## Slide 4: Explaining Snippet 1

- Purpose: Converts Food.com R-style data `c("1","2","flour")` into clean, readable ingredient strings.

- How It Works:

    - Extracts both parts (ingredients) and quantities from separate columns.

    - Joins them safely into "1 cup wheat flour" format.

    - Applies Unicode normalization (clean_text) to remove hidden characters.

- Checkpoint 1 vs. Checkpoint 2:

    - CP1 only read static strings like "2 cups of flour".

    - CP2 dynamically builds those strings from structured data, enabling large-scale automation.

## Slide 5: Code Snippet 2- Neural Network Model

```{python}
class GlutenSubstitutionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_substitutes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.flag_head = nn.Linear(hidden_dim, 2)
        self.sub_head = nn.Linear(hidden_dim, num_substitutes)

    def forward(self, x):
        h = self.shared(x)
        return self.flag_head(h), self.sub_head(h)
```
## Slide 6: Explaining Snippet 2

- Structure:

    - A multitask MLP with shared hidden layers.

    - Two heads:

    - flag_head: Predicts if a recipe contains gluten.

    - sub_head: Suggests appropriate substitute class.

- Training:

    - Optimizes dual losses (CrossEntropy for both outputs).

    - Achieved convergence in 6 epochs with minimal loss (~0.002).

- Improvement:

    - CP1: static dictionary lookup.

    - CP2: model learns associations between text patterns and substitutions.

## Slide 7: Results

| Metric                        |      Result      |
| :---------------------------- | :--------------: |
| **Dataset**                   | Food.com Recipes |
| **Training Samples**          |       2000       |
| **Model Parameters**          |      25,991      |
| **Epochs**                    |         6        |
| **Final Loss**                |      ~0.002      |
| **Gluten Flag Accuracy**      |       100%       |
| **Substitute Class Accuracy** |       100%       |
| **Average Rule Coverage**     |      60–70%      |

## Slide 8: Analysis and next Steps

- Findings

    - Successfully merged AI-driven inference with explainable rule validation.

    - Represents a major leap from manual rules → learned, adaptive AI system.

    - Cleaned noisy, R-style dataset into usable ingredient text.

    - Neural classifier effectively mimics rule-based mappings.

- Remaining Limitations

    - Overfitting due to deterministic labels.

    - Current text representation is still an issue, struggling to fix spelling still.

- Planned Next Steps

    - Add semantic embeddings (Word2Vec/BERT) for richer text understanding.

    - Collect real gluten-free recipe pairs to train on non-deterministic targets.

    - Implement model persistence (model.pth) for future inference reuse.

    - Build a front-end for interactive recipe conversion.

