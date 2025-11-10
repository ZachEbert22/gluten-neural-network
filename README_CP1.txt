## The Problem with Gluten-Free Solutions
	- Currently individuals with celiac's disease are suffering from quality recipes.
	- Most Recipes online dont account for gluten-fre individuals
	- The Ones that do dont provide good alternatives, so the tastes degrades severly.
	
	- The goal is to provide a way for gluten-free Individuals with a way to change any recipe into a gluten-free one
	- This can be acomplished by finding the correct alternatives dependind on the original recipe.

## Model Approach
	- Goal is to have to eventually implement a Ingredient Substituion Module that will use something similar to GNN's
	- This will be used to help switch out ingredients that have gluten with their gluten-free alternatives

	- The Initial design for this is several json maps, they are then read in by a parser.
 	- Once they are read and parse, they will be formatted, and then processed
	- Once formatted, a substute function can inject these maps and match via a key dictionary
	- Once matched, It can calculate the substitution total to the rest of the recipe for accuracy and coverage

## Code Snippet 1

```{python}

def substitute_ingredient(parsed_ingredient, substitutions):

    # Substitute gluten ingredients and adjust ratios.
    name = parsed_ingredient["ingredient"].lower()
    quantity = float(parsed_ingredient["quantity"] or 1)
    unit = parsed_ingredient["unit"]

    for gluten_item, details in substitutions.items():
        if gluten_item in name:
            ratio = details.get("ratio", 1.0)
            new_quantity = round(quantity * ratio, 2)
            return {
                "quantity": str(new_quantity),
                "unit": unit,
                "ingredient": details["substitute"]
            }

    # if no substitution needed
    return parsed_ingredient
```

## Substitution Explanation
	- This code currently grabs formatted and parsed indegredient information and substitutes it based off of json maps
	- This function grabs the name, quality and unit intialy and checks the keys in the json files
	- if a match is find, it multiplies the unit by the specified ratio. If there is not ratio it defaults to 1
	
	- This a key part of my project as this function is the initial steps for a large scale network of maps
	- From here, The substition can evolve to include more maps and check for more specific ingredients
	- Eventually switch to a graph-based ingredient substition module for more intelligent substitutions
	- For finding gluten-free solutions, this is the most important function in the code.

## Code Snippet 2

```{python}
def substitution_accuracy(original, modified, gluten_ingredients):

    #Measures fraction of gluten ingredients successfully replaced
    correct_replacements = 0
    total_gluten_items = 0
    for o, m in zip(original, modified):
        if any(g in o.lower() for g in gluten_ingredients):
            total_gluten_items += 1
            if not any(g in m.lower() for g in gluten_ingredients):
                correct_replacements += 1
    if total_gluten_items == 0:
        return 1.0
    return correct_replacements / total_gluten_items

def substitution_coverage(original, modified):

    #Measures fraction of all ingredients changed
    total = len(original)
    replaced = sum(1 for o, m in zip(original, modified) if o != m)
    return replaced / total if total > 0 else 0
```

## Accuracy and Coverage Explanation
	- The First function emasures how well the ingredients were replaced
	- This is done by comparing each ingredient to the newly modified recipe
	- Returns a fraction for how accurate and safe the new recipe is for individuals with celiacs disease
	
	- The Second Function measures how much the recipe was changed in general
	- Compares the newly created recipe to the modified one, and how much it was changed

	- These are important to ensure everything was correctly substituted
	- Coverage will be important to determine how much was changed
	- Changing too much can alos degrade taste, and accuracy is critical in order to not harm others using this model.

## Model Results

	- Below is a Prelimary Result for a basic Recipe that was used as the input
	- The Model Shows Input, Output, Substition Accuracy and Coverage.

--- Ingredient Substitution Results ---
  2 cups of wheat flour     →  1.5 cups of almond flour
  1 cup of sugar            →  1 cup of sugar
  2 eggs                    →  2 egg
---------------------------------------
Substitution Accuracy: 100.00%
Substitution Coverage: 66.67%

Gluten-Free Recipe: ['1.5 cups of almond flour', '1 cup of sugar', '2 egg']

## Analysis and next Steps
	
	### Analysis
	- Shown above is the results for using a basic recipe only containing three ingredients
	- The Model was able to use the json key system to make the correct substition based off the recipe input
	- Provides the Correct, substituion, and gives a percentage for the accuracy and covergae
	
	### Next Steps
	- Short Term: Expand the Json maps, and Try to fix Substition naming conventions.
	- Long Term: Expand the ingredient function to start injesting a graph-based subsitution.
	- Start Testing Edge cases, by providing another key for ingredients that dont explicity say they contain gluten
	  	- Ex: Brownie Batter, Bread Crumbs, etc.
	- Have this map break down into components already included in other graphical-based json files.

## Dependencies

pip install pandas scikit-learn torch kagglehub


	
