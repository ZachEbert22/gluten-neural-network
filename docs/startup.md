## How to Run

pip install torch transformers datasets pandas streamlit scikit-learn beautifulsoup4 requests

python build_ingredient_dataset.py --max-pos 30000 --max-neg 30000
python3 train_ingredient_classifier.py

python -m uvicorn unified_api:app --reload --port 8000
streamlit run frontend.py

Websites to Try:
    https://www.bbcgoodfood.com/recipes/fruit-spice-soda-bread

    https://tastesbetterfromscratch.com/classic-french-toast/

    https://www.loveandlemons.com/fettuccine-alfredo-recipe/

    https://www.kingarthurbaking.com/recipes/creamy-tomato-soup-recipe

    https://www.loveandlemons.com/brownies-recipe/

    https://www.loveandlemons.com/french-toast/

2.75 cups of all-purpose flour
2 teaspoons of cornstarch
1.25 teaspoons of kosher salt
1 teaspoons of baking soda
.75 cups of butter
1.25 cups of granulated sugar
1 large egg, lightly beaten
2 teaspoons of vanilla extract
1 cup of well-mashed very ripe banana (from about 2 1/2 large bananas
1.75 cups dark and milk chocolate chunks (or chopped chocolate bars)
2 Cups of wholemeal Flour
3 loafs of Bread

It supresses Irrelevant Regions, How does the Model Decide whats Irrelevant

What Made you decide to not use a transformer for the 1024 Bottleneck?


