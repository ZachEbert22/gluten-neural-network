# statistics_pipeline.py
import time
import json
from models.ingredient_classifier.predict import predict

INGREDIENT_FILE = "../data/ingredients.json"

def main():
    print("\n===== PIPELINE PERFORMANCE =====\n")

    with open(INGREDIENT_FILE, "r") as fp:
        ingredients = json.load(fp)

    times = []
    counts = {"gluten": 0, "gluten_free": 0, "other": 0}

    for ing in ingredients:
        t0 = time.time()
        label = predict(ing)
        t1 = time.time()

        times.append(t1 - t0)

        if label in counts:
            counts[label] += 1
        else:
            counts["other"] += 1

    avg_t = sum(times) / len(times)

    print(f"Total ingredients: {len(ingredients)}")
    print(f"Avg prediction time: {avg_t:.5f} seconds")
    print("Label counts:", counts)

    stats = {
        "num_ingredients": len(ingredients),
        "avg_prediction_time": avg_t,
        "label_counts": counts
    }

    with open("stats_pipeline.json", "w") as fp:
        json.dump(stats, fp, indent=4)

    print("\nSaved → stats_pipeline.json")

if __name__ == "__main__":
    main()

