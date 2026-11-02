"""Evaluate the trained model on the test set.

Input:  data/processed_data/X_test_scaled.csv, y_test.csv, models/gbr_model.pkl
Output: data/predictions.csv (true and predicted silica_concentrate)
        metrics/scores.json (MSE, RMSE, MAE, R2)
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/processed_data")
MODEL_FILE = Path("models/gbr_model.pkl")
PREDICTIONS_FILE = Path("data/predictions.csv")
SCORES_FILE = Path("metrics/scores.json")


def main() -> None:
    X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    pd.DataFrame({"silica_concentrate": y_test, "predicted_silica_concentrate": y_pred}).to_csv(
        PREDICTIONS_FILE, index=False
    )

    mse = mean_squared_error(y_test, y_pred)
    scores = {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=4)

    print(json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
