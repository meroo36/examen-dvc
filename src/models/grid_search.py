"""Find the best hyperparameters of a GradientBoostingRegressor with GridSearchCV.

Input:  data/processed_data/X_train_scaled.csv, y_train.csv, params.yaml (grid_search)
Output: models/best_params.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

DATA_DIR = Path("data/processed_data")
BEST_PARAMS_FILE = Path("models/best_params.pkl")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    grid_params = params["grid_search"]

    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=params["train"]["random_state"]),
        param_grid=grid_params["param_grid"],
        cv=grid_params["cv"],
        scoring=grid_params["scoring"],
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_FILE, "wb") as f:
        pickle.dump(grid.best_params_, f)

    print(f"Best params: {grid.best_params_} (CV {grid_params['scoring']}: {grid.best_score_:.4f})")


if __name__ == "__main__":
    main()
