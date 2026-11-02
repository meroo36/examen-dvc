"""Normalize the features with a StandardScaler fitted on the training set only.

Input:  data/processed_data/X_train.csv, X_test.csv
Output: data/processed_data/X_train_scaled.csv, X_test_scaled.csv
        models/scaler.pkl (to apply the same scaling to new data)
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed_data")
SCALER_FILE = Path("models/scaler.pkl")


def main() -> None:
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    # Fit on train only so no information from the test set leaks into training.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train_scaled.to_csv(DATA_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(DATA_DIR / "X_test_scaled.csv", index=False)

    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaled {X_train_scaled.shape[1]} features for {len(X_train_scaled)} train / {len(X_test_scaled)} test rows")


if __name__ == "__main__":
    main()
