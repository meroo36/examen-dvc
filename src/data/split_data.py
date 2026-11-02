"""Split the raw dataset into train and test sets.

Input:  data/raw_data/raw.csv
Output: data/processed_data/X_train.csv, X_test.csv, y_train.csv, y_test.csv

The target `silica_concentrate` is the last column. The `date` column is only a
timestamp, not an operational parameter, so it is dropped from the features.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_FILE = Path("data/raw_data/raw.csv")
OUTPUT_DIR = Path("data/processed_data")
TARGET = "silica_concentrate"


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["split"]

    df = pd.read_csv(RAW_FILE)
    X = df.drop(columns=[TARGET, "date"], errors="ignore")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items():
        data.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows, features: {list(X.columns)}")


if __name__ == "__main__":
    main()
