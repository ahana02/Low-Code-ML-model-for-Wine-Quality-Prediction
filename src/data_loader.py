"""
Step 1: Data loading.

Reads the raw wine quality CSV and does the minimal, non-destructive
sanity checks (right shape/columns, no silent type coercion). All actual
cleaning decisions live in preprocessing.py -- this file's only job is to
get the raw data into a DataFrame reliably.
"""

import pandas as pd

from . import config


EXPECTED_COLUMNS = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
}


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the wine quality CSV.

    Handles both the plain UCI file (12 columns) and the Kaggle
    'yasserh/wine-quality-dataset' variant, which adds a leading `Id`
    column -- that column carries no predictive signal so it's dropped
    here rather than being treated as a feature.
    """
    path = path or config.DATA_PATH
    df = pd.read_csv(path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Loaded data is missing expected wine-quality columns: {missing_cols}"
        )

    return df


def load_summary(df: pd.DataFrame) -> dict:
    """Quick structural summary, useful for logging at pipeline start."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_missing_values": int(df.isna().sum().sum()),
        "quality_value_counts": df["quality"].value_counts().sort_index().to_dict(),
    }


if __name__ == "__main__":
    data = load_raw_data()
    print(load_summary(data))
