"""
Step 2: Preprocessing / data cleaning.

Handles the data-quality issues actually present in this dataset:
  - 240 exact duplicate rows (verified via df.duplicated().sum())
  - no missing values today, but a SimpleImputer is still wired in so the
    pipeline doesn't silently break if a future data pull has gaps
  - long right tails in a handful of chemistry columns (residual sugar,
    chlorides, free/total sulfur dioxide, sulphates) that would otherwise
    let a handful of extreme wines dominate distance/variance-based models

Outliers are capped (winsorized) rather than dropped: with only 1599 rows
to begin with, deleting rows for a small dataset throws away signal for
little benefit, and capping still neutralizes the distortion.
"""

import pandas as pd
from sklearn.impute import SimpleImputer

from . import config


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"       -> removed {removed} exact duplicate rows")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """No-op today (dataset has 0 missing values) but keeps the pipeline
    robust if a future data source has gaps in numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != config.TARGET_COLUMN]

    if df[numeric_cols].isna().sum().sum() == 0:
        return df

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns=None, multiplier=None) -> pd.DataFrame:
    """Winsorize the given columns to [Q1 - k*IQR, Q3 + k*IQR]."""
    columns = columns or config.SKEWED_COLUMNS
    multiplier = multiplier or config.IQR_MULTIPLIER

    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped:
            print(f"       -> capped {n_capped} outliers in '{col}'")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence."""
    df = drop_duplicates(df)
    df = impute_missing(df)
    df = cap_outliers_iqr(df)
    return df


if __name__ == "__main__":
    from . import data_loader

    raw = data_loader.load_raw_data()
    cleaned = clean(raw)
    print(f"raw shape: {raw.shape}  cleaned shape: {cleaned.shape}")
