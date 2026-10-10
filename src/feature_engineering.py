"""
Step 3: Feature engineering.


  (a) turn the ordinal `quality` score into the binary target the model
      actually trains on, plus a coarse bucket used only for stratifying
      the train/test split

"""

import numpy as np
import pandas as pd

from . import config

EPSILON = 1e-6


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_acidity"] = (
        df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    )

    df["free_so2_ratio"] = df["free sulfur dioxide"] / (
        df["total sulfur dioxide"] + EPSILON
    )

    df["acid_to_sulphate_ratio"] = df["total_acidity"] / (df["sulphates"] + EPSILON)

    df["alcohol_density_interaction"] = df["alcohol"] * df["density"]

    df["sugar_to_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + EPSILON)

    return df


def add_quality_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse 3-way bucket (low/medium/high) used only so
    train_test_split(..., stratify=...) has enough support per class --
    stratifying on raw quality leaves the quality=3 bucket with only 10
    rows total, too thin to stratify a train/test split on directly."""
    df = df.copy()

    def bucket(q):
        for label, (lo, hi) in config.QUALITY_BUCKETS.items():
            if lo <= q <= hi:
                return label
        return "medium"

    df["quality_bucket"] = df[config.TARGET_COLUMN].apply(bucket)
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """quality >= GOOD_QUALITY_THRESHOLD -> 1 ('good' wine), else 0."""
    df = df.copy()
    df[config.BINARY_TARGET_COLUMN] = (
        df[config.TARGET_COLUMN] >= config.GOOD_QUALITY_THRESHOLD
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering sequence."""
    # df = add_domain_features(df)
    df = add_quality_bucket(df)
    df = add_binary_target(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Everything except the raw/derived target columns is a model feature."""
    exclude = {
        config.TARGET_COLUMN,
        config.BINARY_TARGET_COLUMN,
        "quality_bucket",
    }
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from . import data_loader, preprocessing

    raw = data_loader.load_raw_data()
    cleaned = preprocessing.clean(raw)
    engineered = engineer_features(cleaned)
    print("New columns added:", set(engineered.columns) - set(cleaned.columns))
    print(engineered[["quality", "quality_bucket", config.BINARY_TARGET_COLUMN]].head())
