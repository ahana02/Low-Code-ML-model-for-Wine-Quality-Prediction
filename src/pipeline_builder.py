"""
Step 5: Model pipeline construction.

All 11 features here are numeric, so the
ColumnTransformer is simple -- but it's still built explicitly (rather
than scaling the whole DataFrame directly) so adding a categorical
feature later is a one-line change instead of a rewrite.

Model choice: RandomForestClassifier.
  - Wine chemistry features interact nonlinearly (e.g. the effect of
    volatile acidity on perceived quality depends on sulphate/alcohol
    levels) -- a tree ensemble captures that without manual interaction
    terms for every pair.
  - Robust to the differing scales/skew across features, so it's a
    reasonable baseline even before scaling is perfect.
  - class_weight='balanced' (set in config.MODEL_PARAMS) directly
    addresses the ~86/14 class imbalance without needing a separate
    resampling step (SMOTE etc.) for a first-pass baseline.
  - Gives free feature_importances_ for the evaluation step.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def build_preprocessor(feature_columns) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), feature_columns),
        ],
        remainder="drop",
    )


def build_pipeline(feature_columns) -> Pipeline:
    preprocessor = build_preprocessor(feature_columns)
    model = RandomForestClassifier(**config.MODEL_PARAMS)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
