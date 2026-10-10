"""
Step 6: Training.

Thin wrapper around pipeline.fit -- kept as its own file so cross-
validation, early stopping, or hyperparameter search can be added here
later without touching pipeline construction or evaluation code.
"""

from sklearn.model_selection import cross_val_score

from . import config


def train(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate(pipeline, X_train, y_train, cv=5, scoring="f1"):
    """Optional sanity check: cross-val score on the training split only,
    to catch overfitting before ever touching the held-out test set."""
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    return {
        "cv_scoring": scoring,
        "cv_folds": cv,
        "cv_scores": scores.tolist(),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
    }
