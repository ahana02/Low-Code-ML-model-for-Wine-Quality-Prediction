"""
Step 7: Evaluation.

Reports metrics chosen for an imbalanced binary problem (accuracy alone
is misleading when ~86% of wines are the negative class -- a model that
always predicts "not good" would still score ~86% accuracy). precision/
recall/f1/roc_auc together give a fuller picture of minority-class
("good" wine) performance, plus a confusion matrix and feature
importances for interpretability.
"""

import pandas as pd
from sklearn import metrics as sk_metrics


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": float(sk_metrics.accuracy_score(y_test, y_pred)),
        "precision": float(sk_metrics.precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(sk_metrics.recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(sk_metrics.f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(sk_metrics.roc_auc_score(y_test, y_proba)),
        "confusion_matrix": sk_metrics.confusion_matrix(y_test, y_pred).tolist(),
    }
    return results, y_pred, y_proba


def feature_importances(pipeline, feature_columns) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    importances = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importances
