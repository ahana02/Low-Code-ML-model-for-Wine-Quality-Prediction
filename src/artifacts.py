"""
Step 9: Persist run artifacts (model, metrics, predictions, feature
importances) so results are reproducible and inspectable after the run.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from . import config


def save_all(pipeline, metrics: dict, X_test, y_test, y_pred, importances: pd.DataFrame):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, config.MODEL_PATH)

    metrics_payload = {
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    preds_df = X_test.reset_index(drop=True).copy()
    preds_df["y_true"] = pd.Series(y_test).reset_index(drop=True)
    preds_df["y_pred"] = y_pred
    preds_df.to_csv(config.PREDICTIONS_PATH, index=False)

    importances.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
