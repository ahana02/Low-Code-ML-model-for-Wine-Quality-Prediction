"""
Step 8: Inference on new, unseen wine samples.

Loads the persisted pipeline and applies it to raw measurements coming
in as a dict/DataFrame of the 11 original chemistry columns -- the same
feature-engineering step used at training time is re-applied here so a
caller only ever has to supply the raw lab measurements, never the
derived features.
"""

import joblib
import pandas as pd

from . import config


def load_model(path=None):
    path = path or config.MODEL_PATH
    return joblib.load(path)


def predict_new_samples(pipeline, raw_samples: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in raw_samples.columns if c != config.TARGET_COLUMN]

    predictions = pipeline.predict(raw_samples[feature_cols])
    probabilities = pipeline.predict_proba(raw_samples[feature_cols])[:, 1]

    result = raw_samples.copy()
    result["predicted_is_good_quality"] = predictions
    result["predicted_probability_good"] = probabilities.round(4)
    return result


if __name__ == "__main__":
    # A couple of illustrative samples: one chemically close to a typical
    # high-quality wine in this dataset, one close to a typical low one.
    sample_wines = pd.DataFrame(
        [
            {
                "fixed acidity": 9.0,
                "volatile acidity": 0.3,
                "citric acid": 0.45,
                "residual sugar": 2.1,
                "chlorides": 0.06,
                "free sulfur dioxide": 12,
                "total sulfur dioxide": 30,
                "density": 0.9958,
                "pH": 3.3,
                "sulphates": 0.85,
                "alcohol": 12.5,
            },
            {
                "fixed acidity": 7.2,
                "volatile acidity": 0.85,
                "citric acid": 0.02,
                "residual sugar": 2.0,
                "chlorides": 0.09,
                "free sulfur dioxide": 20,
                "total sulfur dioxide": 90,
                "density": 0.9975,
                "pH": 3.4,
                "sulphates": 0.5,
                "alcohol": 9.2,
            },
        ]
    )

    model = load_model()
    print(predict_new_samples(model, sample_wines))
