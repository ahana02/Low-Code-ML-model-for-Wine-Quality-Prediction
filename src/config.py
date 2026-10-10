"""
Central configuration for the wine quality pipeline.

Keeping every tunable in one place means each pipeline-step file
(preprocessing, feature_engineering, pipeline_builder, ...) stays free
of magic numbers and can be tuned from a single spot.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "winequality-red.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# The raw label is an ordinal score (3-8) with very few samples at the
# extremes (10 wines at quality=3, 18 at quality=8). Kaggle/UCI community
# convention -- and the more decision-useful framing for a winery -- is to
# reduce this to a binary "is this a good wine?" classification problem.
# GOOD_QUALITY_THRESHOLD is the cut point; raise/lower it to change how
# strict "good" means.
GOOD_QUALITY_THRESHOLD = 7
TARGET_COLUMN = "quality"
BINARY_TARGET_COLUMN = "is_good_quality"

# Quality is also bucketed into 3 coarse bins purely so train_test_split can
# stratify on something with enough support in every split (stratifying
# directly on 6 raw quality values leaves buckets like quality=3 with only
# 10 total rows, too few to split reliably).
QUALITY_BUCKETS = {
    "low": (0, 4),    # quality 3-4
    "medium": (5, 6),  # quality 5-6
    "high": (7, 10),  # quality 7-8
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Columns with heavy right-skew / long tails in this dataset (verified via
# df.describe(): max values several std-devs above the 75th percentile).
# These get IQR-based outlier capping rather than removal, so we don't
# throw away rows for a dataset this small (1599 rows).
SKEWED_COLUMNS = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "sulphates",
]
IQR_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "class_weight": "balanced",  # counteracts the ~86/14 class imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
