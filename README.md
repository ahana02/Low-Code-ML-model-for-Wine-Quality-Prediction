# Wine Quality ML Pipeline

An ML pipeline for the [Wine Quality (Red) dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data),
built one file per pipeline step so each stage (loading, cleaning,
feature engineering, training, evaluation, inference) can be read,
tested, and modified independently.

> **Note on the data file:** the sandbox this was built in can't reach
> kaggle.com, so `data/winequality-red.csv` here is pulled from a public
> mirror of the original UCI dataset -- **identical 12 columns** to the
> Kaggle version (Kaggle just adds a leading `Id` column, which
> `data_loader.py` already detects and drops). Drop the actual Kaggle
> CSV into `data/winequality-red.csv` and everything below runs unchanged.

## Quick start

```bash
pip install -r requirements.txt
python -m src.main          # runs the full pipeline, prints progress + metrics
python -m src.predict        # scores two example wines with the saved model
```

## Pipeline steps (one file each)

| # | File | What it does |
|---|------|---------------|
| 1 | `src/data_loader.py` | Reads the CSV, drops Kaggle's `Id` column if present, validates expected columns |
| 2 | `src/preprocessing.py` | Drops 240 exact duplicate rows, imputes any missing values (none today, wired in for robustness), IQR-caps outliers in 5 skewed columns |
| 3 | `src/feature_engineering.py` | Buckets quality for stratified splitting, creates the binary training target |
| 4 | `src/data_splitter.py` | 80/20 train/test split, stratified on the quality bucket |
| 5 | `src/pipeline_builder.py` | Builds the sklearn `Pipeline` (StandardScaler → RandomForestClassifier) |
| 6 | `src/train.py` | Fits the pipeline; optional 5-fold CV sanity check |
| 7 | `src/evaluate.py` | Computes accuracy/precision/recall/F1/ROC-AUC + confusion matrix + feature importances |
| 8 | `src/artifacts.py` | Saves `model.joblib`, `metrics.json`, `predictions.csv`, `feature_importance.csv` |
| 9 | `src/predict.py` | Loads the saved model and scores new, unseen wine samples |
| — | `src/config.py` | Every path/threshold/hyperparameter in one place |
| — | `src/main.py` | Orchestrates steps 1–8 in order |

## Dataset-specific design decisions

**Target: binary, not raw ordinal quality.** Raw `quality` is 3–8 with
almost no support at the extremes (10 wines at quality=3, 18 at
quality=8) — too sparse to model as a 6-way classification or reliably
as regression. `config.GOOD_QUALITY_THRESHOLD = 7` reduces it to "is
this a good wine?", which is both the standard framing for this dataset
and the more decision-useful question for a winery. Change the threshold
in one place (`config.py`) to make "good" stricter/looser.

**Preprocessing found and fixed real issues in this data:**
- 240 of 1599 rows (15%) are exact duplicates — dropped.
- `residual sugar`, `chlorides`, `free/total sulfur dioxide`, and
  `sulphates` all have long right tails (verified via `.describe()` —
  max values several IQRs above the 75th percentile). These are capped
  (winsorized) rather than dropped, since the dataset is small enough
  that discarding rows loses more signal than it's worth.

**Stratify on a quality bucket, not the binary target directly.**
`quality_bucket` (low/medium/high, `config.QUALITY_BUCKETS`) groups raw
quality into 3 bins purely for `train_test_split(..., stratify=...)` —
stratifying on the binary target alone would still let the extreme
quality=3 and quality=8 tails land unevenly within each class.

**Model: RandomForestClassifier with `class_weight='balanced'`.**
Only ~14% of wines are "good," so class weighting is used instead of a
separate oversampling step (SMOTE, etc.) for this baseline. A tree
ensemble was chosen over linear models because wine-quality chemistry
interacts nonlinearly (e.g. volatile acidity's effect depends on
alcohol/sulphate levels) — trees capture that without hand-built
interaction terms for every pair.

**Metrics beyond accuracy.** With an 86/14 class split, a model that
always predicts "not good" scores ~86% accuracy while being useless.
Precision/recall/F1/ROC-AUC are reported together so minority-class
("good" wine) performance is visible, not masked by the majority class.

## Current results (on the fetched data, your numbers may vary slightly by
environment/library version)

```
accuracy:  0.868
precision: 0.512
recall:    0.568
f1:        0.538
roc_auc:   0.887
```

Recall/precision on the minority class have real room to improve —
natural next steps: hyperparameter search (grid/random search over
`n_estimators`/`max_depth`), trying gradient boosting, or explicit
resampling (SMOTE) instead of only class weighting. The one-file-per-step
structure means any of these is a change to a single file
(`pipeline_builder.py` or `train.py`) without touching the rest.

## Scoring new wines

```python
from src import predict
import pandas as pd

model = predict.load_model()
new_wine = pd.DataFrame([{
    "fixed acidity": 9.0, "volatile acidity": 0.3, "citric acid": 0.45,
    "residual sugar": 2.1, "chlorides": 0.06, "free sulfur dioxide": 12,
    "total sulfur dioxide": 30, "density": 0.9958, "pH": 3.3,
    "sulphates": 0.85, "alcohol": 12.5,
}])
print(predict.predict_new_samples(model, new_wine))
```
