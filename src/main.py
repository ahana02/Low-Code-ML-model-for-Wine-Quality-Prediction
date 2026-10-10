"""
Orchestrates the full wine-quality pipeline end to end:

  load -> clean -> engineer features -> split -> build pipeline ->
  train -> evaluate -> save artifacts -> demo prediction on new samples

Usage:
    python -m src.main
"""

from . import (
    config,
    data_loader,
    preprocessing,
    feature_engineering,
    data_splitter,
    pipeline_builder,
    train as train_module,
    evaluate as evaluate_module,
    artifacts,
    predict,
)


def run():
    print("[1/8] Loading raw data")
    raw = data_loader.load_raw_data()
    print(f"       -> {data_loader.load_summary(raw)}")

    print("[2/8] Preprocessing (dedupe, impute, outlier capping)")
    cleaned = preprocessing.clean(raw)

    print("[3/8] Target Creation")
    engineered = feature_engineering.engineer_features(cleaned)
    feature_cols = feature_engineering.get_feature_columns(engineered)
    print(f"       -> {len(feature_cols)} features: {feature_cols}")

    print("[4/8] Train/test split (stratified on quality bucket)")
    X_train, X_test, y_train, y_test = data_splitter.split(engineered)
    print(f"       -> train={X_train.shape[0]} test={X_test.shape[0]}")
    print(f"       -> train positive rate: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    print("[5/8] Building pipeline (StandardScaler -> RandomForestClassifier)")
    pipeline = pipeline_builder.build_pipeline(feature_cols)

    print("[6/8] Training (+ 5-fold CV sanity check on train split)")
    train_module.train(pipeline, X_train, y_train)
    cv_results = train_module.cross_validate(pipeline, X_train, y_train)
    print(f"       -> CV f1: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")

    print("[7/8] Evaluating on held-out test set")
    metrics, y_pred, y_proba = evaluate_module.evaluate(pipeline, X_test, y_test)
    for k, v in metrics.items():
        print(f"       {k}: {v}")

    importances = evaluate_module.feature_importances(pipeline, feature_cols)
    print("       top 5 features by importance:")
    print(importances.head(5).to_string(index=False))

    print("[8/8] Saving artifacts")
    artifacts.save_all(pipeline, metrics, X_test, y_test, y_pred, importances)
    print(f"       -> written to {config.OUTPUT_DIR}")

    return pipeline, metrics


if __name__ == "__main__":
    run()
