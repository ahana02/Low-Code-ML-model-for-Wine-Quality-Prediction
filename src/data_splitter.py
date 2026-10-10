"""
Step 4: Train/test split.

Stratifies on the coarse `quality_bucket` (not raw quality, and not the
binary target either) so both the minority "good" wines and the
low-quality tail stay proportionally represented in train and test --
splitting on the binary target alone would still let a train split get
unlucky on the extreme quality=3/quality=8 tails within each class.
"""

from sklearn.model_selection import train_test_split

from . import config, feature_engineering


def split(df):
    """Returns X_train, X_test, y_train, y_test (y = binary target)."""
    feature_cols = feature_engineering.get_feature_columns(df)
    X = df[feature_cols]
    y = df[config.BINARY_TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["quality_bucket"],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from . import data_loader, preprocessing

    raw = data_loader.load_raw_data()
    cleaned = preprocessing.clean(raw)
    engineered = feature_engineering.engineer_features(cleaned)
    X_train, X_test, y_train, y_test = split(engineered)
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print("train positive rate:", y_train.mean().round(3))
    print("test  positive rate:", y_test.mean().round(3))
