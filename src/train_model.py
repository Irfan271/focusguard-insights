import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    CLASSIFIER_FILE,
    CLEAN_DATA_FILE,
    MODEL_METADATA_FILE,
    MODEL_DIR,
    PRODUCTIVITY_LABELS,
    TARGET_COLUMN,
)


NUMERIC_FEATURES = [
    "usage_minutes",
    "sessions",
    "notifications",
    "hour_of_day",
    "study_minutes",
    "sleep_hours",
    "daily_total_usage_minutes",
    "daily_study_minutes",
    "is_weekend",
    "notifications_per_session",
    "usage_per_session",
]

CATEGORICAL_FEATURES = [
    "category",
]

MODEL_FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    return pd.read_csv(CLEAN_DATA_FILE)


def prepare_features(df: pd.DataFrame):
    """
    Create raw model features and the target.

    The category column remains in its original form. Scaling and one-hot
    encoding are handled by the saved sklearn Pipeline.
    """
    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN]

    return X, y


def rule_based_baseline(df: pd.DataFrame) -> pd.Series:
    """Reproduce the rule used to create productivity_label."""
    return df[TARGET_COLUMN]


def build_pipeline() -> Pipeline:
    """
    Build the complete preprocessing and classification pipeline.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def evaluate_future_date_holdout(df: pd.DataFrame):
    """Train on earlier dates and evaluate on later dates."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = sorted(df["date"].unique())
    cutoff_index = int(len(dates) * 0.8)

    training_dates = dates[:cutoff_index]
    testing_dates = dates[cutoff_index:]

    training_mask = df["date"].isin(training_dates)
    testing_mask = df["date"].isin(testing_dates)

    X, y = prepare_features(df)

    X_train = X[training_mask]
    X_test = X[testing_mask]

    y_train = y[training_mask]
    y_test = y[testing_mask]

    missing_training_classes = set(y.unique()) - set(
        y_train.unique()
    )

    if missing_training_classes:
        raise ValueError(
            "Future-date training set is missing these classes: "
            f"{sorted(missing_training_classes)}"
        )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\nFuture-date holdout evaluation")
    print(f"Training dates: {len(training_dates)}")
    print(f"Testing dates: {len(testing_dates)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=PRODUCTIVITY_LABELS,
            zero_division=0,
        )
    )

    return {
        "training_dates": len(training_dates),
        "testing_dates": len(testing_dates),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "accuracy": round(float(accuracy), 4),
        "weighted_f1_score": round(float(weighted_f1), 4),
    }


def train_productivity_classifier():
    """Train and evaluate using a user-day grouped split."""
    df = load_clean_data()

    X, y = prepare_features(df)

    baseline_predictions = rule_based_baseline(df)
    baseline_accuracy = accuracy_score(
        y,
        baseline_predictions,
    )

    print(
        f"Rule-based baseline accuracy: {baseline_accuracy:.4f}"
    )

    future_date_results = evaluate_future_date_holdout(df)

    # All app records for the same user and date share one group.
    groups = (
        df["user_id"].astype(str)
        + "_"
        + pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlapping_groups = train_groups.intersection(
        test_groups
    )

    if overlapping_groups:
        raise ValueError(
            "User-day overlap detected between training and testing."
        )

    missing_training_classes = set(y.unique()) - set(
        y_train.unique()
    )

    if missing_training_classes:
        raise ValueError(
            "Training set is missing these classes: "
            f"{sorted(missing_training_classes)}"
        )

    print("\nUser-day grouped split evaluation")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Training user-days: {len(train_groups)}")
    print(f"Testing user-days: {len(test_groups)}")
    print(f"Overlapping user-days: {len(overlapping_groups)}")

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=PRODUCTIVITY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=PRODUCTIVITY_LABELS,
    )

    print("\nModel training complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=PRODUCTIVITY_LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Rows = actual labels; columns = predicted labels")
    print(f"Label order: {PRODUCTIVITY_LABELS}")
    print(matrix)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, CLASSIFIER_FILE)
    joblib.dump(
        MODEL_FEATURE_COLUMNS,
        MODEL_DIR / "feature_columns.joblib",
    )

    metadata = {
        "model_type": "RandomForestClassifier",
        "pipeline": (
            "ColumnTransformer with StandardScaler and OneHotEncoder, "
            "followed by RandomForestClassifier"
        ),
        "target_column": TARGET_COLUMN,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": pipeline.named_steps[
            "classifier"
        ].classes_.tolist(),
        "accuracy": round(float(accuracy), 4),
        "weighted_f1_score": round(float(weighted_f1), 4),
        "rule_based_baseline_accuracy": round(
            float(baseline_accuracy),
            4,
        ),
        "split_method": "GroupShuffleSplit",
        "grouping_columns": ["user_id", "date"],
        "test_group_fraction": 0.2,
        "random_state": 42,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "training_user_days": len(train_groups),
        "testing_user_days": len(test_groups),
        "overlapping_user_days": len(overlapping_groups),
        "future_date_holdout": future_date_results,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_label_order": PRODUCTIVITY_LABELS,
        "data_source": "synthetic",
        "evaluation_note": (
            "User-day groups are separated between training and testing, "
            "and a future-date holdout is also evaluated. However, "
            "productivity_label is generated from rules based on the same "
            "features used for prediction, so the high accuracy reflects "
            "label leakage, not real-world predictive performance. "
            "This model should be treated as a demonstration of the pipeline."
        ),
        "trained_at": datetime.now().astimezone().isoformat(),
    }

    with open(
        MODEL_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel saved: {CLASSIFIER_FILE}")
    print(f"Metadata saved: {MODEL_METADATA_FILE}")

    return pipeline


if __name__ == "__main__":
    train_productivity_classifier()