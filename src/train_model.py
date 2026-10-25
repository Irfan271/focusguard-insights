import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSIFIER_FILE,
    CLEAN_DATA_FILE,
    MODEL_METADATA_FILE,
    MODEL_DIR,
    TARGET_COLUMN,
)


FEATURE_COLUMNS = [
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

CATEGORY_COLUMN = "category"


def load_clean_data() -> pd.DataFrame:
    """
    Load the cleaned dataset.
    """
    return pd.read_csv(CLEAN_DATA_FILE)


def prepare_features(df: pd.DataFrame):
    """
    Create the feature matrix and target variable.
    """
    df = df.copy()

    # Convert categorical app category into numeric columns.
    category_dummies = pd.get_dummies(
        df[CATEGORY_COLUMN],
        prefix="category",
    )

    X = pd.concat(
        [df[FEATURE_COLUMNS], category_dummies],
        axis=1,
    )

    y = df[TARGET_COLUMN]

    return X, y


def train_productivity_classifier():
    """
    Train a Random Forest classifier to predict productivity labels.
    """
    df = load_clean_data()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Model training complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model artifacts.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, CLASSIFIER_FILE)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "target_column": TARGET_COLUMN,
        "feature_columns": list(X.columns),
        "accuracy": round(float(accuracy), 4),
        "weighted_f1_score": round(float(f1), 4),
        "trained_at": datetime.now().isoformat(),
    }

    with open(MODEL_METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel saved: {CLASSIFIER_FILE}")
    print(f"Metadata saved: {MODEL_METADATA_FILE}")

    return model, scaler, X.columns.tolist()


if __name__ == "__main__":
    train_productivity_classifier()