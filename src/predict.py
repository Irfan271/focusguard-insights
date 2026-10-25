import joblib
import pandas as pd

from src.config import (
    CLASSIFIER_FILE,
    MODEL_DIR,
    TARGET_COLUMN,
)


def load_model_artifacts():
    """
    Load the trained classifier, scaler, and feature columns.
    """
    model = joblib.load(CLASSIFIER_FILE)
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

    return model, scaler, feature_columns


def prepare_input_features(record: dict, feature_columns: list) -> pd.DataFrame:
    """
    Convert one input record into the exact feature format used during training.
    """
    input_df = pd.DataFrame([record])

    # Add missing one-hot category columns.
    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    # Keep columns in the same order as training.
    input_df = input_df[feature_columns]

    return input_df


def predict_productivity(record: dict) -> dict:
    """
    Predict the productivity label for one app-usage record.
    """
    model, scaler, feature_columns = load_model_artifacts()

    input_df = prepare_input_features(record, feature_columns)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    probability_by_class = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return {
        TARGET_COLUMN: prediction,
        "probabilities": probability_by_class,
    }


if __name__ == "__main__":
    sample_record = {
        "usage_minutes": 120,
        "sessions": 14,
        "notifications": 45,
        "hour_of_day": 22,
        "study_minutes": 35,
        "sleep_hours": 6.0,
        "daily_total_usage_minutes": 420,
        "daily_study_minutes": 45,
        "is_weekend": 1,
        "notifications_per_session": 3.21,
        "usage_per_session": 8.57,
        "category_Entertainment": 1,
        "category_Learning": 0,
        "category_Productivity": 0,
        "category_Social": 0,
        "category_Utility": 0,
    }

    result = predict_productivity(sample_record)

    print("Prediction result:")
    print(result)