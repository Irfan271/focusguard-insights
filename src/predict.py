import joblib
import pandas as pd

from src.config import (
    CLASSIFIER_FILE,
    TARGET_COLUMN,
)


def load_model_artifacts():
    """
    Load the complete trained prediction pipeline.

    The pipeline contains numeric scaling, categorical one-hot encoding,
    and the Random Forest classifier.
    """
    return joblib.load(CLASSIFIER_FILE)


def prepare_input_features(record: dict) -> pd.DataFrame:
    """
    Convert one raw input record into a one-row DataFrame.

    The record must contain raw values, including the original category
    such as "Social", "Learning", or "Entertainment". The saved pipeline
    handles scaling and category encoding.
    """
    return pd.DataFrame([record])


def predict_productivity(record: dict) -> dict:
    """
    Predict the productivity label for one app-usage record.
    """
    pipeline = load_model_artifacts()

    input_df = prepare_input_features(record)

    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    classifier = pipeline.named_steps["classifier"]

    probability_by_class = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(
            classifier.classes_,
            probabilities,
        )
    }

    return {
        TARGET_COLUMN: prediction,
        "probabilities": probability_by_class,
    }


if __name__ == "__main__":
    sample_record = {
    "usage_minutes": 25,
    "sessions": 4,
    "notifications": 8,
    "hour_of_day": 10,
    "study_minutes": 120,
    "sleep_hours": 7.5,
    "daily_total_usage_minutes": 180,
    "daily_study_minutes": 180,
    "is_weekend": 0,
    "notifications_per_session": 2.0,
    "usage_per_session": 6.25,
    "category": "Learning",
    }

    result = predict_productivity(sample_record)

    print("Prediction result:")
    print(result)