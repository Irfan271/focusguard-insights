import json

import joblib
import pandas as pd
import pytest

from src.config import (
    CLASSIFIER_FILE,
    CLEAN_DATA_FILE,
    MODEL_METADATA_FILE,
    PRODUCTIVITY_LABELS,
)
from src.train_model import (
    MODEL_FEATURE_COLUMNS,
    prepare_features,
    rule_based_baseline,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "usage_minutes": [30, 45],
            "sessions": [3, 5],
            "notifications": [10, 20],
            "hour_of_day": [9, 22],
            "study_minutes": [120, 30],
            "sleep_hours": [7.5, 5.0],
            "daily_total_usage_minutes": [180, 320],
            "daily_study_minutes": [180, 45],
            "is_weekend": [0, 1],
            "notifications_per_session": [3.33, 4.0],
            "usage_per_session": [10.0, 9.0],
            "category": ["Study", "Social"],
            "productivity_label": [
                "Productive",
                "Distracted",
            ],
        }
    )


def test_prepare_features_returns_expected_shape(
    sample_dataframe,
):
    X, y = prepare_features(sample_dataframe)

    assert len(X) == 2
    assert len(y) == 2

    assert list(X.columns) == MODEL_FEATURE_COLUMNS

    assert "category" in X.columns
    assert "category_Study" not in X.columns
    assert "category_Social" not in X.columns


def test_prepare_features_preserves_raw_categories(
    sample_dataframe,
):
    X, _ = prepare_features(sample_dataframe)

    assert X.loc[0, "category"] == "Study"
    assert X.loc[1, "category"] == "Social"


def test_rule_based_baseline_returns_target_column(
    sample_dataframe,
):
    baseline = rule_based_baseline(sample_dataframe)

    assert baseline.tolist() == [
        "Productive",
        "Distracted",
    ]


def test_clean_data_file_exists():
    assert CLEAN_DATA_FILE.exists()


def test_trained_model_artifacts_exist():
    assert CLASSIFIER_FILE.exists()


def test_model_metadata_contains_transparency_fields():
    assert MODEL_METADATA_FILE.exists()

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["data_source"] == "synthetic"
    assert (
        "rule_based_baseline_accuracy" in metadata
    )
    assert "future_date_holdout" in metadata
    assert (
        "label leakage"
        in metadata["evaluation_note"].lower()
    )


def test_model_metadata_classes_are_valid():
    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert set(metadata["classes"]).issubset(
        set(PRODUCTIVITY_LABELS)
    )


def test_saved_model_can_make_predictions():
    pipeline = joblib.load(CLASSIFIER_FILE)

    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")