import pandas as pd

from src.config import (
    CLEAN_DATA_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw app-usage dataset.
    """
    df = pd.read_csv(RAW_DATA_FILE)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset and prepare it for analysis and ML.
    """
    df = df.copy()

    # Convert date column to datetime.
    df["date"] = pd.to_datetime(df["date"])

    # Remove exact duplicate rows.
    df = df.drop_duplicates()

    # Remove rows with missing critical values.
    critical_columns = [
        "date",
        "user_id",
        "app_name",
        "category",
        "usage_minutes",
        "study_minutes",
        TARGET_COLUMN,
    ]
    df = df.dropna(subset=critical_columns)

    # Remove invalid values.
    df = df[df["usage_minutes"] >= 0]
    df = df[df["sessions"] >= 0]
    df = df[df["notifications"] >= 0]
    df = df[df["study_minutes"] >= 0]
    df = df[df["sleep_hours"] > 0]

    # Keep only valid productivity labels.
    valid_labels = ["Productive", "Moderate", "Distracted"]
    df = df[df[TARGET_COLUMN].isin(valid_labels)]

    # Create date-based features.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    # Create useful ratio features.
    df["notifications_per_session"] = (
        df["notifications"] / df["sessions"].replace(0, 1)
    ).round(2)

    df["usage_per_session"] = (
        df["usage_minutes"] / df["sessions"].replace(0, 1)
    ).round(2)

    # Sort data for readability.
    df = df.sort_values(
        by=["user_id", "date", "app_name"]
    ).reset_index(drop=True)

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned dataset.
    """
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_FILE, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    save_clean_data(cleaned_df)

    print(f"Cleaned dataset saved: {CLEAN_DATA_FILE}")
    print(f"Original rows: {len(raw_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(cleaned_df.head())