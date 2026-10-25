import numpy as np
import pandas as pd


from src.config import (
    APPS,
    APP_CATEGORIES,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


HIGH_USAGE_APPS = [
    "Instagram",
    "YouTube",
]

MEDIUM_USAGE_APPS = [
    "WhatsApp",
    "Chrome",
    "Netflix",
    "Spotify",
    "Twitch",
]


def generate_app_usage_data(
    start_date: str = "2026-08-01",
    days: int = 90,
    users: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic daily app-usage dataset.


    Each row represents one user's usage of one app on one date.
    """
    rng = np.random.default_rng(seed)


    dates = pd.date_range(start=start_date, periods=days, freq="D")
    rows = []


    for user_id in range(1, users + 1):
        user_id_text = f"U{user_id:03d}"


        for date in dates:
            day_of_week = date.day_name()


            # Weekend usage tends to be higher.
            weekend_factor = (
                1.25
                if day_of_week in ["Saturday", "Sunday"]
                else 1.0
            )


            for app in APPS:
                category = APP_CATEGORIES[app]


                if app in HIGH_USAGE_APPS:
                    base_minutes = rng.integers(40, 180)
                elif app in MEDIUM_USAGE_APPS:
                    base_minutes = rng.integers(15, 90)
                else:
                    base_minutes = rng.integers(5, 60)


                usage_minutes = int(base_minutes * weekend_factor)


                sessions = max(
                    1,
                    int(usage_minutes / rng.integers(5, 20)),
                )
                notifications = max(
                    0,
                    int(usage_minutes / rng.integers(2, 8)),
                )


                if category in ["Productivity", "Learning"]:
                    study_minutes = rng.integers(20, 150)
                else:
                    study_minutes = rng.integers(0, 60)


                hour_of_day = int(
                    rng.choice([8, 10, 13, 16, 19, 21, 22, 23])
                )
                sleep_hours = round(float(rng.normal(7, 1)), 1)
                sleep_hours = max(4.0, min(10.0, sleep_hours))


                rows.append(
                    {
                        "date": date.date(),
                        "user_id": user_id_text,
                        "app_name": app,
                        "category": category,
                        "usage_minutes": usage_minutes,
                        "sessions": sessions,
                        "notifications": notifications,
                        "hour_of_day": hour_of_day,
                        "day_of_week": day_of_week,
                        "study_minutes": study_minutes,
                        "sleep_hours": sleep_hours,
                    }
                )


    df = pd.DataFrame(rows)


    # Calculate each user's total app usage for the day.
    df["daily_total_usage_minutes"] = df.groupby(
        ["user_id", "date"]
    )["usage_minutes"].transform("sum")


    # Calculate each user's total study/focus time for the day.
    df["daily_study_minutes"] = df.groupby(
        ["user_id", "date"]
    )["study_minutes"].transform("sum")


    df[TARGET_COLUMN] = df.apply(
        assign_productivity_label,
        axis=1,
    )


    return df



def assign_productivity_label(row: pd.Series) -> str:
    """
    Assign a productivity label using simple, explainable rules.
    """
    usage = row["daily_total_usage_minutes"]
    study = row["daily_study_minutes"]
    sleep = row["sleep_hours"]


    if study >= 90 and usage <= 350 and sleep >= 6.5:
        return "Productive"
    elif usage >= 550 or study <= 30:
        return "Distracted"
    else:
        return "Moderate"



def save_dataset(df: pd.DataFrame) -> None:
    """
    Save the generated dataset to data/raw/app_usage.csv.
    """
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)



if __name__ == "__main__":
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo


    # Use India time regardless of where the script runs.
    now = datetime.now(ZoneInfo("Asia/Kolkata"))


    # Generate a rolling 90-day report ending yesterday.
    end_date = now.date() - timedelta(days=1)
    start_date = end_date - timedelta(days=89)


    # Different synthetic data for each four-hour time window.
    refresh_seed = int(now.timestamp()) // (4 * 60 * 60)


    dataset = generate_app_usage_data(
        start_date=start_date.isoformat(),
        days=90,
        users=5,
        seed=refresh_seed,
    )


    save_dataset(dataset)


    print(f"Dataset created: {RAW_DATA_FILE}")
    print(f"Report period: {start_date} to {end_date}")
    print(f"Refresh time: {now.isoformat()}")
    print(f"Rows: {len(dataset)}")
    print(dataset.head())