from src.config import TARGET_COLUMN


def generate_recommendations(record: dict, prediction_result: dict) -> list:
    """
    Generate simple, explainable recommendations based on usage
    and the predicted productivity label.
    """
    recommendations = []

    prediction = prediction_result[TARGET_COLUMN]

    usage = record.get("usage_minutes", 0)
    study = record.get("study_minutes", 0)
    notifications = record.get("notifications", 0)
    sleep = record.get("sleep_hours", 0)
    daily_usage = record.get("daily_total_usage_minutes", 0)
    daily_study = record.get("daily_study_minutes", 0)

    # Recommendations based on predicted productivity.
    if prediction == "Distracted":
        recommendations.append(
            "Your usage pattern suggests a high distraction risk. "
            "Try a 45-minute focused study session without social apps."
        )
    elif prediction == "Moderate":
        recommendations.append(
            "Your focus is moderate. Reduce one high-usage app by "
            "20 minutes to improve productivity."
        )
    else:
        recommendations.append(
            "You are maintaining a productive routine. Keep your current "
            "study schedule and app limits."
        )

    # Usage-based recommendation.
    if daily_usage > 450:
        recommendations.append(
            f"Total screen time is {daily_usage} minutes today. "
            "Set a daily usage limit of 5 hours."
        )

    # Study-based recommendation.
    if daily_study < 60:
        recommendations.append(
            f"You studied only {daily_study} minutes today. "
            "Schedule at least one 45-minute focus block tomorrow."
        )

    # Notification-based recommendation.
    if notifications > 40:
        recommendations.append(
            f"You received {notifications} notifications. "
            "Enable Do Not Disturb during study hours."
        )

    # Sleep-based recommendation.
    if sleep < 6.5:
        recommendations.append(
            f"Your sleep is {sleep} hours. Aim for at least "
            "7 hours to support better focus."
        )

    # Avoid duplicate recommendations.
    unique_recommendations = []
    for recommendation in recommendations:
        if recommendation not in unique_recommendations:
            unique_recommendations.append(recommendation)

    return unique_recommendations


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

    sample_prediction = {
        TARGET_COLUMN: "Distracted",
        "probabilities": {
            "Productive": 0.02,
            "Moderate": 0.11,
            "Distracted": 0.87,
        },
    }

    recommendations = generate_recommendations(
        sample_record,
        sample_prediction,
    )

    print("Personalized recommendations:")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"{index}. {recommendation}")