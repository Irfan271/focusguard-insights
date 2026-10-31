from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import TARGET_COLUMN
from src.predict import predict_productivity
from src.recommendations import generate_recommendations

from dashboard.report_utils import build_prediction_report


def render_predictor_tab():
    """Render the Predict Productivity tab."""
    st.subheader("Predict Productivity")
    st.write(
        "Enter a daily app-usage summary to generate a productivity "
        "prediction and focus recommendations."
    )

    inputs = render_prediction_form()

    if st.button("Predict Productivity"):
        validation_errors = validate_prediction_inputs(inputs)

        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()

        render_prediction_result(inputs)


def render_prediction_form():
    """Render prediction inputs and return the entered values."""
    col1, col2, col3 = st.columns(3)

    usage_minutes = col1.number_input(
        "App usage minutes",
        min_value=0,
        value=120,
    )

    sessions = col2.number_input(
        "Sessions",
        min_value=0,
        value=14,
    )

    notifications = col3.number_input(
        "Notifications",
        min_value=0,
        value=45,
    )

    col4, col5, col6 = st.columns(3)

    study_minutes = col4.number_input(
        "Study minutes",
        min_value=0,
        value=35,
    )

    sleep_hours = col5.number_input(
        "Sleep hours",
        min_value=0.0,
        value=6.0,
    )

    hour_of_day = col6.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=22,
    )

    category = st.selectbox(
        "App category",
        [
            "Social",
            "Entertainment",
            "Utility",
            "Productivity",
            "Learning",
        ],
    )

    st.write("Daily totals")

    daily_col1, daily_col2 = st.columns(2)

    daily_total_usage_minutes = daily_col1.number_input(
        "Total screen time across all apps today (minutes)",
        min_value=0,
        max_value=1440,
        value=420,
        step=10,
    )

    daily_study_minutes = daily_col2.number_input(
        "Total study time today (minutes)",
        min_value=0,
        max_value=1440,
        value=60,
        step=10,
    )

    is_weekend = st.checkbox(
        "This record is for a Saturday or Sunday"
    )

    return {
        "usage_minutes": usage_minutes,
        "sessions": sessions,
        "notifications": notifications,
        "study_minutes": study_minutes,
        "sleep_hours": sleep_hours,
        "hour_of_day": hour_of_day,
        "category": category,
        "daily_total_usage_minutes": daily_total_usage_minutes,
        "daily_study_minutes": daily_study_minutes,
        "is_weekend": int(is_weekend),
    }


def validate_prediction_inputs(inputs):
    """Return a list of validation errors for the entered inputs."""
    errors = []

    if (
        inputs["usage_minutes"]
        > inputs["daily_total_usage_minutes"]
    ):
        errors.append(
            "Total daily screen time cannot be less than "
            "the usage time for this app."
        )

    if (
        inputs["study_minutes"]
        > inputs["daily_study_minutes"]
    ):
        errors.append(
            "Total daily study time cannot be less than "
            "the study minutes entered above."
        )

    if inputs["usage_minutes"] > 0 and inputs["sessions"] == 0:
        errors.append(
            "Enter at least one session when app usage is greater "
            "than zero."
        )

    return errors


def build_model_record(inputs):
    """
    Convert dashboard inputs into a raw model input record.

    The saved pipeline handles scaling and category encoding, so this
    function must not create one-hot columns or scale values manually.
    """
    sessions = max(inputs["sessions"], 1)

    return {
        "usage_minutes": inputs["usage_minutes"],
        "sessions": inputs["sessions"],
        "notifications": inputs["notifications"],
        "hour_of_day": inputs["hour_of_day"],
        "study_minutes": inputs["study_minutes"],
        "sleep_hours": inputs["sleep_hours"],
        "daily_total_usage_minutes": inputs[
            "daily_total_usage_minutes"
        ],
        "daily_study_minutes": inputs[
            "daily_study_minutes"
        ],
        "is_weekend": inputs["is_weekend"],
        "notifications_per_session": round(
            inputs["notifications"] / sessions,
            2,
        ),
        "usage_per_session": round(
            inputs["usage_minutes"] / sessions,
            2,
        ),
        "category": inputs["category"],
    }


def render_prediction_result(inputs):
    """Generate, display, and store one productivity prediction."""
    record = build_model_record(inputs)

    prediction_result = predict_productivity(record)
    recommendations = generate_recommendations(
        record,
        prediction_result,
    )

    prediction = prediction_result[TARGET_COLUMN]
    probabilities = prediction_result["probabilities"]
    confidence = probabilities[prediction]

    render_prediction_alert(prediction, confidence)
    render_confidence(probabilities, confidence)
    render_recommendations(recommendations)

    add_prediction_to_history(
        inputs,
        prediction,
        confidence,
    )

    report_text = build_prediction_report(
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        inputs={
            **inputs,
            "notifications_per_session": round(
                inputs["notifications"]
                / max(inputs["sessions"], 1),
                2,
            ),
            "usage_per_session": round(
                inputs["usage_minutes"]
                / max(inputs["sessions"], 1),
                2,
            ),
        },
        recommendations=recommendations,
    )

    st.download_button(
        label="Download prediction report",
        data=report_text,
        file_name="focusguard_prediction_report.txt",
        mime="text/plain",
    )

    render_prediction_history()


def render_prediction_alert(prediction, confidence):
    """Display the prediction with a color-coded alert."""
    message = (
        f"Predicted productivity: {prediction} "
        f"({confidence:.1%} confidence)"
    )

    if prediction == "Productive":
        st.success(message)
    elif prediction == "Moderate":
        st.warning(message)
    else:
        st.error(message)


def render_confidence(probabilities, confidence):
    """Display class probabilities and confidence explanation."""
    st.write("### Prediction Confidence")

    confidence_df = pd.DataFrame(
        [
            {
                "Productivity Label": class_name,
                "Probability": probability,
            }
            for class_name, probability in probabilities.items()
        ]
    )

    st.dataframe(
        confidence_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Confidence is the model’s predicted probability for the "
        "selected productivity label. Because this model is trained "
        "on synthetic rule-generated labels, confidence should not be "
        "interpreted as real-world certainty."
    )


def render_recommendations(recommendations):
    """Display personalized recommendations."""
    st.write("### Personalized Recommendations")

    for recommendation in recommendations:
        st.write(f"- {recommendation}")


def add_prediction_to_history(
    inputs,
    prediction,
    confidence,
):
    """Add the current prediction to session history."""
    history_record = {
        "predicted_at": datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "prediction": prediction,
        "confidence": confidence,
        "usage_minutes": inputs["usage_minutes"],
        "sessions": inputs["sessions"],
        "notifications": inputs["notifications"],
        "study_minutes": inputs["study_minutes"],
        "sleep_hours": inputs["sleep_hours"],
        "hour_of_day": inputs["hour_of_day"],
        "category": inputs["category"],
        "daily_total_usage_minutes": inputs[
            "daily_total_usage_minutes"
        ],
        "daily_study_minutes": inputs[
            "daily_study_minutes"
        ],
        "is_weekend": inputs["is_weekend"],
    }

    st.session_state.prediction_history.append(
        history_record
    )


def render_prediction_history():
    """Render prediction history, CSV export, and clear control."""
    st.divider()
    st.subheader("Prediction History")

    if not st.session_state.prediction_history:
        st.info(
            "No predictions yet. Generate a prediction to build your "
            "session history."
        )
        return

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_history = history_df.to_csv(index=False).encode(
        "utf-8"
    )

    download_col, clear_col = st.columns(2)

    download_col.download_button(
        label="Download prediction history (CSV)",
        data=csv_history,
        file_name="focusguard_prediction_history.csv",
        mime="text/csv",
    )

    if clear_col.button("Clear prediction history"):
        st.session_state.prediction_history = []
        st.rerun()