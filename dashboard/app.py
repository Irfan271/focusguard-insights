import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import plotly.express as px
import streamlit as st


from src.config import CLEAN_DATA_FILE, TARGET_COLUMN
from src.predict import predict_productivity
from src.recommendations import generate_recommendations


st.set_page_config(
    page_title="FocusGuard Insights",
    page_icon="📱",
    layout="wide",
)


st.title("📱 FocusGuard Insights")
st.write(
    "Analyze app usage, predict productivity, and receive personalized "
    "focus recommendations."
)


# Load cleaned dataset.
@st.cache_data(ttl=900)
def load_data():
    return pd.read_csv(CLEAN_DATA_FILE)



df = load_data()


dashboard_tab, predictor_tab, about_tab = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Predict Productivity",
        "ℹ️ About the Model",
    ]
)


with dashboard_tab:
    # Sidebar filters.
    st.sidebar.header("Filters")


    selected_user = st.sidebar.selectbox(
        "Select user",
        sorted(df["user_id"].unique()),
    )


    selected_category = st.sidebar.selectbox(
        "Select app category",
        ["All"] + sorted(df["category"].unique()),
    )


    filtered_df = df[df["user_id"] == selected_user]


    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["category"] == selected_category
        ]


    # KPI cards.
    total_usage = int(filtered_df["usage_minutes"].sum())
    total_study = int(filtered_df["study_minutes"].sum())
    average_sleep = round(float(filtered_df["sleep_hours"].mean()), 1)


    col1, col2, col3 = st.columns(3)


    col1.metric("Total App Usage", f"{total_usage:,} min")
    col2.metric("Total Study Time", f"{total_study:,} min")
    col3.metric("Average Sleep", f"{average_sleep} hrs")


    st.divider()


    # Usage by app chart.
    st.subheader("Usage by App")


    app_usage = (
        filtered_df.groupby("app_name")["usage_minutes"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


    fig_app_usage = px.bar(
        app_usage,
        x="usage_minutes",
        y="app_name",
        orientation="h",
        title="Total Usage Minutes by App",
        labels={
            "usage_minutes": "Usage Minutes",
            "app_name": "App",
        },
    )


    st.plotly_chart(fig_app_usage, use_container_width=True)


    # Weekly usage trend.
    st.subheader("Weekly Usage Trend")


    daily_usage = (
        filtered_df.groupby("date")["usage_minutes"]
        .sum()
        .reset_index()
    )


    daily_usage["date"] = pd.to_datetime(daily_usage["date"])
    daily_usage["week"] = daily_usage["date"].dt.strftime("%Y-%U")


    weekly_usage = (
        daily_usage.groupby("week")["usage_minutes"]
        .sum()
        .reset_index()
    )


    fig_weekly_usage = px.line(
        weekly_usage,
        x="week",
        y="usage_minutes",
        markers=True,
        title="Total Weekly App Usage",
        labels={
            "week": "Week",
            "usage_minutes": "Total Usage Minutes",
        },
    )


    st.plotly_chart(fig_weekly_usage, use_container_width=True)


    # Week-over-week usage comparison.
    st.subheader("Week-over-Week Comparison")


    weekly_usage_sorted = weekly_usage.sort_values("week").reset_index(drop=True)


    if len(weekly_usage_sorted) >= 2:
        current_week_usage = int(
            weekly_usage_sorted.iloc[-1]["usage_minutes"]
        )
        previous_week_usage = int(
            weekly_usage_sorted.iloc[-2]["usage_minutes"]
        )


        usage_change = current_week_usage - previous_week_usage


        change_percentage = 0
        if previous_week_usage > 0:
            change_percentage = round(
                (usage_change / previous_week_usage) * 100,
                1,
            )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "This Week Usage",
            f"{current_week_usage:,} min",
        )


        col2.metric(
            "Last Week Usage",
            f"{previous_week_usage:,} min",
        )


        col3.metric(
            "Weekly Change",
            f"{usage_change:,} min",
            f"{change_percentage}%",
        )


        if usage_change > 0:
            st.warning(
                "Your screen time increased compared with last week. "
                "Consider setting stricter app limits."
            )
        elif usage_change < 0:
            st.success(
                "Your screen time decreased compared with last week. "
                "Keep maintaining your current focus routine."
            )
        else:
            st.info(
                "Your screen time remained the same compared with last week."
            )
    else:
        st.info(
            "Not enough data available to compare this week with the previous week."
        )


    # Top distraction app.
    st.subheader("Top Distraction App")


    distraction_apps = filtered_df[
        filtered_df["category"].isin(["Social", "Entertainment"])
    ]


    if not distraction_apps.empty:
        top_distraction = (
            distraction_apps.groupby("app_name")["usage_minutes"]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )


        top_app_name = top_distraction.index[0]
        top_app_usage = int(top_distraction.values[0])


        st.metric(
            "Top Distraction App",
            top_app_name,
            f"{top_app_usage:,} min",
        )


        if top_app_usage > 500:
            st.warning(
                f"{top_app_name} is your biggest distraction, with "
                f"{top_app_usage:,} total minutes of usage. "
                "Consider setting a daily limit."
            )
        else:
            st.info(
                f"{top_app_name} is your biggest distraction, with "
                f"{top_app_usage:,} total minutes of usage."
            )
    else:
        st.info(
            "No social or entertainment apps available for this selection."
        )


    # Productivity distribution.
    st.subheader("Productivity Distribution")


    productivity_counts = filtered_df[TARGET_COLUMN].value_counts().reset_index()
    productivity_counts.columns = ["Productivity", "Count"]


    fig_productivity = px.pie(
        productivity_counts,
        names="Productivity",
        values="Count",
        title="Productivity Label Distribution",
    )


    st.plotly_chart(fig_productivity, use_container_width=True)


with predictor_tab:
    st.subheader("Predict Productivity")
    st.write(
        "Enter a daily app-usage summary to generate a productivity "
        "prediction and focus recommendations."
    )


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


    is_weekend = st.checkbox("This record is for a Saturday or Sunday")


    prediction_result = None
    recommendations = []
    report_text = ""


    if st.button("Predict Productivity"):
        validation_errors = []


        if usage_minutes > daily_total_usage_minutes:
            validation_errors.append(
                "Total daily screen time cannot be less than "
                "the usage time for this app."
            )


        if study_minutes > daily_study_minutes:
            validation_errors.append(
                "Total daily study time cannot be less than "
                "the study minutes entered above."
            )


        if usage_minutes > 0 and sessions == 0:
            validation_errors.append(
                "Enter at least one session when app usage is greater than zero."
            )


        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()


        record = {
            "usage_minutes": usage_minutes,
            "sessions": sessions,
            "notifications": notifications,
            "hour_of_day": hour_of_day,
            "study_minutes": study_minutes,
            "sleep_hours": sleep_hours,
            "daily_total_usage_minutes": daily_total_usage_minutes,
            "daily_study_minutes": daily_study_minutes,
            "is_weekend": int(is_weekend),
            "notifications_per_session": round(
                notifications / max(sessions, 1), 2
            ),
            "usage_per_session": round(
                usage_minutes / max(sessions, 1), 2
            ),
            f"category_{category}": 1,
        }


        prediction_result = predict_productivity(record)
        recommendations = generate_recommendations(
            record,
            prediction_result,
        )


        prediction = prediction_result[TARGET_COLUMN]


        if prediction == "Productive":
            st.success(f"Predicted productivity: {prediction}")
        elif prediction == "Moderate":
            st.warning(f"Predicted productivity: {prediction}")
        else:
            st.error(f"Predicted productivity: {prediction}")


        st.write("### Personalized Recommendations")


        for recommendation in recommendations:
            st.write(f"- {recommendation}")


        report_text = f"""FocusGuard Insights — Prediction Report
Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}

PREDICTION
----------
Predicted productivity: {prediction_result[TARGET_COLUMN]}

INPUT SUMMARY
-------------
App usage minutes: {usage_minutes}
Sessions: {sessions}
Notifications: {notifications}
Study minutes: {study_minutes}
Sleep hours: {sleep_hours}
Hour of day: {hour_of_day}
App category: {category}
Total daily screen time: {daily_total_usage_minutes} minutes
Total daily study time: {daily_study_minutes} minutes
Weekend record: {'Yes' if is_weekend else 'No'}
Notifications per session: {round(notifications / max(sessions, 1), 2)}
Usage per session: {round(usage_minutes / max(sessions, 1), 2)}

RECOMMENDATIONS
---------------
"""

        for recommendation in recommendations:
            report_text += f"- {recommendation}\n"

        report_text += """
IMPORTANT LIMITATION
--------------------
This prediction is generated by a model trained on synthetic data.
The productivity labels were created from rules based on the same
features used for prediction, so the model's high evaluation scores
reflect label leakage rather than validated real-world predictive
performance.

This report should not be used for medical, academic, employment, or
clinical decisions.
"""

        st.download_button(
            label="Download prediction report",
            data=report_text,
            file_name="focusguard_prediction_report.txt",
            mime="text/plain",
        )


with about_tab:
    st.subheader("About the Model")
    st.write(
        "FocusGuard Insights demonstrates an end-to-end machine-learning "
        "pipeline for classifying synthetic app-usage records into "
        "productivity categories."
    )


    st.info(
        "Predicted categories: Productive, Moderate, and Distracted."
    )


    st.write("### How the model works")


    st.write(
        """
        The classifier uses behavioral features such as:

        - App usage duration
        - Session count
        - Notification count
        - Hour of day
        - Study time
        - Sleep duration
        - Daily total screen time
        - Daily total study time
        - Weekend indicator
        - Notifications per session
        - Usage per session
        - App category

        The app passes these values to a trained Random Forest classifier,
        which returns one of three productivity labels.
        """
    )


    st.write("### Data and evaluation")


    st.write(
        """
        The model was trained on synthetic data covering 50 users across
        90 consecutive days. Evaluation includes:

        - A user-day grouped split, which keeps all records from the same
          user and date in either training or testing.
        - A future-date holdout, which trains on the earliest 80% of dates
          and evaluates on the latest 20%.
        - A rule-based baseline to make the synthetic-label limitation explicit.
        """
    )


    st.warning(
        """
        **Important limitation:** the productivity labels are generated by
        rules that use the same underlying inputs available to the model.
        As a result, the model can reconstruct those rules almost perfectly.

        The high accuracy and F1-score therefore reflect **label leakage**,
        not validated real-world predictive performance.
        """
    )


    st.write("### Intended use")


    st.write(
        """
        This project is intended to demonstrate:

        - Reproducible machine-learning workflow design
        - Feature engineering
        - Grouped and temporal evaluation
        - Model artifact and metadata persistence
        - Automated testing with pytest
        - Streamlit dashboard deployment
        """
    )


    st.error(
        """
        **Not intended for:** diagnosing attention disorders, making
        employment or academic decisions, clinical assessment, or claiming
        real-world productivity prediction.
        """
    )


    st.write(
        "See [MODEL_CARD.md]"
        "(https://github.com/irfan271/focusguard-insights/blob/main/MODEL_CARD.md) "
        "in the project repository for the full model documentation."
    )