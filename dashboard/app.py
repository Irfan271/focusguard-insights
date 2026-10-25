import sys
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
@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA_FILE)


df = load_data()

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

# Productivity predictor.
st.divider()
st.subheader("Predict Productivity")

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

if st.button("Predict Productivity"):
    record = {
        "usage_minutes": usage_minutes,
        "sessions": sessions,
        "notifications": notifications,
        "hour_of_day": hour_of_day,
        "study_minutes": study_minutes,
        "sleep_hours": sleep_hours,
        "daily_total_usage_minutes": usage_minutes * 3,
        "daily_study_minutes": study_minutes,
        "is_weekend": 0,
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