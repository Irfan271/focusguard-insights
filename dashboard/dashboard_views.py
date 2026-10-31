import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TARGET_COLUMN


def render_dashboard_tab(filtered_df):
    """Render the Dashboard tab."""
    st.caption(
        f"Showing **{len(filtered_df):,}** of "
        f"**{st.session_state.total_records:,}** app-usage records."
    )

    render_kpi_cards(filtered_df)

    st.divider()

    if filtered_df.empty:
        st.warning(
            "No app-usage records match the selected filters. "
            "Try widening the date range or clearing the app selection."
        )
        return

    render_usage_by_app_chart(filtered_df)
    render_weekly_usage_trend(filtered_df)
    render_week_over_week_comparison(filtered_df)
    render_top_distraction_app(filtered_df)
    render_productivity_distribution(filtered_df)


def render_kpi_cards(filtered_df):
    """Render high-level usage KPIs."""
    total_usage = int(filtered_df["usage_minutes"].sum())
    total_study = int(filtered_df["study_minutes"].sum())
    average_sleep = round(
        float(filtered_df["sleep_hours"].mean()),
        1,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total App Usage", f"{total_usage:,} min")
    col2.metric("Total Study Time", f"{total_study:,} min")
    col3.metric("Average Sleep", f"{average_sleep} hrs")


def render_usage_by_app_chart(filtered_df):
    """Render total usage by app."""
    st.subheader("Usage by App")

    app_usage = (
        filtered_df.groupby("app_name")["usage_minutes"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
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

    st.plotly_chart(fig, width="stretch")


def render_weekly_usage_trend(filtered_df):
    """Render total usage by week."""
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

    fig = px.line(
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

    st.plotly_chart(fig, width="stretch")


def render_week_over_week_comparison(filtered_df):
    """Compare the latest week with the previous week."""
    st.subheader("Week-over-Week Comparison")

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
        .sort_values("week")
        .reset_index(drop=True)
    )

    if len(weekly_usage) < 2:
        st.info(
            "Not enough data available to compare this week with "
            "the previous week."
        )
        return

    current_week_usage = int(
        weekly_usage.iloc[-1]["usage_minutes"]
    )
    previous_week_usage = int(
        weekly_usage.iloc[-2]["usage_minutes"]
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


def render_top_distraction_app(filtered_df):
    """Identify the highest-usage social or entertainment app."""
    st.subheader("Top Distraction App")

    distraction_apps = filtered_df[
        filtered_df["category"].isin(
            ["Social", "Entertainment"]
        )
    ]

    if distraction_apps.empty:
        st.info(
            "No social or entertainment apps available for this selection."
        )
        return

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


def render_productivity_distribution(filtered_df):
    """Render the productivity-label distribution."""
    st.subheader("Productivity Distribution")

    productivity_counts = (
        filtered_df[TARGET_COLUMN]
        .value_counts()
        .reset_index()
    )

    productivity_counts.columns = [
        "Productivity",
        "Count",
    ]

    fig = px.pie(
        productivity_counts,
        names="Productivity",
        values="Count",
        title="Productivity Label Distribution",
    )

    st.plotly_chart(fig, width="stretch")