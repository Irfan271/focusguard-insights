import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from dashboard.about_views import render_about_tab
from dashboard.data_loader import load_data
from dashboard.dashboard_views import render_dashboard_tab
from dashboard.filters import (
    apply_filters,
    get_default_filters,
    handle_filter_reset,
    initialize_filter_state,
    render_sidebar_filters,
)
from dashboard.predictor_views import render_predictor_tab


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


df = load_data()

initialize_filter_state()
st.session_state.default_filters = get_default_filters(df)
handle_filter_reset()

filters = render_sidebar_filters(df)
filtered_df = apply_filters(df, filters)
st.session_state.total_records = len(df)


dashboard_tab, predictor_tab, about_tab = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Predict Productivity",
        "ℹ️ About the Model",
    ]
)


with dashboard_tab:
    render_dashboard_tab(filtered_df)


with predictor_tab:
    render_predictor_tab()


with about_tab:
    render_about_tab()