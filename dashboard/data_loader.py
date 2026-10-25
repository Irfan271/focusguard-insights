import pandas as pd
import streamlit as st

from src.config import CLEAN_DATA_FILE


@st.cache_data(ttl=900)
def load_data() -> pd.DataFrame:
    """Load and prepare the cleaned FocusGuard dataset."""
    df = pd.read_csv(CLEAN_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    return df