from pathlib import Path

# Base project folder: focusguard-insights/
BASE_DIR = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "app_usage.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "app_usage_clean.csv"

# Model files
MODEL_DIR = BASE_DIR / "models"
CLASSIFIER_FILE = MODEL_DIR / "productivity_classifier.joblib"
REGRESSOR_FILE = MODEL_DIR / "focus_regressor.joblib"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Reports
REPORT_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

# App configuration
APPS = [
    "Instagram",
    "WhatsApp",
    "Snapchat",
    "X",
    "YouTube",
    "Netflix",
    "Spotify",
    "Twitch",
    "Notion",
    "Google Docs",
    "Slack",
    "Outlook",
    "Coursera",
    "Khan Academy",
    "Duolingo",
    "Udemy",
    "Chrome",
    "Gmail",
    "Maps",
    "Calculator",
]

APP_CATEGORIES = {
    "Instagram": "Social",
    "WhatsApp": "Social",
    "Snapchat": "Social",
    "X": "Social",
    "YouTube": "Entertainment",
    "Netflix": "Entertainment",
    "Spotify": "Entertainment",
    "Twitch": "Entertainment",
    "Notion": "Productivity",
    "Google Docs": "Productivity",
    "Slack": "Productivity",
    "Outlook": "Productivity",
    "Coursera": "Learning",
    "Khan Academy": "Learning",
    "Duolingo": "Learning",
    "Udemy": "Learning",
    "Chrome": "Utility",
    "Gmail": "Utility",
    "Maps": "Utility",
    "Calculator": "Utility",
}

TARGET_COLUMN = "productivity_label"

PRODUCTIVITY_LABELS = [
    "Productive",
    "Moderate",
    "Distracted",
]