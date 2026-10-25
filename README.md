# FocusGuard Insights

FocusGuard Insights is a mini AI, Machine Learning, and Data Analysis project that analyzes app-usage data, predicts productivity levels, and provides personalized focus recommendations.

Automated synthetic-data refresh using GitHub Actions on a four-hour schedule, with updated CSV reports reflected in the deployed Streamlit dashboard.

## Project Overview

The project uses synthetic daily app-usage data to answer questions such as:

- Which apps consume the most time?
- How does usage vary across users, categories, and weeks?
- Can machine learning predict whether a user is productive, moderate, or distracted?
- What personalized actions can help improve focus?

## Key Features

- Synthetic app-usage data generation
- Data cleaning and feature engineering
- Exploratory data analysis
- Productivity classification using Random Forest
- Productivity prediction for new usage records
- Personalized recommendations
- Interactive Streamlit dashboard
- Weekly usage trend visualization

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
focusguard-insights/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── dashboard/
├── models/
├── reports/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset is synthetic and stored at:

```text
data/raw/app_usage.csv
```

Each row represents one user’s usage of one app on one date.

Important columns include:

- `date`
- `user_id`
- `app_name`
- `category`
- `usage_minutes`
- `sessions`
- `notifications`
- `study_minutes`
- `sleep_hours`
- `productivity_label`

The target variable is:

```text
productivity_label
```

Possible values are:

- `Productive`
- `Moderate`
- `Distracted`

## Machine Learning Model

The project uses a `RandomForestClassifier` to predict productivity.

Input features include:

- App usage minutes
- Number of sessions
- Notifications
- Hour of day
- Study minutes
- Sleep hours
- Daily total usage
- Daily study time
- Weekend indicator
- Notification and usage ratios
- App category

The trained model is saved at:

```text
models/productivity_classifier.joblib
```

## How to Run

### 1. Create and activate the virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

### 3. Generate the dataset

```cmd
python -m src.data_generator
```

### 4. Clean the dataset

```cmd
python -m src.data_cleaning
```

### 5. Train the model

```cmd
python -m src.train_model
```

### 6. Test prediction

```cmd
python -m src.predict
```

### 7. Test recommendations

```cmd
python -m src.recommendations
```

### 8. Run the dashboard

```cmd
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

## Dashboard Features

- User and app-category filters
- Total app usage, study time, and sleep KPIs
- Usage-by-app visualization
- Weekly usage trend
- Productivity distribution
- Productivity prediction form
- Personalized focus recommendations

## Future Improvements

- Connect real screen-time data from Android or iOS
- Add user authentication
- Store data in a database
- Add time-series forecasting for future usage
- Use an LLM to generate more natural recommendations
- Deploy the dashboard to Streamlit Community Cloud
