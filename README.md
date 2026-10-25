![CI](https://github.com/irfan271/focusguard-insights/actions/workflows/ci.yml/badge.svg)

# FocusGuard Insights

**Status:** Completed portfolio project demonstrating an end-to-end synthetic-data machine-learning pipeline, grouped and temporal evaluation, automated testing, continuous integration, model transparency, and Streamlit deployment.

FocusGuard Insights is an AI, machine-learning, and data-analysis project that analyzes synthetic app-usage data, predicts productivity levels, provides personalized focus recommendations, and allows users to download a prediction report.

The project uses an automated synthetic-data refresh through GitHub Actions on a four-hour schedule. Updated CSV reports are reflected in the deployed Streamlit dashboard.

## Live Dashboard

The application is deployed on Streamlit Community Cloud.

> Add your deployed app URL here after publishing:
>
> ****[Open FocusGuard Insights](https://focusguard-insights.streamlit.app/)****

## Project Overview

The project uses synthetic daily app-usage data to answer questions such as:

- Which apps consume the most time?
- How does usage vary across users, categories, and weeks?
- Can machine learning predict whether a user is productive, moderate, or distracted?
- What personalized actions can help improve focus?

## Key Features

- Synthetic app-usage data generation
- Automated data refresh using GitHub Actions
- Data cleaning and feature engineering
- Exploratory data analysis
- Productivity classification using Random Forest
- User-day grouped model evaluation
- Future-date holdout evaluation
- Rule-based baseline and transparent label-leakage documentation
- Productivity prediction for new usage records
- Downloadable prediction report
- Personalized focus recommendations
- Interactive Streamlit dashboard
- Weekly usage trend visualization
- Automated pytest test suite
- GitHub Actions continuous integration

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
- pytest
- GitHub Actions
- Jupyter Notebook

## Project Structure

```text
focusguard-insights/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── dashboard/
│   └── app.py
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── requirements.txt
├── .gitignore
├── MODEL_CARD.md
└── README.md
```

## Dashboard Features

The deployed dashboard contains three main sections:

### Dashboard

- User and app-category filters
- Total app usage, study time, and sleep KPIs
- Usage-by-app visualization
- Weekly usage trend
- Week-over-week usage comparison
- Top distraction app insight
- Productivity distribution

### Predict Productivity

- App usage, session, notification, study, sleep, and time-of-day inputs
- Daily screen-time and study-time totals
- Weekend indicator
- Input validation for inconsistent values
- Productivity prediction
- Personalized focus recommendations
- Downloadable prediction report
- Prediction confidence and class-probability display
- Session-based prediction history
- Downloadable prediction history in CSV format

### About the Model

- Explanation of model inputs and prediction process
- Description of grouped and temporal evaluation
- Clear disclosure of synthetic-data limitations
- Explanation of label leakage
- Intended-use and non-intended-use guidance
- Link to the full model card

## Model Transparency

The productivity classifier is trained on synthetic data. Its labels are generated from rules based on the same features available to the model, so its high evaluation scores reflect **label leakage** rather than validated real-world predictive performance.

The model should be treated as a demonstration of an end-to-end machine-learning pipeline, not as a validated predictor of real-world productivity.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the full model card, evaluation approach, data limitations, and intended-use guidance.

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
- Daily total screen time
- Daily total study time
- Weekend indicator
- Notifications per session
- Usage per session
- App category

The trained model is saved at:

```text
models/productivity_classifier.joblib
```

Model evaluation metadata is saved at:

```text
models/model_metadata.json
```

## Model Evaluation

The project uses two complementary evaluation approaches:

### User-day grouped split

All app-usage records belonging to the same user and date are kept together in either the training or testing set. This prevents user-day overlap between the two sets.

### Future-date holdout

The model trains on the earliest 80% of dates and evaluates on the latest 20% of dates. This evaluates temporal generalization within the synthetic dataset.

### Rule-based baseline

A rule-based baseline is included because the productivity labels are generated from rules using the same input features. This makes the label-leakage limitation explicit in both the training output and model metadata.

## How to Run

### 1. Clone the repository

```cmd
git clone [https://github.com/irfan271/focusguard-insights.git](https://github.com/irfan271/focusguard-insights.git)
cd focusguard-insights
```

### 2. Create and activate the virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Generate the dataset

```cmd
python -m src.data_generator
```

### 5. Clean the dataset

```cmd
python -m src.data_cleaning
```

### 6. Train the model

```cmd
python -m src.train_model
```

### 7. Test prediction

```cmd
python -m src.predict
```

### 8. Test recommendations

```cmd
python -m src.recommendations
```

### 9. Run the dashboard

```cmd
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

## Run Tests

The project includes automated tests for feature preparation, category encoding, baseline behavior, model artifacts, metadata transparency, and model loading.

```powershell
python -m pytest -q
```
## Screenshots
![FocusGuard Insights dashboard]
<img width="1577" height="733" alt="image" src="https://github.com/user-attachments/assets/3d3bf2e5-7abf-41f0-a2bf-49dfde8e8fe3" />
<img width="1602" height="660" alt="image" src="https://github.com/user-attachments/assets/caf2dc12-c438-4839-ae88-270c47716378" />
<img width="1582" height="621" alt="image" src="https://github.com/user-attachments/assets/fad8fb88-5404-415f-97ec-b2d22baf3113" />

![FocusGuard Insights prediction page]
<img width="1518" height="562" alt="image" src="https://github.com/user-attachments/assets/4d455d44-1bfb-4672-8d92-9fd067c5e96d" />

![FocusGuard Insights prediction page Result]
<img width="1533" height="817" alt="image" src="https://github.com/user-attachments/assets/33129093-aa64-40b2-9e6b-1bcdbaa59bcf" />



## Continuous Integration

GitHub Actions automatically installs the project dependencies and runs the pytest suite on every push and pull request targeting `main`.

The CI workflow is defined at:

```text
.github/workflows/ci.yml
```

## Future Improvements

- Connect real screen-time data from Android or iOS
- Add user authentication
- Store data in a database
- Add time-series forecasting for future usage
- Use an LLM to generate more natural recommendations
- Add downloadable CSV or PDF prediction reports
- Add dashboard screenshots and a product demo
