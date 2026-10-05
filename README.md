# Electricity Demand Forecasting with XGBoost

An end-to-end time-series machine-learning project that forecasts electricity demand at five-minute intervals. The work combines exploratory data analysis (EDA), visualisation, time-based feature engineering, and an XGBoost regression model.

## Project at a glance

| Item | Detail |
| --- | --- |
| Prediction target | Electricity demand (`Power demand`) |
| Observation frequency | 5 minutes |
| Raw observations | 393,440 |
| Raw variables | 15 |
| Coverage | 1 January 2021, 00:30 to 12 December 2024, 00:30 |
| Training period | 8 January 2021 to 31 December 2023 |
| Test period | 2 January 2024 to 12 December 2024 |
| Training observations | 297,159 |
| Test observations | 94,265 |
| Model | `XGBRegressor` |
| Test MAE | **42.32** demand units |
| Test RMSE | **117.79** demand units |

> The first 2,016 records are unavailable for modelling after creating the one-week lag feature. Metrics are taken from the notebook's chronological 2024 hold-out evaluation.

## Results

The final XGBoost model was evaluated on data after the training cut-off of 31 December 2023:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Mean Absolute Error (MAE) | 42.32 | Average absolute difference between predicted and observed demand |
| Root Mean Squared Error (RMSE) | 117.79 | Error measure that gives more weight to larger misses |

The model's features use only calendar attributes, temperature, and historical demand values available before the prediction timestamp. The notebook plots actual versus predicted electricity demand for the held-out period.

## Dataset overview

The included dataset contains 393,440 five-minute demand observations with weather and calendar fields.

| Statistic | Electricity demand | Temperature |
| --- | ---: | ---: |
| Mean | 3,960.74 | 25.53 |
| Median | 3,832.32 | — |
| Minimum | 1,302.08 | 4.00 |
| Maximum | 8,631.53 | 46.40 |

Raw columns include electricity demand, timestamp, temperature, dew point, humidity, wind direction, wind speed, pressure, and calendar components. The modelling workflow retains the variables relevant to the current experiment and removes unused weather columns.

## Workflow

1. Load and inspect the five-minute power-demand data.
2. Clean columns and convert the timestamp to a datetime index.
3. Explore the demand series through daily trends, monthly distributions, demand distribution, temperature-demand scatterplots, and a correlation heatmap.
4. Create calendar features: year, month, day, hour, minute, week of month, day of week, and weekend flag.
5. Create autoregressive features: 5-minute, 1-hour, 1-day, and 1-week lags, plus the trailing one-hour mean.
6. Split chronologically: up to 2023 for training and 2024 for testing.
7. Train an XGBoost regressor and evaluate with MAE and RMSE.
8. Save the fitted model for reuse.

## Features used for modelling

| Category | Features |
| --- | --- |
| Weather | `Temperature` |
| Calendar | `Year`, `Month`, `Day`, `Hour`, `Minute`, `Week`, `Day_of_week`, `Is_Weekend` |
| Demand history | `Lag_5min`, `Lag_1hour`, `Lag_1day`, `Lag_1week`, `mean_last_hour` |

## Repository structure

```text
Electricity Demand Forcasting/
├── Data/
│   └── Power_Demand_2021-2024.csv
├── Model/
│   └── Electricity Demand Prediction Model.pkl
├── Notebook/
│   └── Electricity Demand Forcasting.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting started

```bash
git clone https://github.com/<Er-Krishbhatia-1219>/Electricity-Demand-Forecasting.git
cd Electricity-Demand-Forecasting
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
jupyter notebook "Notebook/Electricity Demand Forcasting.ipynb"
```

The notebook is set up for the included dataset. If you use a different dataset location, update the load path in the first data-loading cell.

## Model configuration

```python
XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    learning_rate=0.01,
    random_state=42,
    objective="reg:squarederror"
)
```

## Limitations and next steps

- This is a one-step-ahead forecasting setup: lag features depend on previously observed demand. Multi-step forecasting needs a strategy for generating future lag values.
- The data source and demand unit were not provided with the project files; add these before presenting the work as a production or domain-specific study.
- Add rolling-origin cross-validation (`TimeSeriesSplit`) and report variability across folds.
- Compare against a seasonal-naive baseline and models such as LightGBM, SARIMAX, or Prophet.
- Consider holiday calendars and additional weather variables after validating their availability at forecast time.
- Add automated data validation and model tracking before deployment.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Joblib · Jupyter

## Author

Krish Bhatia
Data Scientist
