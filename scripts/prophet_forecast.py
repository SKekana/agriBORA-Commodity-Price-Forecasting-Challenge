#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script (Prophet Strategy)
Predicts weekly maize prices for selected counties in Kenya using Prophet.
"""

## Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

print("="*60)
print("agriBORA Commodity Price Forecasting (Prophet Strategy)")
print("="*60)

## Load and preprocess data
print("\n[1/4] Loading and preprocessing data...")
agri_df = pd.read_csv(DATA_DIR / 'agriBORA_maize_prices.csv', parse_dates=["Date"])

# Filter for "White_Maize"
agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

# Normalize county names
def norm_county(s):
    return s.strip() if isinstance(s, str) else s
agri_df["county_norm"] = agri_df["County"].apply(norm_county)

# The counties of interest
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
agri_df = agri_df[agri_df['county_norm'].isin(target_counties)]

# Prepare price column
agri_df['agr_price'] = pd.to_numeric(agri_df['WholeSale'], errors='coerce')

# Aggregate to weekly data
agri_df["week_start"] = agri_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
agri_week = agri_df.groupby(['county_norm', 'week_start'], as_index=False)['agr_price'].mean()

print(f"  - Data loaded for counties: {', '.join(agri_week['county_norm'].unique())}")

print("\n[2/4] Training Prophet models and running cross-validation...")

all_forecasts = []
performance_reports = []
trained_models = {}

for county in target_counties:
    print(f"\nProcessing: {county}")
    county_df = agri_week[agri_week['county_norm'] == county].copy()

    if county_df.empty or len(county_df) < 12:  # Minimum data threshold
        print(f"  - Skipping {county} due to insufficient data.")
        continue

    # Prepare data for Prophet
    prophet_df = county_df[['week_start', 'agr_price']].rename(columns={
        'week_start': 'ds',
        'agr_price': 'y'
    })

    # Set a floor of 0 for the target variable
    prophet_df['y'] = prophet_df['y'].clip(lower=0)
    prophet_df['floor'] = 0


    # Instantiate and fit the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_df)
    trained_models[county] = model

    # Cross-validation with dynamic parameters
    print("  - Running cross-validation...")
    try:
        # Set dynamic parameters based on data length
        n_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
        initial_days = max(int(n_days * 0.5), 90) # At least 90 days
        horizon_days = max(int(n_days * 0.15), 30) # At least 30 days
        period_days = max(int(n_days * 0.15), 30) # At least 30 days

        if n_days < 180: # Not enough data for CV
             raise ValueError("Not enough data for cross-validation.")

        initial = f'{initial_days} days'
        horizon = f'{horizon_days} days'
        period = f'{period_days} days'

        print(f"    - CV params: initial={initial}, horizon={horizon}, period={period}")

        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="processes"
        )

        # Performance metrics
        df_p = performance_metrics(df_cv)
        print("  - Cross-validation performance:")
        print(df_p[['horizon', 'mae', 'rmse']].head())

        performance_reports.append({
            "county": county,
            "mae": df_p['mae'].mean(),
            "rmse": df_p['rmse'].mean(),
            "cv_status": "Success"
        })
    except Exception as e:
        print(f"  - Cross-validation failed: {e}")
        performance_reports.append({
            "county": county,
            "mae": np.nan,
            "rmse": np.nan,
            "cv_status": "Failed"
        })


print("\n[3/4] Generating forecasts...")

# Define the forecast period
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")
forecast_dates = pd.date_range(start=target_start, end=target_end, freq='W-MON')
all_forecast_dfs = []

for county, model in trained_models.items():
    print(f"  - Forecasting for {county}")
    future = pd.DataFrame({'ds': forecast_dates})
    future['floor'] = 0
    forecast = model.predict(future)

    # Store relevant columns
    forecast['county'] = county
    all_forecast_dfs.append(forecast[['ds', 'yhat', 'county']])

if not all_forecast_dfs:
    print("\nNo forecasts were generated. Exiting.")
else:
    forecast_df = pd.concat(all_forecast_dfs)

    print("\n[4/4] Creating submission file...")

    # Format the submission file
    forecast_df["week"] = forecast_df["ds"].dt.isocalendar().week.astype(int)
    forecast_df["ID"] = forecast_df["county"] + "_Week_" + forecast_df["week"].astype(str)
    forecast_df["Target_RMSE"] = forecast_df["yhat"].clip(lower=0)
    forecast_df["Target_MAE"] = forecast_df["yhat"].clip(lower=0)

    submission = forecast_df[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

    # Save submission
    submission_path = SUBMISSIONS_DIR / "submission_prophet.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUCCESS! Submission file created:")
    print(f"  {submission_path}")
    print(f"{'='*60}\n")

    # Display sample of submission
    print("Submission preview:")
    print(submission.head())

print("\nCross-Validation Performance Report:")
performance_df = pd.DataFrame(performance_reports)
print(performance_df.to_string(index=False))
