#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script (Prophet Model)
Predicts weekly maize prices for selected counties in Kenya using Facebook Prophet.
"""

## Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

print("="*60)
print("agriBORA Commodity Price Forecasting (Prophet Model)")
print("="*60)

## Load the data
print("\n[1/5] Loading and preprocessing data...")
kamis_df = pd.read_csv(DATA_DIR / 'kamis_maize_prices.csv')

# Selecting all entries with White Maize
kdf = kamis_df[kamis_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

## The counties of interest
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

## Restricting to target counties
kdf = kdf[kdf['County'].isin(target_counties)]

## Select the price columns (Wholesale)
kdf['price'] = pd.to_numeric(kdf['Wholesale'], errors='coerce')

## Convert Date to datetime object
kdf['Date'] = pd.to_datetime(kdf['Date'])

# Weekly alignment (use Monday as start of ISO week)
kdf["week"] = kdf["Date"].dt.to_period("W").apply(lambda p: p.start_time)

## Aggregate to County-Week wholesale mean
kdf_week = (
    kdf.groupby(['County', 'week'], as_index=False)['price']
    .mean()
    .rename(columns={'price': 'kamis_price'})
)

# BUILD CLEAN WEEKLY PANEL PER COUNTY (gap-filling + smoothing)
all_panels = []

for c in target_counties:
    sub = kdf_week[kdf_week["County"] == c].copy()
    if sub.empty:
        continue

    # Continuous weekly index from first to last observed for that county
    min_d = sub["week"].min()
    max_d = sub["week"].max()
    full_weeks = pd.date_range(min_d, max_d, freq="W-MON")

    df = pd.DataFrame({"week": full_weeks})
    df["County"] = c

    df = df.merge(sub[["week", "kamis_price"]],
                  on="week", how="left")

    # Fill missing KAMIS prices within that county
    df["kamis_price"] = df["kamis_price"].ffill().bfill()

    all_panels.append(df)

panel = pd.concat(all_panels, ignore_index=True)
panel = panel.rename(columns={'week': 'ds', 'kamis_price': 'y'})

print(f"  - Data loaded and preprocessed for {len(target_counties)} counties.")

# FORECAST FOR TARGET WEEKS
print(f"\n[2/5] Training Prophet models and generating forecasts...")
forecast_rows = []
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")

for county in target_counties:
    print(f"  - Forecasting for {county}...")
    county_data = panel[panel['County'] == county].copy()

    if county_data.empty:
        print(f"    - No data for {county}, skipping.")
        continue

    # Initialize and train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(county_data)

    # Calculate the number of weeks needed for the forecast
    last_date = county_data['ds'].max()
    periods_needed = (target_end - last_date).days // 7 + 1

    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=periods_needed, freq='W-MON')
    forecast = model.predict(future)

    # Filter for the target forecast period
    mask = (forecast["ds"] >= target_start) & (forecast["ds"] <= target_end)
    county_forecast = forecast[mask].copy()
    county_forecast['County'] = county

    forecast_rows.append(county_forecast)

forecast_df = pd.concat(forecast_rows, ignore_index=True)
print(f"  - Generated {len(forecast_df)} forecasts.")


print(f"\n[3/5] Creating submission file...")
# Build submission
forecast_df["week"] = forecast_df["ds"].dt.isocalendar().week.astype(int)
forecast_df["ID"] = forecast_df["County"] + "_Week_" + forecast_df["week"].astype(str)
forecast_df["Target_RMSE"] = forecast_df["yhat"]
forecast_df["Target_MAE"] = forecast_df["yhat"]

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
