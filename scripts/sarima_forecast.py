#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script
Predicts weekly maize prices for selected counties in Kenya
"""

## Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

print("="*60)
print("agriBORA Commodity Price Forecasting")
print("="*60)

# Load the data
print("\n[1/7] Loading data...")
agr = pd.read_csv(DATA_DIR / "agriBORA_maize_prices.csv", parse_dates=["Date"])
kamis = pd.read_csv(DATA_DIR / "kamis_maize_prices.csv", parse_dates=["Date"])

# Filter to white maize
agr = agr[agr["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
kamis = kamis[kamis["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

print(f"\n[2/7] Filtering to target counties and commodity...")
def norm_county(s):
    return s.strip() if isinstance(s, str) else s

agr["county_norm"] = agr["County"].apply(norm_county)
kamis["county_norm"] = kamis["County"].apply(norm_county)

agr = agr[agr["county_norm"].isin(target_counties)].copy()
kamis = kamis[kamis["county_norm"].isin(target_counties)].copy()

print(f"\n[3/7] Aggregating to weekly data...")
# Weekly aggregation
agr["week_start"] = agr["Date"].dt.to_period("W").apply(lambda p: p.start_time)
kamis["week_start"] = kamis["Date"].dt.to_period("W").apply(lambda p: p.start_time)

agr["agr_price"] = pd.to_numeric(agr["WholeSale"], errors="coerce")
kamis["kamis_price"] = pd.to_numeric(kamis["Wholesale"], errors="coerce")

agr_week = agr.groupby(["county_norm", "week_start"], as_index=False)["agr_price"].mean()
kamis_week = kamis.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].mean()

print(f"\n[4/7] Building clean weekly panel per county...")
# BUILD CLEAN WEEKLY PANEL PER COUNTY (gap-filling + smoothing)
all_panels = []

for c in target_counties:
    sub = kamis_week[kamis_week["county_norm"] == c].copy()
    if sub.empty:
        continue

    # Continuous weekly index from first to last observed for that county
    min_d = sub["week_start"].min()
    max_d = sub["week_start"].max()
    full_weeks = pd.date_range(min_d, max_d, freq="W-MON")

    df = pd.DataFrame({"week_start": full_weeks})
    df["county_norm"] = c

    df = df.merge(sub[["week_start", "kamis_price"]],
                  on="week_start", how="left")

    # Fill missing KAMIS prices within that county
    df["kamis_price"] = df["kamis_price"].ffill().bfill()

    # 3-week rolling mean smoothing
    df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1).mean()

    all_panels.append(df)

kamis_panel = pd.concat(all_panels, ignore_index=True)

# Merge with Agribora weekly prices
panel = kamis_panel.merge(
    agr_week,
    on=["county_norm", "week_start"],
    how="left"
)

# For SARIMA, we work with each county's time series individually.
print(f"\n[5/7] Preparing data for SARIMA...")
panel = panel.sort_values(["county_norm", "week_start"])
# Ensure we have a proper datetime index
panel['week_start'] = pd.to_datetime(panel['week_start'])
panel = panel.set_index('week_start')

print("  - Data prepared for time series analysis.")

# FORECAST FOR TARGET WEEKS with SARIMA
print(f"\n[6/7] Training SARIMA model and generating forecasts...")
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")
forecast_rows = []

# Define the p, d and q parameters to take any value between 0 and 1
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
base_seasonal_pdq = list(itertools.product(p, d, q))

for c in target_counties:
    print(f"  - Forecasting for {c}...")
    county_data = panel[panel["county_norm"] == c]["agr_price"].dropna()

    if county_data.empty:
        print(f"    - No 'agr_price' data for {c}, skipping.")
        continue

    # Disable seasonality if data is less than 2 seasons
    if len(county_data) < 104:
        seasonal_pdq = [(0,0,0,0)]
    else:
        seasonal_pdq = [(x[0], x[1], x[2], 52) for x in base_seasonal_pdq]


    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    best_model = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(county_data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_model = results
            except:
                continue

    print(f"    - Best SARIMA order for {c}: {best_pdq} x {best_seasonal_pdq} (AIC: {best_aic})")

    # Generate forecasts
    forecast = best_model.get_forecast(steps=(target_end - county_data.index.max()).days // 7)
    forecast_index = pd.date_range(start=county_data.index.max() + timedelta(days=7), periods=len(forecast.predicted_mean), freq='W-MON')

    for date, pred_price in zip(forecast_index, forecast.predicted_mean):
        forecast_rows.append({
            "county": c,
            "week_start": date,
            "agr_pred": pred_price
        })

forecast_df = pd.DataFrame(forecast_rows)

# Filter only target weeks (from Nov 17, 2025 to Jan 10, 2026)
mask = (forecast_df["week_start"] >= target_start) & (forecast_df["week_start"] <= target_end)
forecast_target = forecast_df[mask].copy()
forecast_target = forecast_target.sort_values(['county', 'week_start'])

print(f"  - Generated {len(forecast_target)} forecasts")

print(f"\n[7/7] Creating submission file...")
# Build submission
forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
forecast_target["Target_MAE"] = forecast_target["agr_pred"]

submission = forecast_target[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

# Save submission
submission_path = SUBMISSIONS_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\n{'='*60}")
print(f"SUCCESS! Submission file created:")
print(f"  {submission_path}")
print(f"{'='*60}\n")

# Display sample of submission
print("Submission preview:")
print(submission)
