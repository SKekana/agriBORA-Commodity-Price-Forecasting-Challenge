#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script
Predicts weekly maize prices for selected counties in Kenya
"""

## Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

print("="*60)
print("agriBORA Commodity Price Forecasting")
print("="*60)

## Load the data
# Load the data
print("\n[1/8] Loading data...")
kamis = pd.read_csv(DATA_DIR / "kamis_maize_prices.csv", parse_dates=["Date"])
agr = pd.read_csv(DATA_DIR / "agriBORA_maize_prices.csv", parse_dates=["Date"])
print(f"  - KAMIS data: {kamis.shape[0]} rows")
print(f"  - agriBORA data: {agr.shape[0]} rows")

# The counties of interest
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
print(f"\n[2/8] Filtering to target counties: {', '.join(target_counties)}")

# Filter to white maize
agr = agr[agr["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
kamis = kamis[kamis["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()

def norm_county(s):
    return s.strip() if isinstance(s, str) else s

agr["county_norm"] = agr["County"].apply(norm_county)
kamis["county_norm"] = kamis["County"].apply(norm_county)

agr = agr[agr["county_norm"].isin(target_counties)].copy()
kamis = kamis[kamis["county_norm"].isin(target_counties)].copy()

# Weekly aggregation
agr["week_start"] = agr["Date"].dt.to_period("W").apply(lambda p: p.start_time)
kamis["week_start"] = kamis["Date"].dt.to_period("W").apply(lambda p: p.start_time)

agr["agr_price"] = pd.to_numeric(agr["WholeSale"], errors="coerce")
kamis["kamis_price"] = pd.to_numeric(kamis["Wholesale"], errors="coerce")

agr_week = agr.groupby(["county_norm", "week_start"], as_index=False)["agr_price"].mean()
kamis_week = kamis.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].mean()

print(f"\n[3/8] Building clean weekly panel per county...")
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

print(f"\n[4/8] Creating lagged features and training set...")
# CREATE LAGS & TRAINING SET
panel = panel.sort_values(["county_norm", "week_start"])

for lag in [1, 2, 3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["kamis_smooth"].shift(lag)

# Drop rows with missing lags or missing Agribora price
panel_train = panel.dropna(subset=["lag1", "lag2", "lag3", "agr_price"]).reset_index(drop=True)

X = panel_train[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm"]]
y = panel_train["agr_price"]

print(f"  - Training samples: {len(X)}")

# UNIFIED SINGLE ML MODEL (Gradient Boosting)
print(f"\n[5/8] Training Gradient Boosting model...")
numeric_features = ["kamis_smooth", "lag1", "lag2", "lag3"]
categorical_features = ["county_norm"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

gbr = GradientBoostingRegressor(random_state=42)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", gbr)
])

# Time Series Cross-Validation
print("\n[+] Performing Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring)

mae_scores = -cv_results['test_neg_mean_absolute_error']
rmse_scores = -cv_results['test_neg_root_mean_squared_error']

print(f"  - CV MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")
print(f"  - CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")

model.fit(X, y)

# Inspect training fit quality
y_pred_train = model.predict(X)
train_mae = mean_absolute_error(y, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
print(f"\n  - Train MAE: {train_mae:.2f}")
print(f"  - Train RMSE: {train_rmse:.2f}")

# FORECAST FOR TARGET WEEKS
print(f"\n[6/8] Generating forecasts...")
panel_five = panel[panel["county_norm"].isin(target_counties)].copy()
panel_five = panel_five.sort_values(["county_norm", "week_start"])

# Recursive forecasts up to 2026-01-10
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")

forecast_rows = []

# Determine the global last observed week in the aligned panel
global_last_week = panel_five["week_start"].max()

for c in target_counties:
    hist = panel_five[panel_five["county_norm"] == c].sort_values("week_start")
    if hist.empty or len(hist["kamis_smooth"]) < 3:
        print(f"  - Not enough historical data for {c}, skipping.")
        continue

    # Setup for trend and initial lags
    kamis_history = hist["kamis_smooth"].tolist()
    trend = kamis_history[-1] - kamis_history[-2]
    if trend == 0:
        print(f"  - Zero trend for {c}, forecasts will be flat.")

    current_lags = {
        "kamis_smooth": kamis_history[-1],
        "lag1": kamis_history[-1],
        "lag2": kamis_history[-2],
        "lag3": kamis_history[-3]
    }

    current_week = global_last_week
    while current_week < target_end:
        next_week = current_week + timedelta(days=7)

        # Prepare features for prediction
        X_h = pd.DataFrame({
            "kamis_smooth": [current_lags["kamis_smooth"]],
            "lag1": [current_lags["lag1"]],
            "lag2": [current_lags["lag2"]],
            "lag3": [current_lags["lag3"]],
            "county_norm": [c]
        })
        pred_h = model.predict(X_h)[0]

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        # Update lags for the next step based on a linear trend projection
        projected_kamis_smooth = current_lags["kamis_smooth"] + trend

        new_lags = {
            "kamis_smooth": projected_kamis_smooth,
            "lag1": current_lags["kamis_smooth"],
            "lag2": current_lags["lag1"],
            "lag3": current_lags["lag2"]
        }
        current_lags = new_lags
        current_week = next_week

forecast_df = pd.DataFrame(forecast_rows)

# Filter only target weeks (from Nov 17, 2025 to Jan 10, 2026)
mask = (forecast_df["week_start"] >= target_start) & (forecast_df["week_start"] <= target_end)
forecast_target = forecast_df[mask].copy()
forecast_target = forecast_target.sort_values(['county', 'week_start'])

print(f"  - Generated {len(forecast_target)} forecasts")

print(f"\n[7/8] Creating submission file...")
# Build submission
forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
forecast_target["Target_MAE"] = forecast_target["agr_pred"]

submission = forecast_target[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

# Save submission
submission_path = SUBMISSIONS_DIR / "submission_strategy1.csv"
submission.to_csv(submission_path, index=False)

print(f"\n{'='*60}")
print(f"SUCCESS! Submission file created:")
print(f"  {submission_path}")
print(f"{'='*60}\n")

# Display sample of submission
print("Submission preview:")
print(submission)
