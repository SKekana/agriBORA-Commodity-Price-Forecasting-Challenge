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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
print("\n[1/8] Loading data...")
kamis_df = pd.read_csv(DATA_DIR / 'kamis_maize_prices.csv', parse_dates=["Date"])
agri_df = pd.read_csv(DATA_DIR / 'agriBORA_maize_prices.csv', parse_dates=["Date"])
print(f"  - KAMIS data: {kamis_df.shape[0]} rows")
print(f"  - agriBORA data: {agri_df.shape[0]} rows")

def preprocess_data(df, price_col, commodity="White_Maize", counties=None):
    """Standardize, filter, and preprocess the raw data."""
    df = df[df['Commodity_Classification'].str.contains(commodity, na=False)].copy()

    def norm_county(s):
        return s.strip() if isinstance(s, str) else s
    df["county_norm"] = df["County"].apply(norm_county)

    if counties:
        df = df[df["county_norm"].isin(counties)].copy()

    df['price'] = pd.to_numeric(df[price_col], errors='coerce')
    df["week_start"] = df["Date"].dt.to_period("W").apply(lambda p: p.start_time)

    df_week = (
        df.groupby(['county_norm', 'week_start'], as_index=False)['price']
        .mean()
    )
    return df_week

target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

print(f"\n[2/8] Filtering to target counties: {', '.join(target_counties)}")
kamis_week = preprocess_data(kamis_df, "Wholesale", counties=target_counties)
kamis_week = kamis_week.rename(columns={'price': 'kamis_price'})

agri_week = preprocess_data(agri_df, "WholeSale", counties=target_counties)
agri_week = agri_week.rename(columns={'price': 'agr_price'})

print(f"\n[3/8] Aggregating to weekly data...")
# Note: Aggregation is now part of the preprocess_data function
print("   - (Already done in preprocessing)")

print(f"\n[4/8] Building clean weekly panel per county...")
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
    agri_week,
    on=["county_norm", "week_start"],
    how="left"
)

# For autoregressive features, fill agr_price where possible (historic only)
panel["agr_price_filled"] = panel.groupby("county_norm")["agr_price"].transform(lambda x: x.ffill().bfill())

print(f"\n[5/8] Creating autoregressive features and training set...")
# CREATE AUTOREGRESSIVE FEATURES & TRAINING SET
panel = panel.sort_values(["county_norm", "week_start"])

# Lags, EMAs, and volatility are now based on the target variable
for lag in [1, 2, 3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["agr_price_filled"].shift(lag)

panel["ema3"] = panel.groupby("county_norm")["agr_price_filled"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
panel["ema6"] = panel.groupby("county_norm")["agr_price_filled"].transform(lambda x: x.ewm(span=6, adjust=False).mean())
panel["volatility"] = panel.groupby("county_norm")["agr_price_filled"].transform(lambda x: x.diff().rolling(3, min_periods=1).std())
panel["week_of_year"] = panel["week_start"].dt.isocalendar().week
panel["month"] = panel["week_start"].dt.month

# Drop rows where the original agr_price is missing, or where features could not be created
panel_train = panel.dropna(subset=["lag1", "lag2", "lag3", "agr_price", "volatility"]).reset_index(drop=True)

# Feature set now includes kamis_smooth (exogenous) and autoregressive features
X = panel_train[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm", "ema3", "ema6", "volatility", "week_of_year", "month"]]
y = panel_train["agr_price"]

print(f"  - Training samples: {len(X)}")

# UNIFIED SINGLE ML MODEL (Random Forest)
print(f"\n[6/8] Training RandomForestRegressor model...")
numeric_features = ["kamis_smooth", "lag1", "lag2", "lag3", "ema3", "ema6", "volatility", "week_of_year", "month"]
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

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", rf)
])

model.fit(X, y)

# Inspect training fit quality
y_pred_train = model.predict(X)
train_mae = mean_absolute_error(y, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
print(f"  - Train MAE: {train_mae:.2f}")
print(f"  - Train RMSE: {train_rmse:.2f}")

# FORECAST FOR TARGET WEEKS
print(f"\n[7/8] Generating autoregressive forecasts...")
panel_five = panel[panel["county_norm"].isin(target_counties)].copy()

# Recursive forecasts up to 2026-01-10
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")

forecast_rows = []

# Determine the global last observed week in the aligned panel
global_last_week = panel_five["week_start"].max()

for c in target_counties:
    hist = panel_five[panel_five["county_norm"] == c].copy().sort_values("week_start")
    if hist.empty:
        continue

    current_week = global_last_week

    while current_week < target_end:
        next_week = current_week + timedelta(days=7)

        # Create a placeholder for the next week and append it
        new_row_df = pd.DataFrame([{"week_start": next_week, "county_norm": c}])
        hist = pd.concat([hist, new_row_df], ignore_index=True)

        # Re-calculate autoregressive features based on the target variable
        # Use 'agr_price' which now includes previous predictions
        hist['agr_price_filled'] = hist['agr_price'].ffill()

        for lag in [1, 2, 3]:
            hist[f"lag{lag}"] = hist["agr_price_filled"].shift(lag)

        hist["ema3"] = hist["agr_price_filled"].ewm(span=3, adjust=False).mean()
        hist["ema6"] = hist["agr_price_filled"].ewm(span=6, adjust=False).mean()
        hist["volatility"] = hist["agr_price_filled"].diff().rolling(3, min_periods=1).std()
        hist["week_of_year"] = hist["week_start"].dt.isocalendar().week
        hist["month"] = hist["week_start"].dt.month

        # Carry forward the last known kamis_smooth value
        hist["kamis_smooth"].ffill(inplace=True)

        # Get the last row (the one we want to predict)
        X_h = hist.tail(1)[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm", "ema3", "ema6", "volatility", "week_of_year", "month"]]
        X_h.fillna(0, inplace=True) # Fill any remaining NaNs

        # Make the prediction
        pred_h = model.predict(X_h)[0]

        # Update the placeholder row with the prediction
        hist.loc[hist.index.max(), "agr_price"] = pred_h

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        current_week = next_week

forecast_df = pd.DataFrame(forecast_rows)

# Filter only target weeks (from Nov 17, 2025 to Jan 10, 2026)
mask = (forecast_df["week_start"] >= target_start) & (forecast_df["week_start"] <= target_end)
forecast_target = forecast_df[mask].copy()
forecast_target = forecast_target.sort_values(['county', 'week_start'])

print(f"  - Generated {len(forecast_target)} forecasts")

print(f"\n[8/8] Creating submission file...")
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
