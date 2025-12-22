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
from sklearn.linear_model import ElasticNet
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
kamis_df = pd.read_csv(DATA_DIR / 'kamis_maize_prices.csv')
agri_df = pd.read_csv(DATA_DIR / 'agriBORA_maize_prices.csv')
print(f"  - KAMIS data: {kamis_df.shape[0]} rows")
print(f"  - agriBORA data: {agri_df.shape[0]} rows")

# Selecting all entries with White Maize
kdf = kamis_df[kamis_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

## Selecting all entries in the agriBORA set with White Maize (all the data)
agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

## The counties of interest
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

print(f"\n[2/8] Filtering to target counties: {', '.join(target_counties)}")
## Restricting to target counties
kdf = kdf[kdf['County'].isin(target_counties)]
agri_df = agri_df[agri_df['County'].isin(target_counties)]

## Select the price columns (Wholesale)
kdf['price'] = pd.to_numeric(kdf['Wholesale'], errors='coerce')
agri_df['price'] = pd.to_numeric(agri_df['WholeSale'], errors='coerce')

## Convert Date to datetime object
kdf['Date'] = pd.to_datetime(kdf['Date'])
agri_df['Date'] = pd.to_datetime(agri_df['Date'])

# Weekly alignment (use Monday as start of ISO week)
agri_df["week"] = agri_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
kdf["week"] = kdf["Date"].dt.to_period("W").apply(lambda p: p.start_time)

print(f"\n[3/8] Aggregating to weekly data...")
## Aggregate to County-Week wholesale mean
kdf_week = (
    kdf.groupby(['County', 'week'], as_index=False)['price']
    .mean()
    .rename(columns={'price': 'kamis_price'})
)

agri_df_week = (
    agri_df.groupby(['County', 'week'], as_index=False)['price']
    .mean()
    .rename(columns={'price': 'agri_price'})
)

# Normalize data for modeling
agr = pd.read_csv(DATA_DIR / "agriBORA_maize_prices.csv", parse_dates=["Date"])
kamis = pd.read_csv(DATA_DIR / "kamis_maize_prices.csv", parse_dates=["Date"])

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
    agr_week,
    on=["county_norm", "week_start"],
    how="left"
)

print(f"\n[5/8] Creating lagged features and training set...")
# CREATE LAGS & TRAINING SET
panel = panel.sort_values(["county_norm", "week_start"])

for lag in [1, 2, 3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["kamis_smooth"].shift(lag)

# Drop rows with missing lags or missing Agribora price
panel_train = panel.dropna(subset=["lag1", "lag2", "lag3", "agr_price"]).reset_index(drop=True)

X = panel_train[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm"]]
y = panel_train["agr_price"]

print(f"  - Training samples: {len(X)}")

# UNIFIED SINGLE ML MODEL (ElasticNet)
print(f"\n[6/8] Training ElasticNet model...")
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

enet = ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", enet)
])

model.fit(X, y)

# Inspect training fit quality
y_pred_train = model.predict(X)
train_mae = mean_absolute_error(y, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
print(f"  - Train MAE: {train_mae:.2f}")
print(f"  - Train RMSE: {train_rmse:.2f}")

# FORECAST FOR TARGET WEEKS
print(f"\n[7/8] Generating forecasts...")
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
    if hist.empty:
        continue

    last3 = hist["kamis_smooth"].tail(3).values
    if len(last3) == 1:
        lag1 = lag2 = lag3 = last3[-1]
    elif len(last3) == 2:
        lag1 = last3[-1]
        lag2 = lag3 = last3[-2]
    else:
        lag1 = last3[-1]
        lag2 = last3[-2]
        lag3 = last3[-3]

    current_week = global_last_week

    while current_week < target_end:
        next_week = current_week + timedelta(days=7)
        X_h = pd.DataFrame({
            "kamis_smooth": [lag1],
            "lag1": [lag1],
            "lag2": [lag2],
            "lag3": [lag3],
            "county_norm": [c]
        })
        pred_h = model.predict(X_h)[0]

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        lag3 = lag2
        lag2 = lag1
        lag1 = pred_h
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
