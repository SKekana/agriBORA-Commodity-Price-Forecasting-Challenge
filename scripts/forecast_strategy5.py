#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script
Strategy 5: Random Forest with Autoregressive Features
Predicts weekly maize prices for selected counties in Kenya
"""

## Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

print("="*60)
print("agriBORA Commodity Price Forecasting - Strategy 5")
print("="*60)

## Load the data
print("\n[1/9] Loading data...")
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

print(f"\n[5/9] Creating autoregressive features and training set...")
# CREATE AUTOREGRESSIVE FEATURES FROM agr_price
panel = panel.sort_values(["county_norm", "week_start"]).copy()

# We need to fill gaps in agr_price to create continuous features for feature engineering
# A simple approach is to forward-fill then back-fill
panel['agr_price_filled'] = panel.groupby('county_norm')['agr_price'].transform(
    lambda x: x.ffill().bfill()
)

# Create autoregressive features from the filled agr_price
for lag in [1, 2, 3, 4]:
    panel[f"agr_lag_{lag}"] = panel.groupby("county_norm")["agr_price_filled"].shift(lag)

# Add rolling window features
panel["rolling_mean_4"] = panel.groupby("county_norm")["agr_price_filled"].shift(1).rolling(4, min_periods=1).mean()
panel["rolling_std_4"] = panel.groupby("county_norm")["agr_price_filled"].shift(1).rolling(4, min_periods=1).std()

# Time-based features
panel["month"] = panel["week_start"].dt.month
panel["week_of_year"] = panel["week_start"].dt.isocalendar().week.astype(int)

# The target variable is the original agr_price, not the filled one.
# Drop rows where the original agr_price is missing, as these cannot be used for training.
# Also drop rows where the first lag is missing.
panel_train = panel.dropna(subset=["agr_price", "agr_lag_1"]).reset_index(drop=True)
panel_train = panel_train.fillna(0) # Fill any remaining NaNs (e.g., from rolling_std on first items)

features = [
    "kamis_smooth",
    "agr_lag_1",
    "agr_lag_2",
    "agr_lag_3",
    "agr_lag_4",
    "rolling_mean_4",
    "rolling_std_4",
    "month",
    "week_of_year",
    "county_norm"
]
X = panel_train[features]
y = panel_train["agr_price"]

print(f"  - Training samples: {len(X)}")
print(f"  - Features: {', '.join(features)}")

# TIME SERIES CROSS-VALIDATION
print(f"\n[6/9] Running Time Series Cross-Validation...")
from sklearn.model_selection import TimeSeriesSplit

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
mae_scores = []
rmse_scores = []

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    numeric_features = [
        "kamis_smooth", "agr_lag_1", "agr_lag_2", "agr_lag_3", "agr_lag_4",
        "rolling_mean_4", "rolling_std_4", "month", "week_of_year"
    ]
    categorical_features = ["county_norm"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=5, min_samples_leaf=2)

    model = Pipeline(steps=[
        ("prep", preprocess),
        ("model", rf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    print(f"  - Fold {i+1}/{n_splits} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print("\nCross-Validation Performance Report:")
print(f"  - Average MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
print(f"  - Average RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")


# FINAL MODEL TRAINING
print(f"\n[7/9] Training final model on all data...")
# Re-define model to train on all data
numeric_features = [
    "kamis_smooth", "agr_lag_1", "agr_lag_2", "agr_lag_3", "agr_lag_4",
    "rolling_mean_4", "rolling_std_4", "month", "week_of_year"
]
categorical_features = ["county_norm"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough"
)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=5, min_samples_leaf=2)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", rf)
])

model.fit(X, y)

# Inspect training fit quality
y_pred_train = model.predict(X)
train_mae = mean_absolute_error(y, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
print(f"  - Final Model Train MAE: {train_mae:.2f}")
print(f"  - Final Model Train RMSE: {train_rmse:.2f}")

# FORECAST FOR TARGET WEEKS
print(f"\n[8/9] Generating recursive forecasts...")

# Use the full `panel` dataframe which includes all feature engineering to get the last known state
last_known_data = panel.sort_values("week_start").groupby("county_norm").last().reset_index()

forecast_rows = []
target_start = pd.Timestamp("2025-11-17")
target_end = pd.Timestamp("2026-01-10")

for c in target_counties:
    county_last_data = last_known_data[last_known_data["county_norm"] == c]
    if county_last_data.empty:
        print(f"  - WARNING: No historical data for {c}, skipping forecast.")
        continue

    # Initialize feature values from the last known data point
    current_week = county_last_data["week_start"].iloc[0]

    # Use the filled price for lags, as it's our continuous signal for feature generation
    lag1 = county_last_data["agr_price_filled"].iloc[0]
    lag2 = county_last_data["agr_lag_1"].iloc[0]
    lag3 = county_last_data["agr_lag_2"].iloc[0]
    lag4 = county_last_data["agr_lag_3"].iloc[0]

    # Assume kamis_smooth stays constant at the last known value
    kamis_smooth_forecast = county_last_data["kamis_smooth"].iloc[0]

    # Maintain a history of prices for calculating rolling features
    price_history = list(panel[panel["county_norm"] == c]["agr_price_filled"].dropna().values)[-4:]

    while current_week < target_end:
        next_week = current_week + timedelta(days=7)

        # Update rolling features based on the price history
        rolling_mean = np.mean(price_history[-4:])
        rolling_std = np.std(price_history[-4:])

        # Time-based features for the week we are forecasting
        month = next_week.month
        week_of_year = next_week.isocalendar().week

        # Create input DataFrame for prediction, ensuring correct feature order
        X_h = pd.DataFrame({
            "kamis_smooth": [kamis_smooth_forecast],
            "agr_lag_1": [lag1],
            "agr_lag_2": [lag2],
            "agr_lag_3": [lag3],
            "agr_lag_4": [lag4],
            "rolling_mean_4": [rolling_mean],
            "rolling_std_4": [rolling_std if np.isfinite(rolling_std) else 0],
            "month": [month],
            "week_of_year": [week_of_year],
            "county_norm": [c]
        })

        # Ensure the column order matches the training `features` list
        X_h = X_h[features]

        pred_h = model.predict(X_h)[0]

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        # Update lags and history for the next iteration in the recursive forecast
        lag4, lag3, lag2, lag1 = lag3, lag2, lag1, pred_h
        current_week = next_week
        price_history.append(pred_h)

forecast_df = pd.DataFrame(forecast_rows)

# Filter only target weeks (from Nov 17, 2025 to Jan 10, 2026)
mask = (forecast_df["week_start"] >= target_start) & (forecast_df["week_start"] <= target_end)
forecast_target = forecast_df[mask].copy()
forecast_target = forecast_target.sort_values(['county', 'week_start'])

print(f"  - Generated {len(forecast_target)} forecasts")


print(f"\n[8/9] Creating submission file...")
# Build submission
forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
forecast_target["Target_MAE"] = forecast_target["agr_pred"]

submission = forecast_target[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

# Save submission
submission_path = SUBMISSIONS_DIR / "submission_strategy5.csv"
submission.to_csv(submission_path, index=False)

print(f"\n{'='*60}")
print(f"SUCCESS! Submission file created:")
print(f"  {submission_path}")
print(f"{'='*60}\n")

# Display sample of submission
print("Submission preview:")
print(submission)
