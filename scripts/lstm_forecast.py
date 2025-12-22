#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script (LSTM Strategy)
Predicts weekly maize prices for selected counties in Kenya using an LSTM model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

# --- 1. SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

print("="*60)
print("agriBORA Commodity Price Forecasting - LSTM Strategy")
print("="*60)

# --- 2. LOAD AND PREPROCESS DATA ---
def load_and_preprocess_data():
    """Loads, cleans, and prepares the weekly time series data."""
    print("\n[1/6] Loading and preprocessing data...")

    # Load datasets
    kamis_df = pd.read_csv(DATA_DIR / 'kamis_maize_prices.csv', parse_dates=["Date"])
    agri_df = pd.read_csv(DATA_DIR / 'agriBORA_maize_prices.csv', parse_dates=["Date"])

    # Filter for "White_Maize"
    kamis_df = kamis_df[kamis_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()
    agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

    # Define target counties
    target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

    # Filter for target counties
    kamis_df = kamis_df[kamis_df['County'].isin(target_counties)]
    agri_df = agri_df[agri_df['County'].isin(target_counties)]

    # Coerce price columns to numeric
    kamis_df['kamis_price'] = pd.to_numeric(kamis_df['Wholesale'], errors='coerce')
    agri_df['agr_price'] = pd.to_numeric(agri_df['WholeSale'], errors='coerce')

    # Aggregate to weekly means
    kamis_df["week_start"] = kamis_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    agri_df["week_start"] = agri_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)

    kamis_week = kamis_df.groupby(['County', 'week_start'], as_index=False)['kamis_price'].mean()
    agr_week = agri_df.groupby(['County', 'week_start'], as_index=False)['agr_price'].mean()

    # Build a clean, continuous weekly panel for each county
    all_panels = []
    for county in target_counties:
        sub = kamis_week[kamis_week["County"] == county].copy()
        if sub.empty:
            continue

        min_d, max_d = sub["week_start"].min(), sub["week_start"].max()
        full_weeks = pd.date_range(min_d, max_d, freq="W-MON")

        df = pd.DataFrame({"week_start": full_weeks, "County": county})
        df = df.merge(sub[["week_start", "kamis_price"]], on="week_start", how="left")

        # Fill missing values and apply smoothing
        df["kamis_price"] = df["kamis_price"].ffill().bfill()
        df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1).mean()

        all_panels.append(df)

    kamis_panel = pd.concat(all_panels, ignore_index=True)

    # Merge with agriBORA prices to create the final panel
    panel = kamis_panel.merge(
        agr_week,
        on=["County", "week_start"],
        how="left"
    )

    print("  - Data loaded and preprocessed successfully.")
    return panel.sort_values(["County", "week_start"])

# --- 3. CREATE SEQUENCES FOR LSTM ---
def create_sequences(data, sequence_length):
    """Creates sequences of data for LSTM model training."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 4. BUILD THE LSTM MODEL ---
def build_lstm_model(input_shape):
    """Builds and compiles the LSTM model."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 5. MAIN EXECUTION ---
def main():
    """Main function to run the forecasting pipeline."""
    panel = load_and_preprocess_data()

    SEQUENCE_LENGTH = 8  # 8 weeks of historical data to predict the next week
    N_SPLITS = 3         # 3 folds for cross-validation

    all_forecasts = []
    performance_metrics = []

    for county in panel['County'].unique():
        print(f"\n[2/6] Processing data for {county}...")

        county_data = panel[panel['County'] == county].copy()

        # Use 'kamis_smooth' as the primary feature, handle missing 'agr_price'
        county_data['target_price'] = county_data['agr_price'].fillna(county_data['kamis_smooth'])

        # Skip county if it has insufficient data
        if len(county_data) < (SEQUENCE_LENGTH + N_SPLITS):
            print(f"  - Skipping {county} due to insufficient data ({len(county_data)} weeks).")
            continue

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(county_data[['target_price']])

        # Create sequences
        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

        if len(X) == 0:
            print(f"  - Skipping {county} due to no sequences generated.")
            continue

        print(f"  - Training LSTM model for {county} with Time Series Cross-Validation...")

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold = 0
        county_scores = []

        for train_index, val_index in tscv.split(X):
            fold += 1
            print(f"    - Fold {fold}/{N_SPLITS}...")

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Reshape input for LSTM: [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

            model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, 1))

            # Callbacks for robust training
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=16,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0  # Suppress verbose output for cleaner logs
            )

            # Evaluate on validation set
            y_pred_scaled = model.predict(X_val)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_true = scaler.inverse_transform(y_val)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            county_scores.append({'mae': mae, 'rmse': rmse})
            print(f"      - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Average performance across folds
        avg_mae = np.mean([s['mae'] for s in county_scores])
        avg_rmse = np.mean([s['rmse'] for s in county_scores])
        performance_metrics.append({'county': county, 'avg_mae': avg_mae, 'avg_rmse': avg_rmse})
        print(f"  - Average CV Performance for {county}: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}")

        # --- Retrain on full data and forecast ---
        print(f"  - Retraining model on full data for {county}...")
        X_full = np.reshape(X, (X.shape[0], X.shape[1], 1))

        final_model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, 1))
        final_model.fit(X_full, y, epochs=100, batch_size=16, verbose=0,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=10),
                            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5)
                        ])

        print("  - Generating forecasts...")

        # Get the last sequence from the historical data
        last_sequence = scaled_data[-SEQUENCE_LENGTH:]
        current_sequence = np.reshape(last_sequence, (1, SEQUENCE_LENGTH, 1))

        # Define forecast period
        target_start = pd.Timestamp("2025-11-17")
        target_end = pd.Timestamp("2026-01-10")
        last_historical_date = county_data['week_start'].max()

        current_date = last_historical_date
        while current_date < target_end:
            # Predict the next price
            next_pred_scaled = final_model.predict(current_sequence, verbose=0)
            next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]

            # Update the date
            current_date += timedelta(days=7)

            # Store forecast if it's within the target window
            if current_date >= target_start:
                all_forecasts.append({
                    "county": county,
                    "week_start": current_date,
                    "agr_pred": next_pred
                })

            # Update the sequence for the next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred_scaled[0, 0]

    # --- 6. PERFORMANCE REPORT AND SUBMISSION ---
    print("\n[3/6] Generating performance report...")
    perf_df = pd.DataFrame(performance_metrics)
    print("\nCross-Validation Performance Report:")
    print(perf_df.to_string(index=False))

    # Calculate and print overall average performance
    if not perf_df.empty:
        overall_avg_mae = perf_df['avg_mae'].mean()
        overall_avg_rmse = perf_df['avg_rmse'].mean()
        print(f"\nOverall Average CV MAE: {overall_avg_mae:.2f}")
        print(f"Overall Average CV RMSE: {overall_avg_rmse:.2f}")

    print("\n[4/6] Creating submission file...")
    if not all_forecasts:
        print("  - No forecasts were generated. Skipping submission file creation.")
        return

    forecast_df = pd.DataFrame(all_forecasts)

    # Create the submission ID
    forecast_df["week"] = forecast_df["week_start"].dt.isocalendar().week.astype(int)
    forecast_df["ID"] = forecast_df["county"] + "_Week_" + forecast_df["week"].astype(str)

    # Format the submission file
    submission = forecast_df[["ID", "agr_pred"]].copy()
    submission.rename(columns={"agr_pred": "Target_RMSE"}, inplace=True)
    submission["Target_MAE"] = submission["Target_RMSE"]

    submission_path = SUBMISSIONS_DIR / "submission_lstm.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUCCESS! Submission file created at:")
    print(f"  {submission_path}")
    print(f"{'='*60}\n")
    print("Submission preview:")
    print(submission.head())

if __name__ == '__main__':
    main()
