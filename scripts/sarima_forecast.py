#!/usr/bin/env python3
"""
agriBORA Commodity Price Forecasting Script (SARIMA Model)
Predicts weekly maize prices for selected counties in Kenya using SARIMA.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the SARIMA forecasting pipeline.
    """
    # Set up paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    SUBMISSIONS_DIR = BASE_DIR / "submissions"

    print("="*60)
    print("agriBORA Commodity Price Forecasting (SARIMA)")
    print("="*60)

    # 1. Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")

    # Load agriBORA data
    agri_df = pd.read_csv(DATA_DIR / 'agriBORA_maize_prices.csv', parse_dates=["Date"])

    # Filter for White Maize and target counties
    target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
    agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()
    agri_df = agri_df[agri_df['County'].isin(target_counties)].copy()

    # Normalize county names
    agri_df["county_norm"] = agri_df["County"].str.strip()

    # Convert price to numeric
    agri_df["agr_price"] = pd.to_numeric(agri_df["WholeSale"], errors="coerce")

    # Aggregate to weekly mean
    agri_df["week_start"] = agri_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly_prices = agri_df.groupby(["county_norm", "week_start"], as_index=False)["agr_price"].mean()

    # Create a complete date range for each county to handle missing weeks
    all_county_data = []
    for county in target_counties:
        county_df = weekly_prices[weekly_prices["county_norm"] == county].copy()
        if not county_df.empty:
            min_date = weekly_prices["week_start"].min()
            max_date = weekly_prices["week_start"].max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='W-MON')

            # Set week_start as index for reindexing
            county_df = county_df.set_index('week_start')

            # Reindex to fill missing weeks
            county_df = county_df.reindex(full_date_range)
            county_df['county_norm'] = county

            # Interpolate missing agr_price values
            county_df['agr_price'] = county_df['agr_price'].interpolate(method='time')

            all_county_data.append(county_df.reset_index().rename(columns={'index': 'week_start'}))

    if all_county_data:
        processed_data = pd.concat(all_county_data, ignore_index=True)
        print("  - Data preprocessing complete.")
        print(f"  - Processed data shape: {processed_data.shape}")
    else:
        print("  - No data available for the target counties.")
        processed_data = pd.DataFrame() # Ensure processed_data is defined

    # 2. SARIMA model training and cross-validation
    print("\n[2/5] Running SARIMA model training and cross-validation...")

    all_county_forecasts = {}
    performance_metrics = {}
    best_params = {}

    for county in target_counties:
        print(f"\n--- Processing County: {county} ---")
        county_ts = processed_data[processed_data["county_norm"] == county].copy()

        if county_ts.shape[0] < 20: # Basic check for sufficient data
            print(f"  - Skipping {county} due to insufficient data ({county_ts.shape[0]} weeks).")
            continue

        county_ts = county_ts.set_index('week_start')['agr_price']

        # Automated parameter selection (Grid Search)
        # Define the p, d, q parameters to take any value between 0 and 2
        # Define a limited set of candidate models to prevent timeout
        candidate_models = [
            ((1, 1, 1), (0, 0, 0, 0)), # Standard non-seasonal ARIMA
        ]

        # Add seasonal models only if there's enough data
        if len(county_ts) >= 104:
            candidate_models.extend([
                ((1, 1, 1), (1, 1, 1, 52)), # Common seasonal model
                ((0, 1, 1), (0, 1, 1, 52)), # Seasonal MA model
            ])

        best_aic = np.inf
        best_pdq = None
        best_seasonal_pdq = None

        print("  - Evaluating candidate SARIMA models...")
        for param, param_seasonal in candidate_models:
            try:
                mod = sm.tsa.statespace.SARIMAX(county_ts,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except Exception as e:
                # print(f"    - Error fitting {param}x{param_seasonal}: {e}")
                continue

        print(f"  - Best SARIMA params for {county}: {best_pdq} x {best_seasonal_pdq}")

        best_params[county] = {'order': best_pdq, 'seasonal_order': best_seasonal_pdq}

        # Time Series Cross-Validation (Rolling Forecast)
        # Using a simple rolling forecast for evaluation
        train_size = int(len(county_ts) * 0.8)
        train, test = county_ts[0:train_size], county_ts[train_size:]
        history = [x for x in train]
        predictions = []

        for t in range(len(test)):
            model = SARIMAX(history, order=best_pdq, seasonal_order=best_seasonal_pdq)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        # Calculate performance metrics
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        performance_metrics[county] = {'MAE': mae, 'RMSE': rmse}

        print(f"  - CV MAE: {mae:.2f}")
        print(f"  - CV RMSE: {rmse:.2f}")

    # 3. Generate final forecasts
    print("\n[3/5] Generating final forecasts...")

    # Retrain on full data and forecast
    all_forecasts = []
    target_start = pd.Timestamp("2025-11-17")
    target_end = pd.Timestamp("2026-01-10")

    for county in target_counties:
        print(f"  - Forecasting for {county}")
        county_ts = processed_data[processed_data["county_norm"] == county].copy()
        if county_ts.empty:
            continue

        county_ts = county_ts.set_index('week_start')['agr_price']

        # Get best params from CV step
        order = best_params.get(county, {}).get('order', (1,1,1)) # Default if something went wrong
        seasonal_order = best_params.get(county, {}).get('seasonal_order', (0,0,0,0))

        # Fit the model on the entire dataset
        model = SARIMAX(county_ts, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        # Forecast for the required period
        # The number of steps is the number of weeks between the end of the data and the target end date
        last_date = county_ts.index.max()
        weeks_to_forecast = (target_end - last_date).days // 7 + 1

        forecast = model_fit.get_forecast(steps=weeks_to_forecast).summary_frame()

        # Format the forecast output
        forecast['county'] = county
        forecast = forecast.reset_index().rename(columns={'index': 'week_start', 'mean': 'agr_pred'})

        # Filter to the exact target weeks required for submission
        forecast = forecast[
            (forecast["week_start"] >= target_start) & (forecast["week_start"] <= target_end)
        ]
        all_forecasts.append(forecast)

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    print(f"  - Generated {len(forecast_df)} total forecasts.")

    # 4. Create submission file
    print("\n[4/5] Creating submission file...")

    if not forecast_df.empty:
        # Format for submission
        forecast_df["week"] = forecast_df["week_start"].dt.isocalendar().week.astype(int)
        forecast_df["ID"] = forecast_df["county"] + "_Week_" + forecast_df["week"].astype(str)
        forecast_df["Target_RMSE"] = forecast_df["agr_pred"]
        forecast_df["Target_MAE"] = forecast_df["agr_pred"]

        submission = forecast_df[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

        # Save submission file
        submission_path = SUBMISSIONS_DIR / "submission.csv"
        submission.to_csv(submission_path, index=False)

        print(f"\n{'='*60}")
        print(f"SUCCESS! Submission file created:")
        print(f"  {submission_path}")
        print(f"{'='*60}\n")
        print("Submission preview:")
        print(submission.head())
    else:
        print("  - No forecasts were generated, skipping submission file creation.")

    print(f"\n[5/5] Script finished.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
