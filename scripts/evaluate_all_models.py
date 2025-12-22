#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates all models from PRs with time series cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

print("="*80)
print("MODEL EVALUATION WITH TIME SERIES CROSS-VALIDATION")
print("="*80)

# Load and prepare data
print("\n[1/6] Loading and preparing data...")
agr = pd.read_csv(DATA_DIR / "agriBORA_maize_prices.csv", parse_dates=["Date"])
kamis = pd.read_csv(DATA_DIR / "kamis_maize_prices.csv", parse_dates=["Date"])

# Filter to white maize
agr = agr[agr["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
kamis = kamis[kamis["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()

# Target counties
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

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

# Build clean weekly panel per county
all_panels = []
for c in target_counties:
    sub = kamis_week[kamis_week["county_norm"] == c].copy()
    if sub.empty:
        continue

    min_d = sub["week_start"].min()
    max_d = sub["week_start"].max()
    full_weeks = pd.date_range(min_d, max_d, freq="W-MON")

    df = pd.DataFrame({"week_start": full_weeks})
    df["county_norm"] = c

    df = df.merge(sub[["week_start", "kamis_price"]], on="week_start", how="left")
    df["kamis_price"] = df["kamis_price"].ffill().bfill()
    df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1).mean()

    all_panels.append(df)

kamis_panel = pd.concat(all_panels, ignore_index=True)

# Merge with Agribora weekly prices
panel = kamis_panel.merge(agr_week, on=["county_norm", "week_start"], how="left")

# Create lags
panel = panel.sort_values(["county_norm", "week_start"])
for lag in [1, 2, 3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["kamis_smooth"].shift(lag)

# Drop rows with missing lags or missing Agribora price
panel_train = panel.dropna(subset=["lag1", "lag2", "lag3", "agr_price"]).reset_index(drop=True)

X = panel_train[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm"]]
y = panel_train["agr_price"]

print(f"  - Total samples: {len(X)}")
print(f"  - Features: {list(X.columns)}")
print(f"  - Counties: {target_counties}")

# =============================================================================
# MODEL 1: GRADIENT BOOSTING (ElasticNet from PR #6)
# =============================================================================
print("\n[2/6] Evaluating PR #6: Gradient Boosting (ElasticNet)...")

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
model1_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    enet = ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42)
    model = Pipeline(steps=[("prep", preprocess), ("model", enet)])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    model1_scores.append({'fold': fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

model1_results = pd.DataFrame(model1_scores)
print(f"  Average: MAE={model1_results['MAE'].mean():.2f}, RMSE={model1_results['RMSE'].mean():.2f}, R²={model1_results['R2'].mean():.4f}")

# =============================================================================
# MODEL 2: PROPHET (PR #7)
# =============================================================================
print("\n[3/6] Evaluating PR #7: Prophet...")

try:
    from prophet import Prophet
    
    model2_scores = []
    
    # For Prophet, we need time series format per county
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        fold_mae, fold_rmse, fold_r2 = [], [], []
        
        for county in target_counties:
            # Get data for this county
            county_data = panel_train[panel_train["county_norm"] == county].copy()
            
            if len(county_data) < 10:
                continue
                
            # Split based on time index
            n_train = int(len(county_data) * 0.8)
            train_data = county_data.iloc[:n_train]
            test_data = county_data.iloc[n_train:]
            
            if len(test_data) == 0:
                continue
            
            # Prepare for Prophet
            prophet_train = pd.DataFrame({
                'ds': train_data['week_start'],
                'y': train_data['agr_price']
            })
            
            # Train Prophet
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                          daily_seasonality=False, changepoint_prior_scale=0.05)
            model.fit(prophet_train)
            
            # Forecast
            future = pd.DataFrame({'ds': test_data['week_start']})
            forecast = model.predict(future)
            
            y_true = test_data['agr_price'].values
            y_pred = forecast['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0
            
            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)
        
        if fold_mae:
            model2_scores.append({
                'fold': fold, 
                'MAE': np.mean(fold_mae), 
                'RMSE': np.mean(fold_rmse), 
                'R2': np.mean(fold_r2)
            })
            print(f"  Fold {fold}: MAE={np.mean(fold_mae):.2f}, RMSE={np.mean(fold_rmse):.2f}, R²={np.mean(fold_r2):.4f}")
    
    model2_results = pd.DataFrame(model2_scores)
    print(f"  Average: MAE={model2_results['MAE'].mean():.2f}, RMSE={model2_results['RMSE'].mean():.2f}, R²={model2_results['R2'].mean():.4f}")
    
except ImportError:
    print("  Prophet not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "prophet"])
    print("  Prophet installed. Rerun script to evaluate.")
    model2_results = pd.DataFrame([{'fold': 1, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}])

# =============================================================================
# MODEL 3: SARIMA (PR #8)
# =============================================================================
print("\n[4/6] Evaluating PR #8: SARIMA...")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    model3_scores = []
    
    for fold in range(1, 6):
        fold_mae, fold_rmse, fold_r2 = [], [], []
        
        for county in target_counties:
            county_data = panel_train[panel_train["county_norm"] == county].copy()
            
            if len(county_data) < 20:
                continue
            
            # Split
            n_train = int(len(county_data) * 0.8)
            train_data = county_data.iloc[:n_train]
            test_data = county_data.iloc[n_train:]
            
            if len(test_data) == 0:
                continue
            
            try:
                # Fit SARIMA
                model = SARIMAX(train_data['agr_price'].values,
                              order=(1, 1, 1),
                              seasonal_order=(1, 1, 1, 12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit(disp=False)
                
                # Forecast
                forecast = results.forecast(steps=len(test_data))
                
                y_true = test_data['agr_price'].values
                y_pred = forecast
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0
                
                fold_mae.append(mae)
                fold_rmse.append(rmse)
                fold_r2.append(r2)
            except:
                continue
        
        if fold_mae:
            model3_scores.append({
                'fold': fold,
                'MAE': np.mean(fold_mae),
                'RMSE': np.mean(fold_rmse),
                'R2': np.mean(fold_r2)
            })
            print(f"  Fold {fold}: MAE={np.mean(fold_mae):.2f}, RMSE={np.mean(fold_rmse):.2f}, R²={np.mean(fold_r2):.4f}")
    
    model3_results = pd.DataFrame(model3_scores)
    print(f"  Average: MAE={model3_results['MAE'].mean():.2f}, RMSE={model3_results['RMSE'].mean():.2f}, R²={model3_results['R2'].mean():.4f}")
    
except Exception as e:
    print(f"  SARIMA evaluation failed: {str(e)}")
    model3_results = pd.DataFrame([{'fold': 1, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}])

# =============================================================================
# MODEL 4: LSTM (PR #9)
# =============================================================================
print("\n[5/6] Evaluating PR #9: LSTM...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model4_scores = []
    
    # Prepare data for LSTM (3D: samples, timesteps, features)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # One-hot encode county using pandas get_dummies
        X_train_encoded = pd.get_dummies(X_train, columns=['county_norm'], drop_first=False)
        X_test_encoded = pd.get_dummies(X_test, columns=['county_norm'], drop_first=False)
        
        # Align columns
        missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0
        # Align columns
        missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        # Select all numeric features (now all features are numeric after one-hot encoding)
        X_train_lstm = X_train_encoded.values
        X_test_lstm = X_test_encoded.values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_lstm = scaler.fit_transform(X_train_lstm)
        X_test_lstm = scaler.transform(X_test_lstm)
        
        # Reshape for LSTM (samples, timesteps=1, features)
        X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], 1, X_train_lstm.shape[1])
        X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], 1, X_test_lstm.shape[1])
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(1, X_train_lstm.shape[2])),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Predict
        y_pred = model.predict(X_test_lstm, verbose=0).flatten()
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        model4_scores.append({'fold': fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        print(f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    model4_results = pd.DataFrame(model4_scores)
    print(f"  Average: MAE={model4_results['MAE'].mean():.2f}, RMSE={model4_results['RMSE'].mean():.2f}, R²={model4_results['R2'].mean():.4f}")
    
except Exception as e:
    print(f"  LSTM evaluation failed: {str(e)}")
    model4_results = pd.DataFrame([{'fold': 1, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}])

# =============================================================================
# MODEL 5: RANDOM FOREST (PR #10)
# =============================================================================
print("\n[6/6] Evaluating PR #10: Random Forest...")

from sklearn.ensemble import RandomForestRegressor

model5_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # One-hot encode county
    X_train_encoded = pd.get_dummies(X_train, columns=['county_norm'], drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=['county_norm'], drop_first=False)
    
    # Align columns
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                               min_samples_split=5, random_state=42)
    rf.fit(X_train_encoded, y_train)
    
    # Predict
    y_pred = rf.predict(X_test_encoded)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    model5_scores.append({'fold': fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

model5_results = pd.DataFrame(model5_scores)
print(f"  Average: MAE={model5_results['MAE'].mean():.2f}, RMSE={model5_results['RMSE'].mean():.2f}, R²={model5_results['R2'].mean():.4f}")

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': [
        'PR #6: Gradient Boosting (ElasticNet)',
        'PR #7: Prophet',
        'PR #8: SARIMA',
        'PR #9: LSTM',
        'PR #10: Random Forest'
    ],
    'Avg_MAE': [
        model1_results['MAE'].mean(),
        model2_results['MAE'].mean(),
        model3_results['MAE'].mean(),
        model4_results['MAE'].mean(),
        model5_results['MAE'].mean()
    ],
    'Avg_RMSE': [
        model1_results['RMSE'].mean(),
        model2_results['RMSE'].mean(),
        model3_results['RMSE'].mean(),
        model4_results['RMSE'].mean(),
        model5_results['RMSE'].mean()
    ],
    'Avg_R2': [
        model1_results['R2'].mean(),
        model2_results['R2'].mean(),
        model3_results['R2'].mean(),
        model4_results['R2'].mean(),
        model5_results['R2'].mean()
    ]
})

# Sort by RMSE (lower is better)
comparison = comparison.sort_values('Avg_RMSE')

print("\n")
print(comparison.to_string(index=False))

print("\n" + "="*80)
print("BEST MODEL:")
best_model = comparison.iloc[0]
print(f"  {best_model['Model']}")
print(f"  MAE: {best_model['Avg_MAE']:.2f}")
print(f"  RMSE: {best_model['Avg_RMSE']:.2f}")
print(f"  R²: {best_model['Avg_R2']:.4f}")
print("="*80 + "\n")

# Save results
results_path = BASE_DIR / "model_comparison_results.csv"
comparison.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")
