# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Forecasting Script
```bash
python scripts/forecast_maize_prices.py
```

### 3. Check Your Submission
```bash
cat submissions/submission.csv
```

## What the Script Does

The forecasting script performs the following steps:

1. **Loads Data**: Reads KAMIS and agriBORA maize price datasets
2. **Filters Data**: Selects white maize for target counties (Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu)
3. **Aggregates**: Converts daily prices to weekly averages
4. **Smooths**: Applies 3-week rolling mean to reduce noise
5. **Creates Features**: Generates lag features (1-3 weeks) for time series modeling
6. **Trains Model**: Fits an ElasticNet regression model
7. **Generates Forecasts**: Produces recursive predictions for 8 weeks ahead
8. **Saves Submission**: Writes predictions to `submissions/submission.csv`

## Expected Output

```
============================================================
agriBORA Commodity Price Forecasting
============================================================

[1/8] Loading data...
  - KAMIS data: 21888 rows
  - agriBORA data: 239 rows

[2/8] Filtering to target counties: Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu

[3/8] Aggregating to weekly data...

[4/8] Building clean weekly panel per county...

[5/8] Creating lagged features and training set...
  - Training samples: 134

[6/8] Training ElasticNet model...
  - Train MAE: 3.83
  - Train RMSE: 4.76

[7/8] Generating forecasts...
  - Generated 40 forecasts

[8/8] Creating submission file...

============================================================
SUCCESS! Submission file created:
  /workspaces/.../submissions/submission.csv
============================================================
```

## Submission File Format

The generated `submissions/submission.csv` contains:
- 40 rows (5 counties × 8 weeks)
- 3 columns: ID, Target_RMSE, Target_MAE
- Week numbers: 47, 48, 49, 50, 51, 52, 1, 2

Example:
```csv
ID,Target_RMSE,Target_MAE
Kiambu_Week_50,36.86,36.86
Kirinyaga_Week_50,39.11,39.11
Mombasa_Week_50,36.86,36.86
...
```

## Troubleshooting

### Module Not Found Error
```bash
# Install missing packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

### File Not Found Error
- Ensure you run the script from the project root directory
- Verify data files exist in `data/raw/`

### Permission Errors
```bash
# Make script executable (Linux/Mac)
chmod +x scripts/forecast_maize_prices.py
```

## Next Steps

1. Review the predictions in `submissions/submission.csv`
2. Explore the original notebook: `notebooks/StarterNotebook.ipynb`
3. Experiment with different models or features
4. Check [README.md](README.md) for detailed methodology
