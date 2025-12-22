# agriBORA Commodity Price Forecasting Challenge

## Overview

This repository contains a machine learning solution for forecasting weekly maize prices in Kenya as part of the agriBORA Commodity Price Forecasting Challenge. The goal is to predict average weekly wholesale prices of white maize for five counties: **Kiambu, Kirinyaga, Mombasa, Nairobi, and Uasin-Gishu**.

The forecasting period spans from **November 17, 2025 to January 10, 2026**, generating predictions for two consecutive weeks at a time.

## Challenge Context

Smallholder farmers across Africa often face price volatility and post-harvest losses of up to 40%. agriBORA aims to help farmers by offering certified warehouses for safe storage and digital warehouse certificates, enabling access to loans and flexible selling options. Accurate price forecasts help farmers time their sales effectively and increase earnings.

## Project Structure

```
.
├── data/
│   ├── raw/                          # Raw data files
│   │   ├── agriBORA_maize_prices.csv
│   │   ├── kamis_maize_prices.csv
│   │   └── ...
│   └── processed/                    # Processed data (generated)
├── notebooks/
│   └── StarterNotebook.ipynb        # Original Jupyter notebook
├── scripts/
│   └── forecast_maize_prices.py     # Main forecasting script
├── submissions/
│   ├── SampleSubmission.csv         # Sample submission template
│   └── submission.csv               # Generated predictions
├── Background.md                     # Challenge details
└── README.md                         # This file
```

## Methodology

### Data Sources

1. **KAMIS Dataset**: Historical wholesale and retail prices of maize (white, yellow, and mixed-traditional) from various markets in Kenya (2021-2025)
2. **agriBORA Dataset**: Transaction data showing wholesale prices of white maize

### Model Approach

The solution uses an **ElasticNet regression model** with the following features:

- **Smoothed KAMIS prices**: 3-week rolling mean of historical KAMIS wholesale prices
- **Lag features**: Previous 1, 2, and 3 weeks of smoothed prices
- **County encoding**: One-hot encoding for different counties

#### Key Steps:

1. **Data Preprocessing**:
   - Filter for white maize commodity only
   - Aggregate daily prices to weekly averages
   - Fill missing values using forward/backward filling
   - Apply 3-week rolling mean smoothing

2. **Feature Engineering**:
   - Create lagged features (1-3 weeks)
   - County-based encoding for spatial patterns
   - Weekly alignment using ISO week standards

3. **Model Training**:
   - ElasticNet with alpha=0.05 and l1_ratio=0.2
   - Preprocessing pipeline with standardization and one-hot encoding
   - Training performance: MAE ~3.83, RMSE ~4.76

4. **Forecasting**:
   - Recursive multi-step ahead predictions
   - Generate forecasts for 8 weeks (Week 47 through Week 2 of the following year)
   - Predictions for all 5 target counties

## Installation & Usage

### Prerequisites

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Running the Forecasting Script

**Option 1: Python Script (Recommended)**
```bash
# From the project root directory
python scripts/forecast_maize_prices.py
```

**Option 2: Jupyter Notebook**
```bash
# Open and run the original starter notebook
jupyter notebook notebooks/StarterNotebook.ipynb
```

The Python script ([scripts/forecast_maize_prices.py](scripts/forecast_maize_prices.py)) is a converted version of the starter notebook with updated paths and improved logging.

### Output

The script generates:
- Console output showing progress and model performance
- `submissions/submission.csv` - Final predictions in the required format

### Submission Format

The submission file contains three columns:
- **ID**: County name + Week number (e.g., "Kiambu_Week_50")
- **Target_RMSE**: Predicted price for RMSE evaluation
- **Target_MAE**: Predicted price for MAE evaluation (same value)

Sample output:
```
ID,Target_RMSE,Target_MAE
Kiambu_Week_50,36.86,36.86
Kirinyaga_Week_50,39.11,39.11
Mombasa_Week_50,36.86,36.86
Nairobi_Week_50,33.86,33.86
Uasin-Gishu_Week_50,37.21,37.21
...
```

## Evaluation Metrics

The challenge uses multi-metric evaluation:
- **MAE (50%)**: Mean Absolute Error - average magnitude of prediction errors
- **RMSE (50%)**: Root Mean Square Error - penalizes large errors more heavily

## Results

**Training Performance**:
- Train MAE: 3.83
- Train RMSE: 4.76
- Training samples: 134

**Predictions Generated**: 40 forecasts (8 weeks × 5 counties)

## Future Improvements

Potential enhancements to explore:
- Incorporate external data sources (precipitation, NDVI vegetation indices)
- Experiment with time series models (ARIMA, Prophet, LSTM)
- Add seasonal and trend decomposition
- Ensemble multiple models
- Feature selection and hyperparameter tuning
- Cross-validation for robust performance estimation

## About agriBORA

agriBORA is an agri-fintech company redefining how grain is stored, financed, and traded across East Africa through its agriGHALA platform, which integrates climate-smart warehousing, digital finance, and structured market linkages.

## License

This project is part of the Digital Africa initiative's agriBORA Commodity Price Forecasting Challenge.

## References

- [Challenge Details](Background.md)
- KAMIS (Kenya Agricultural Market Information System)
- Digital Africa Initiative