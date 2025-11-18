# Global ML Model - Cross-Company Earnings Beat Prediction

## Overview

Single machine learning model trained on all S&P 500 companies combined (~14,000 quarterly earnings observations).

**Key Difference from Company-Specific Models:**
- Company-specific: 414 separate models (AAPL model, IBM model, etc.)
- Global: 1 model trained on all companies together
- Advantage: Works for ANY company, even new ones not in training set

## Data Leak Prevention ⚠️

### Excluded Features (LEAK!)

**After-the-fact data:**
- `actual_eps` - Known only after report
- `eps_delta` - Actual - estimate (leak!)
- `elo_after`, `elo_change` - Post-report Elo updates
- `K_adaptive` - Used in elo_after calculation

**Report DAY price (CRITICAL LEAK!):**
- `price_at_report` - Price ON report day (not yet known!)
- `price_change_1m_pct` - Uses price_at_report in calculation!
- `price_change_3m_pct` - Uses price_at_report in calculation!

**Current quarter metrics (LEAK!):**
- `total_revenue_yoy_growth` - Current quarter (report needed!)
- `actual_eps_yoy_growth` - Current quarter
- Only `*_lag1` versions are safe (previous quarter)

### Safe Features ✅

**Price (BEFORE report):**
- `price_1m_before`, `price_3m_before` - Absolute prices before report

**Estimates & Revisions:**
- `eps_estimate_average`, `eps_estimate_high`, `eps_estimate_low`
- `eps_estimate_analyst_count`
- `eps_estimate_average_X_days_ago` (7, 30, 60, 90)
- `eps_estimate_revision_*` (up/down, 7/30 days)
- `revenue_estimate_*` (average, high, low, analyst_count)

**Historical Performance:**
- `elo_before`, `elo_decay`, `elo_momentum`, `elo_vol_4q`

**Previous Quarter Metrics (lag1):**
- `total_revenue_yoy_growth_lag1`
- `actual_eps_yoy_growth_lag1`
- `ebitda_yoy_growth_lag1`
- `operating_income_yoy_growth_lag1`
- `gross_margin_yoy_change_lag1`, `operating_margin_yoy_change_lag1`

## Train/Test Split

**Temporal Split (35% train / 65% test):**
- Sort all data by date (oldest to newest)
- First 35% → Training set
- Last 65% → Test set
- Maintains temporal order (no future leakage)

**Example with ~14,000 observations:**
- Train: ~4,900 observations (oldest quarters: 2017-2021)
- Test: ~9,100 observations (newest quarters: 2022-2025)

## Model Performance

Trained models:
- Random Forest (best overall: 66.4% avg accuracy on company-specific)
- XGBoost
- Logistic Regression

## Files

```
data/
  combined_data.csv          # All 414 companies merged
  X_train_global.csv         # Training features
  y_train_global.csv         # Training target
  X_test_global.csv          # Test features
  y_test_global.csv          # Test target

models/
  global_rf_model.pkl        # Random Forest
  global_xgb_model.pkl       # XGBoost
  global_lr_model.pkl        # Logistic Regression
  global_preprocessor.pkl    # Scaler + Imputer

results/
  model_performance.csv      # Metrics
  feature_importance_rf.png
  confusion_matrix.png
  evaluation_report.txt
  leak_verification.txt      # Verification results
```

## Usage

Train:
```bash
python train_global_model.py
```

Predict (for Polymarket backtest):
```python
import joblib
model = joblib.load('models/global_rf_model.pkl')
preprocessor = joblib.load('models/global_preprocessor.pkl')

# Predict for any company
features = prepare_features(ticker, quarter)
X = preprocessor.transform([features])
prediction = model.predict(X)[0]  # 0=miss, 1=beat
```

