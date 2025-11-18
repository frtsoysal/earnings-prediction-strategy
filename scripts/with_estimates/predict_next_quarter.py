#!/usr/bin/env python3
"""
Predict Next Quarter EPS Beat/Miss
Load trained models and predict future quarters
Save predictions to central predictions.csv

Usage: python3 scripts/with_estimates/predict_next_quarter.py --symbol IBM
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_paths

# Parse arguments
parser = argparse.ArgumentParser(description='Predict next quarter EPS beat')
parser.add_argument('--symbol', required=True, help='Stock symbol')
args = parser.parse_args()

symbol = args.symbol.upper()
paths = get_paths(symbol)

# Central predictions file
PREDICTIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'predictions.csv')

print("=" * 80)
print(f"NEXT QUARTER PREDICTION - {symbol}")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND FIND NEXT QUARTER
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv(paths['raw_csv'])

# Filter for future quarters (no actual_eps yet)
future_quarters = df[
    (~df['horizon'].str.contains('fiscal year', case=False, na=False)) &
    (df['actual_eps'].isna() | (df['actual_eps'] == ''))
].copy()

if len(future_quarters) == 0:
    print(f"   ⚠️  No future quarters found for {symbol}")
    sys.exit(0)

# Get the closest future quarter
future_quarters['date'] = pd.to_datetime(future_quarters['date'])
future_quarters = future_quarters.sort_values('date')
next_quarter = future_quarters.iloc[0]

print(f"   Next Quarter: {next_quarter['date'].date()}")
print(f"   Horizon: {next_quarter['horizon']}")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n2. Preparing features...")

# Feature columns (same as prepare_data.py)
feature_cols = [
    'price_1m_before', 'price_3m_before', 
    'price_change_1m_pct', 'price_change_3m_pct',
    'eps_estimate_average', 'eps_estimate_high', 'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago', 'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago', 'eps_estimate_average_90_days_ago',
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    'revenue_estimate_average', 'revenue_estimate_high', 'revenue_estimate_low',
    'revenue_estimate_analyst_count',
    'elo_before', 'elo_decay', 'elo_momentum', 'elo_vol_4q',
    'total_revenue_yoy_growth_lag1',
    'total_revenue_qoq_growth_lag1',
    'total_revenue_ttm_yoy_growth_lag1',
    'actual_eps_yoy_growth_lag1',
    'actual_eps_qoq_growth_lag1',
    'ebitda_yoy_growth_lag1',
    'operating_income_yoy_growth_lag1',
    'gross_margin_yoy_change_lag1',
    'operating_margin_yoy_change_lag1'
]

X_next = next_quarter[feature_cols].to_frame().T

# Handle missing values (same as training)
revision_cols = [col for col in feature_cols if 'revision' in col]
for col in revision_cols:
    X_next[col].fillna(0, inplace=True)

print(f"   Features extracted: {len(feature_cols)}")

# ============================================================================
# 3. LOAD MODELS AND PREDICT
# ============================================================================
print("\n3. Loading models and making predictions...")

predictions = {
    'symbol': symbol,
    'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'next_quarter_date': next_quarter['date'].strftime('%Y-%m-%d'),
    'eps_estimate': float(next_quarter.get('eps_estimate_average', np.nan)) if next_quarter.get('eps_estimate_average') else np.nan
}

# Load preprocessor
try:
    preprocessor = joblib.load(paths['preprocessor'])
    X_processed = preprocessor.transform(X_next)
except Exception as e:
    print(f"   ⚠️  Error loading preprocessor: {e}")
    sys.exit(1)

# Load and predict with each model
models_to_load = {
    'RF': paths['rf_model'],
    'XGB': paths['xgb_model'],
    'LR': paths['lr_model']
}

for model_name, model_path in models_to_load.items():
    try:
        model = joblib.load(model_path)
        
        # Prediction
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0]
        
        # Store results
        predictions[f'{model_name}_prediction'] = 'BEAT' if pred == 1 else 'MISS'
        predictions[f'{model_name}_beat_prob'] = round(proba[1] * 100, 2) if len(proba) > 1 else 0.0
        
        print(f"   {model_name}: {predictions[f'{model_name}_prediction']} "
              f"(prob: {predictions[f'{model_name}_beat_prob']:.1f}%)")
        
    except FileNotFoundError:
        print(f"   ⚠️  {model_name} model not found")
        predictions[f'{model_name}_prediction'] = 'N/A'
        predictions[f'{model_name}_beat_prob'] = np.nan

# ============================================================================
# 4. LOAD TIME SERIES MODELS (if available)
# ============================================================================
print("\n4. Loading Time Series CV models (latest fold)...")

ts_dir = paths['timeseries_cv_dir']
if os.path.exists(ts_dir):
    # Find latest fold
    folds = [d for d in os.listdir(ts_dir) if d.startswith('fold_')]
    if folds:
        latest_fold = sorted(folds)[-1]
        fold_path = os.path.join(ts_dir, latest_fold)
        
        for model_name in ['randomforest', 'xgboost', 'logisticregression']:
            model_file = os.path.join(fold_path, f'{model_name}_model.pkl')
            try:
                model = joblib.load(model_file)
                pred = model.predict(X_processed)[0]
                proba = model.predict_proba(X_processed)[0]
                
                short_name = {'randomforest': 'TS_RF', 'xgboost': 'TS_XGB', 'logisticregression': 'TS_LR'}[model_name]
                predictions[f'{short_name}_prediction'] = 'BEAT' if pred == 1 else 'MISS'
                predictions[f'{short_name}_beat_prob'] = round(proba[1] * 100, 2) if len(proba) > 1 else 0.0
                
                print(f"   {short_name}: {predictions[f'{short_name}_prediction']} "
                      f"(prob: {predictions[f'{short_name}_beat_prob']:.1f}%)")
            except:
                pass

# ============================================================================
# 5. ENSEMBLE PREDICTION
# ============================================================================
print("\n5. Creating ensemble prediction...")

# Collect all valid predictions
beat_votes = []
probs = []

for key in predictions:
    if key.endswith('_prediction') and predictions[key] in ['BEAT', 'MISS']:
        if predictions[key] == 'BEAT':
            beat_votes.append(1)
        else:
            beat_votes.append(0)
    
    if key.endswith('_beat_prob') and not pd.isna(predictions[key]):
        probs.append(predictions[key])

if beat_votes:
    ensemble_pred = 'BEAT' if np.mean(beat_votes) >= 0.5 else 'MISS'
    confidence = np.mean(probs) if probs else 50.0
    
    predictions['ensemble_prediction'] = ensemble_pred
    predictions['confidence'] = round(confidence, 2)
    predictions['models_count'] = len(beat_votes)
    
    print(f"   Ensemble: {ensemble_pred} (confidence: {confidence:.1f}%)")
    print(f"   Based on {len(beat_votes)} models")
else:
    predictions['ensemble_prediction'] = 'N/A'
    predictions['confidence'] = np.nan
    predictions['models_count'] = 0

# ============================================================================
# 6. SAVE TO CENTRAL CSV
# ============================================================================
print("\n6. Saving to central predictions file...")

# Convert to DataFrame
pred_df = pd.DataFrame([predictions])

# Load or create predictions file
if os.path.exists(PREDICTIONS_FILE):
    existing = pd.read_csv(PREDICTIONS_FILE)
    
    # Remove old prediction for same symbol if exists
    existing = existing[existing['symbol'] != symbol]
    
    # Append new prediction
    updated = pd.concat([existing, pred_df], ignore_index=True)
else:
    updated = pred_df

# Sort by symbol
updated = updated.sort_values('symbol')

# Save
updated.to_csv(PREDICTIONS_FILE, index=False)

print(f"   ✓ Saved to {PREDICTIONS_FILE}")
print(f"   Total predictions in file: {len(updated)}")

print("\n" + "=" * 80)
print("PREDICTION COMPLETE!")
print("=" * 80)


