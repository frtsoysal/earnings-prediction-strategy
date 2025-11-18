#!/usr/bin/env python3
"""
Time Series Cross-Validation Training
Walk-forward validation: Train on [2017-202X], Test on [202X+1]

Usage: python3 scripts/train_model_timeseries.py --symbol TSLA
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import argparse
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_paths, ensure_directories

# Parse arguments
parser = argparse.ArgumentParser(description='Train models with TimeSeriesSplit')
parser.add_argument('--symbol', required=True, help='Stock symbol')
parser.add_argument('--n-splits', type=int, default=4, help='Number of time series splits')
args = parser.parse_args()

symbol = args.symbol.upper()
n_splits = args.n_splits
paths = ensure_directories(symbol)

print("=" * 80)
print(f"TIME SERIES CV TRAINING - {symbol}")
print("=" * 80)

# ============================================================================
# 1. LOAD ALL DATA (don't split yet)
# ============================================================================
print("\n1. Loading full dataset...")
df = pd.read_csv(paths['raw_csv'])

# Filter same as prepare_data.py
df_filtered = df[~df['horizon'].str.contains('fiscal year', case=False, na=False)].copy()
df_filtered = df_filtered[df_filtered['date'] < '2025-12-31'].copy()
df_filtered = df_filtered[df_filtered['actual_eps'].notna()].copy()

# Sort by date (CRITICAL for time series)
df_filtered = df_filtered.sort_values('date').reset_index(drop=True)

print(f"   Total samples: {len(df_filtered)}")
print(f"   Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n2. Preparing features...")

# Target
y = df_filtered['eps_beat'].astype(int)

# Features (same as prepare_data.py)
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
    # Historical growth metrics (lag-1)
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

X = df_filtered[feature_cols].copy()
dates = pd.to_datetime(df_filtered['date'])

# Handle missing values
revision_cols = [col for col in feature_cols if 'revision' in col]
for col in revision_cols:
    X[col].fillna(0, inplace=True)

print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 3. TIME SERIES CROSS-VALIDATION
# ============================================================================
print(f"\n3. Running TimeSeriesSplit with {n_splits} folds...")

# Create results directory
ts_results_dir = os.path.join(paths['results_dir'], 'timeseries_cv')
os.makedirs(ts_results_dir, exist_ok=True)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Time Series Split
tscv = TimeSeriesSplit(n_splits=n_splits)

# Store results for all folds
all_fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}/{n_splits}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_dates, test_dates = dates.iloc[train_idx], dates.iloc[test_idx]
    
    print(f"\nTrain: {train_dates.min().date()} to {train_dates.max().date()} ({len(X_train)} samples)")
    print(f"Test:  {test_dates.min().date()} to {test_dates.max().date()} ({len(X_test)} samples)")
    print(f"Train beat rate: {y_train.mean():.2%}")
    print(f"Test beat rate:  {y_test.mean():.2%}")
    
    # Create fold directory
    fold_dir = os.path.join(ts_results_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train models
    fold_results = {
        'fold': fold_idx,
        'train_start': str(train_dates.min().date()),
        'train_end': str(train_dates.max().date()),
        'test_start': str(test_dates.min().date()),
        'test_end': str(test_dates.max().date()),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_beat_rate': float(y_train.mean()),
        'test_beat_rate': float(y_test.mean()),
        'models': {}
    }
    
    # Calculate class weights (handle single class case)
    class_counts = y_train.value_counts()
    if 0 in class_counts.index and 1 in class_counts.index:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0
    
    # Train each model
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=20, 
            class_weight='balanced', random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        ),
        'LogisticRegression': LogisticRegression(
            C=1, penalty='l1', solver='liblinear',
            class_weight='balanced', random_state=42
        )
    }
    
    # Check if we have at least 2 classes (most models need this)
    has_both_classes = len(np.unique(y_train)) >= 2
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Skip models if only one class in training (they can't handle this)
        if not has_both_classes:
            print(f"  Skipping {model_name} (only one class in training set)")
            fold_results['models'][model_name] = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'roc_auc': None
            }
            continue
        
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        
        # Get probability for positive class (handle single class case)
        proba = model.predict_proba(X_test_processed)
        if proba.shape[1] > 1:
            y_proba = proba[:, 1]
        else:
            y_proba = proba[:, 0]  # Only one class was seen in training
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        # ROC-AUC (if both classes present)
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        else:
            metrics['roc_auc'] = None
        
        fold_results['models'][model_name] = metrics
        
        roc_str = f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else "N/A"
        print(f"  Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {roc_str}")
        
        # Save model
        model_path = os.path.join(fold_dir, f'{model_name.lower()}_model.pkl')
        joblib.dump(model, model_path)
    
    # Save fold results
    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    all_fold_results.append(fold_results)

# ============================================================================
# 4. AGGREGATE RESULTS
# ============================================================================
print(f"\n{'='*80}")
print("AGGREGATING RESULTS ACROSS FOLDS")
print(f"{'='*80}\n")

# Create aggregate DataFrame
aggregate_data = []
for fold_res in all_fold_results:
    for model_name, metrics in fold_res['models'].items():
        aggregate_data.append({
            'Fold': fold_res['fold'],
            'Model': model_name,
            'Test_Period': f"{fold_res['test_start']} to {fold_res['test_end']}",
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })

aggregate_df = pd.DataFrame(aggregate_data)
aggregate_df.to_csv(os.path.join(ts_results_dir, 'aggregate_results.csv'), index=False)

# Calculate average performance per model
print("Average Performance Across Folds:\n")
summary = aggregate_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].mean()
print(summary)
print()

# Save summary
summary.to_csv(os.path.join(ts_results_dir, 'model_summary.csv'))

# Save comprehensive report
with open(os.path.join(ts_results_dir, 'timeseries_cv_report.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write(f"TIME SERIES CROSS-VALIDATION REPORT - {symbol}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Strategy: Walk-Forward Validation\n")
    f.write(f"Number of Folds: {n_splits}\n\n")
    
    for fold_res in all_fold_results:
        f.write(f"Fold {fold_res['fold']}:\n")
        f.write(f"  Train: {fold_res['train_start']} to {fold_res['train_end']} ({fold_res['train_samples']} samples)\n")
        f.write(f"  Test:  {fold_res['test_start']} to {fold_res['test_end']} ({fold_res['test_samples']} samples)\n")
        f.write(f"  Train beat rate: {fold_res['train_beat_rate']:.2%}\n")
        f.write(f"  Test beat rate:  {fold_res['test_beat_rate']:.2%}\n\n")
        
        for model_name, metrics in fold_res['models'].items():
            f.write(f"    {model_name}:\n")
            for metric_name, value in metrics.items():
                if value is not None:
                    f.write(f"      {metric_name}: {value:.4f}\n")
                else:
                    f.write(f"      {metric_name}: N/A\n")
        f.write("\n")
    
    f.write("\nAVERAGE PERFORMANCE:\n")
    f.write(summary.to_string())

print(f"\n✅ All results saved to: {ts_results_dir}")
print("\nFiles created:")
print(f"  - aggregate_results.csv")
print(f"  - model_summary.csv")
print(f"  - timeseries_cv_report.txt")
print(f"  - fold_1/ to fold_{n_splits}/ (models + metrics)")

print("\n" + "=" * 80)
print("TIME SERIES CV COMPLETE!")
print("=" * 80)

