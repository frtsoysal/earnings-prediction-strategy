"""
Data Preparation Script for EPS Beat Prediction
Loads, cleans, and prepares training/test data with temporal split
Usage: python3 prepare_data.py --symbol IBM
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_fundamentals_paths as get_paths, ensure_fundamentals_directories as ensure_directories

# Parse command line arguments
parser = argparse.ArgumentParser(description='Prepare data for ML pipeline')
parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., IBM, TSLA)')
args = parser.parse_args()

symbol = args.symbol.upper()
paths = ensure_directories(symbol)

print("=" * 80)
print(f"EPS BEAT PREDICTION - DATA PREPARATION - {symbol}")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv(paths['raw_csv'])
print(f"   Initial rows: {len(df)}")

# ============================================================================
# 2. FILTER DATA
# ============================================================================
print("\n2. Filtering data...")

# Keep only quarterly rows (exclude 'fiscal year')
df_filtered = df[~df['horizon'].str.contains('fiscal year', case=False, na=False)].copy()
print(f"   After removing fiscal year rows: {len(df_filtered)}")

# Remove future quarters (date >= '2025-12-31') - no actual_eps yet
df_filtered = df_filtered[df_filtered['date'] < '2025-12-31'].copy()
print(f"   After removing future quarters: {len(df_filtered)}")

# Remove rows where actual_eps is NaN (can't train without target)
df_filtered = df_filtered[df_filtered['actual_eps'].notna()].copy()
print(f"   After removing NaN actual_eps: {len(df_filtered)}")

# ============================================================================
# 3. DEFINE TARGET AND FEATURES
# ============================================================================
print("\n3. Defining target and features...")

# Target variable
y = df_filtered['eps_beat'].astype(int)
print(f"   Target distribution: {y.value_counts().to_dict()}")
print(f"   Class balance: {y.mean():.2%} beat rate")

# Features to EXCLUDE (leakage risk - computed after the fact)
leakage_cols = ['actual_eps', 'eps_beat', 'eps_delta', 
                'elo_after', 'elo_change', 'K_adaptive', 
                'date', 'horizon', 'reported_date', 'price_at_report']

# Features to USE (available before earnings report)
feature_cols = [
    # Price momentum
    'price_1m_before', 'price_3m_before', 
    'price_change_1m_pct', 'price_change_3m_pct',
    # EPS estimates
    'eps_estimate_average', 'eps_estimate_high', 'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago', 'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago', 'eps_estimate_average_90_days_ago',
    # Analyst revisions
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    # Revenue estimates
    'revenue_estimate_average', 'revenue_estimate_high', 'revenue_estimate_low',
    'revenue_estimate_analyst_count',
    # Elo metrics (historical performance)
    'elo_before', 'elo_decay', 'elo_momentum', 'elo_vol_4q',
    # Historical growth metrics (lag-1: previous quarter's performance)
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

print(f"   Total features: {len(feature_cols)}")

# ============================================================================
# 4. HANDLE MISSING VALUES
# ============================================================================
print("\n4. Analyzing missing values...")

X = df_filtered[feature_cols].copy()
missing_stats = {}

for col in X.columns:
    missing_count = X[col].isna().sum()
    missing_pct = (missing_count / len(X)) * 100
    missing_stats[col] = {
        'count': int(missing_count),
        'percentage': round(missing_pct, 2)
    }
    if missing_count > 0:
        print(f"   {col}: {missing_count} ({missing_pct:.1f}%)")

# Handle missing values based on feature type
print("\n   Handling missing values...")

# Revision features: NaN means "no revision" → fill with 0
revision_cols = [col for col in feature_cols if 'revision' in col]
for col in revision_cols:
    X[col].fillna(0, inplace=True)
print(f"   - Filled {len(revision_cols)} revision features with 0")

# For remaining NaNs, we'll let the imputer in train_model.py handle them
# but report them here
remaining_nan = X.isna().sum().sum()
print(f"   - Remaining NaN values: {remaining_nan} (will be handled by imputer)")

# ============================================================================
# 5. TEMPORAL TRAIN/TEST SPLIT
# ============================================================================
print("\n5. Creating temporal train/test split...")

dates = pd.to_datetime(df_filtered['date'])

# Train: All quarters before 2023
train_mask = dates < '2023-01-01'
test_mask = dates >= '2023-01-01'

X_train = X[train_mask].copy()
y_train = y[train_mask].copy()
X_test = X[test_mask].copy()
y_test = y[test_mask].copy()

train_dates = dates[train_mask]
test_dates = dates[test_mask]

print(f"\n   Train set:")
print(f"   - Size: {len(X_train)} samples")
print(f"   - Date range: {train_dates.min()} to {train_dates.max()}")
print(f"   - Class balance: {y_train.mean():.2%} beat rate")
print(f"   - Class distribution: {y_train.value_counts().to_dict()}")

print(f"\n   Test set:")
print(f"   - Size: {len(X_test)} samples")
print(f"   - Date range: {test_dates.min()} to {test_dates.max()}")
print(f"   - Class balance: {y_test.mean():.2%} beat rate")
print(f"   - Class distribution: {y_test.value_counts().to_dict()}")

# ============================================================================
# 6. SAVE OUTPUTS
# ============================================================================
print("\n6. Saving prepared data...")

# Save train/test data
X_train.to_csv(paths['X_train'], index=False)
y_train.to_csv(paths['y_train'], index=False, header=['eps_beat'])
X_test.to_csv(paths['X_test'], index=False)
y_test.to_csv(paths['y_test'], index=False, header=['eps_beat'])

# Save feature names
with open(paths['features'], 'w') as f:
    f.write('\n'.join(feature_cols))

# Save metadata
data_info = {
    'symbol': symbol,
    'total_samples': len(df_filtered),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_date_range': {
        'start': str(train_dates.min()),
        'end': str(train_dates.max())
    },
    'test_date_range': {
        'start': str(test_dates.min()),
        'end': str(test_dates.max())
    },
    'train_class_balance': {
        'beat_rate': float(y_train.mean()),
        'class_0': int((y_train == 0).sum()),
        'class_1': int((y_train == 1).sum())
    },
    'test_class_balance': {
        'beat_rate': float(y_test.mean()),
        'class_0': int((y_test == 0).sum()),
        'class_1': int((y_test == 1).sum())
    },
    'feature_count': len(feature_cols),
    'missing_value_stats': missing_stats
}

with open(paths['data_info'], 'w') as f:
    json.dump(data_info, f, indent=2)

print("\n   Saved files:")
print(f"   - {paths['X_train']}")
print(f"   - {paths['y_train']}")
print(f"   - {paths['X_test']}")
print(f"   - {paths['y_test']}")
print(f"   - {paths['features']}")
print(f"   - {paths['data_info']}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)

