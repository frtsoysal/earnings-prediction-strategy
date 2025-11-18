#!/usr/bin/env python3
"""
Global ML Model Training - Cross-Company EPS Beat Prediction
=============================================================

Trains a single model on ALL S&P 500 companies combined.
Strict temporal split (35% train / 65% test) with comprehensive leak prevention.

Output:
    data/combined_data.csv
    data/X_train_global.csv, y_train_global.csv
    data/X_test_global.csv, y_test_global.csv
    models/global_rf_model.pkl
    models/global_xgb_model.pkl
    models/global_lr_model.pkl
    models/global_preprocessor.pkl
    results/model_performance.csv
    results/evaluation_report.txt
    results/leak_verification.txt
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
import warnings
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("=" * 80)
print("GLOBAL ML MODEL TRAINING - S&P 500 CROSS-COMPANY PREDICTION")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD & COMBINE DATA (LEAK CHECK #1)
# =============================================================================

print(f"\n📊 STEP 1/7: Loading and combining all companies...")

csv_pattern = '../../data/raw/*_earnings_with_q4.csv'
csv_files = glob.glob(csv_pattern)
print(f"   • Found {len(csv_files)} CSV files")

all_dfs = []
for fp in csv_files:
    filename = os.path.basename(fp)
    symbol = filename.replace('_earnings_with_q4.csv', '')
    
    try:
        df = pd.read_csv(fp, low_memory=False)
        df['symbol'] = symbol
        all_dfs.append(df)
    except Exception as e:
        print(f"   ⚠️  {symbol}: {str(e)[:50]}")
        continue

print(f"   • Loaded {len(all_dfs)} companies")

# Combine all companies
combined = pd.concat(all_dfs, ignore_index=True)
print(f"   • Total rows before filtering: {len(combined):,}")

# Filter: ONLY quarterly (exclude fiscal year)
combined = combined[~combined['horizon'].str.contains('fiscal year', case=False, na=False)].copy()
print(f"   • After removing fiscal year: {len(combined):,}")

# Filter: ONLY historical (has actual_eps - means it's not future)
combined = combined[combined['actual_eps'].notna()].copy()
print(f"   • After removing future quarters: {len(combined):,}")

# Filter: Remove any dates >= 2026 (extra safety)
combined = combined[combined['date'] < '2026-01-01'].copy()
print(f"   • After date safety check: {len(combined):,}")

# Ensure eps_beat exists
combined = combined[combined['eps_beat'].notna()].copy()
combined['eps_beat'] = combined['eps_beat'].astype(int)
print(f"   • Final dataset: {len(combined):,} observations from {combined['symbol'].nunique()} companies")

# Save combined data
combined.to_csv('data/combined_data.csv', index=False)
print(f"   ✓ Saved: data/combined_data.csv")

# =============================================================================
# STEP 2: DEFINE SAFE FEATURES (LEAK CHECK #2)
# =============================================================================

print(f"\n🔒 STEP 2/7: Defining safe features (LEAK PREVENTION)...")

# LEAK COLUMNS - MUST BE EXCLUDED
LEAK_COLS = [
    # After-the-fact data
    'actual_eps',              # Known only after report
    'eps_beat',                # Target variable
    'eps_delta',               # actual - estimate (LEAK!)
    'elo_after',               # Post-report Elo
    'elo_change',              # Elo change (LEAK!)
    'K_adaptive',              # Used in elo_after
    
    # Report DAY price (CRITICAL LEAK!)
    'price_at_report',         # Price ON report day
    'price_change_1m_pct',     # Uses price_at_report! (LEAK!)
    'price_change_3m_pct',     # Uses price_at_report! (LEAK!)
    
    # Current quarter growth (LEAK - needs actual results!)
    'total_revenue_yoy_growth',     # Current Q (LEAK!)
    'total_revenue_qoq_growth',     # Current Q (LEAK!)
    'total_revenue_ttm_yoy_growth', # Current Q (LEAK!)
    'actual_eps_yoy_growth',        # Current Q (LEAK!)
    'actual_eps_qoq_growth',        # Current Q (LEAK!)
    'ebitda_yoy_growth',            # Current Q (LEAK!)
    'operating_income_yoy_growth',  # Current Q (LEAK!)
    'free_cash_flow_yoy_growth',    # Current Q (LEAK!)
    'gross_margin_yoy_change',      # Current Q (LEAK!)
    'operating_margin_yoy_change',  # Current Q (LEAK!)
    'net_margin_yoy_change',        # Current Q (LEAK!)
    
    # Any non-lag growth metrics
    'total_revenue_qoq_2q_avg',
    'total_revenue_qoq_4q_avg',
    'actual_eps_qoq_2q_avg',
    'actual_eps_qoq_4q_avg',
    'actual_eps_ttm_yoy_growth',
    'ebitda_qoq_growth',
    'ebitda_ttm_yoy_growth',
    'ebitda_qoq_2q_avg',
    'ebitda_qoq_4q_avg',
    'operating_income_qoq_growth',
    'operating_income_ttm_yoy_growth',
    'operating_income_qoq_2q_avg',
    'operating_income_qoq_4q_avg',
    'free_cash_flow_qoq_growth',
    'free_cash_flow_ttm_yoy_growth',
    'free_cash_flow_qoq_2q_avg',
    'free_cash_flow_qoq_4q_avg',
    
    # Metadata
    'date', 'horizon', 'reported_date', 'symbol'
]

# SAFE FEATURES - Available BEFORE earnings report
SAFE_FEATURES = [
    # Price levels BEFORE report (absolute values - OK!)
    'price_1m_before',
    'price_3m_before',
    
    # EPS estimates
    'eps_estimate_average',
    'eps_estimate_high',
    'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago',
    'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago',
    'eps_estimate_average_90_days_ago',
    
    # Analyst revisions
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    
    # Revenue estimates
    'revenue_estimate_average',
    'revenue_estimate_high',
    'revenue_estimate_low',
    'revenue_estimate_analyst_count',
    
    # Historical Elo (BEFORE current quarter)
    'elo_before',
    'elo_decay',
    'elo_momentum',
    'elo_vol_4q',
    
    # LAGGED growth metrics (PREVIOUS quarter - SAFE!)
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

print(f"   • Safe features defined: {len(SAFE_FEATURES)}")
print(f"   • Leak columns to exclude: {len(LEAK_COLS)}")

# Verify safe features exist in data
available_features = [f for f in SAFE_FEATURES if f in combined.columns]
missing_features = [f for f in SAFE_FEATURES if f not in combined.columns]

if missing_features:
    print(f"   ⚠️  Missing features: {missing_features}")

SAFE_FEATURES = available_features
print(f"   ✓ Available safe features: {len(SAFE_FEATURES)}")

# =============================================================================
# STEP 3: TEMPORAL TRAIN/TEST SPLIT - 35% / 65% (LEAK CHECK #3)
# =============================================================================

print(f"\n📅 STEP 3/7: Creating temporal split (35% train / 65% test)...")

# Sort by date (CRITICAL for temporal split)
combined_sorted = combined.sort_values('date').reset_index(drop=True)
dates = pd.to_datetime(combined_sorted['date'])

# Calculate split point
total_samples = len(combined_sorted)
train_size = int(total_samples * 0.35)
split_idx = train_size

print(f"   • Total samples: {total_samples:,}")
print(f"   • Train size (35%): {train_size:,}")
print(f"   • Test size (65%): {total_samples - train_size:,}")

# Create train/test sets
train_data = combined_sorted.iloc[:split_idx].copy()
test_data = combined_sorted.iloc[split_idx:].copy()

train_dates = dates.iloc[:split_idx]
test_dates = dates.iloc[split_idx:]

# Extract features and target
X_train = train_data[SAFE_FEATURES].copy()
y_train = train_data['eps_beat'].copy()
X_test = test_data[SAFE_FEATURES].copy()
y_test = test_data['eps_beat'].copy()

# CRITICAL VERIFICATION
print(f"\n   📋 VERIFICATION:")
print(f"   Train date range: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"   Test date range:  {test_dates.min().date()} to {test_dates.max().date()}")

cutoff_date = train_dates.max()
first_test_date = test_dates.min()

assert first_test_date >= cutoff_date, f"LEAK! Test starts {first_test_date} before train ends {cutoff_date}"
print(f"   ✓ VERIFIED: No temporal overlap (cutoff: {cutoff_date.date()})")

# Class distribution
print(f"\n   Train class distribution: {y_train.value_counts().to_dict()}")
print(f"   Train beat rate: {y_train.mean():.2%}")
print(f"   Test class distribution: {y_test.value_counts().to_dict()}")
print(f"   Test beat rate: {y_test.mean():.2%}")

# Company distribution
print(f"\n   Train companies: {train_data['symbol'].nunique()}")
print(f"   Test companies: {test_data['symbol'].nunique()}")

# Save train/test data
X_train.to_csv('data/X_train_global.csv', index=False)
y_train.to_csv('data/y_train_global.csv', index=False, header=['eps_beat'])
X_test.to_csv('data/X_test_global.csv', index=False)
y_test.to_csv('data/y_test_global.csv', index=False, header=['eps_beat'])

print(f"   ✓ Saved train/test datasets")

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

print(f"\n🔧 STEP 4/7: Handling missing values...")

# Revision features: NaN means "no revision" → fill with 0
revision_cols = [c for c in SAFE_FEATURES if 'revision' in c]
X_train[revision_cols] = X_train[revision_cols].fillna(0)
X_test[revision_cols] = X_test[revision_cols].fillna(0)
print(f"   • Filled {len(revision_cols)} revision columns with 0")

# Check remaining NaN
nan_count_train = X_train.isna().sum().sum()
nan_count_test = X_test.isna().sum().sum()
print(f"   • Remaining NaN (train): {nan_count_train}")
print(f"   • Remaining NaN (test): {nan_count_test}")

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fit on train, transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"   ✓ Preprocessing pipeline created (median imputer + scaler)")

# =============================================================================
# STEP 5: TRAIN MODELS
# =============================================================================

print(f"\n🎯 STEP 5/7: Training models...")

# Calculate class weights
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1

print(f"   • Class imbalance ratio: {scale_pos_weight:.2f}")

models = {}

# Model 1: Random Forest
print(f"\n   Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf.fit(X_train_processed, y_train)
models['Random Forest'] = rf
print(f"   ✓ Random Forest trained")

# Model 2: XGBoost
print(f"\n   Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train_processed, y_train)
models['XGBoost'] = xgb
print(f"   ✓ XGBoost trained")

# Model 3: Logistic Regression
print(f"\n   Training Logistic Regression...")
lr = LogisticRegression(
    C=1,
    penalty='l2',
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_processed, y_train)
models['Logistic Regression'] = lr
print(f"   ✓ Logistic Regression trained")

# =============================================================================
# STEP 6: EVALUATE MODELS
# =============================================================================

print(f"\n📈 STEP 6/7: Evaluating models on test set...")

results = []

for model_name, model in models.items():
    print(f"\n   Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = np.nan
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Train_Size': len(X_train),
        'Test_Size': len(X_test)
    })
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      ROC-AUC: {roc_auc:.4f}")

# Save performance metrics
perf_df = pd.DataFrame(results)
perf_df.to_csv('results/model_performance.csv', index=False)
print(f"\n   ✓ Saved: results/model_performance.csv")

# =============================================================================
# STEP 7: SAVE MODELS
# =============================================================================

print(f"\n💾 STEP 7/7: Saving models...")

joblib.dump(rf, 'models/global_rf_model.pkl')
print(f"   ✓ models/global_rf_model.pkl")

joblib.dump(xgb, 'models/global_xgb_model.pkl')
print(f"   ✓ models/global_xgb_model.pkl")

joblib.dump(lr, 'models/global_lr_model.pkl')
print(f"   ✓ models/global_lr_model.pkl")

joblib.dump(preprocessor, 'models/global_preprocessor.pkl')
print(f"   ✓ models/global_preprocessor.pkl")

# =============================================================================
# LEAK VERIFICATION TESTS
# =============================================================================

print(f"\n🔍 LEAK VERIFICATION TESTS...")

verification_results = []

# Test 1: No leak columns in features
leak_in_features = [c for c in LEAK_COLS if c in X_train.columns]
test1 = len(leak_in_features) == 0
verification_results.append(('No leak columns in features', test1, leak_in_features if not test1 else 'PASS'))
print(f"   Test 1: No leak columns in features... {'✓ PASS' if test1 else '✗ FAIL: ' + str(leak_in_features)}")

# Test 2: eps_beat not in features
test2 = 'eps_beat' not in X_train.columns
verification_results.append(('Target not in features', test2, 'PASS' if test2 else 'FAIL'))
print(f"   Test 2: Target not in features... {'✓ PASS' if test2 else '✗ FAIL'}")

# Test 3: Temporal separation
test3 = cutoff_date < first_test_date
verification_results.append(('Temporal separation', test3, f"Train ends {cutoff_date.date()}, Test starts {first_test_date.date()}"))
print(f"   Test 3: Temporal separation... {'✓ PASS' if test3 else '✗ FAIL'}")

# Test 4: No price_change_*_pct columns
price_change_cols = [c for c in X_train.columns if 'price_change' in c]
test4 = len(price_change_cols) == 0
verification_results.append(('No price_change columns', test4, price_change_cols if not test4 else 'PASS'))
print(f"   Test 4: No price_change columns... {'✓ PASS' if test4 else '✗ FAIL: ' + str(price_change_cols)}")

# Test 5: Only lag features for growth metrics
non_lag_growth = [c for c in X_train.columns if ('growth' in c or 'change' in c) and 'lag' not in c]
test5 = len(non_lag_growth) == 0
verification_results.append(('Only lagged growth metrics', test5, non_lag_growth if not test5 else 'PASS'))
print(f"   Test 5: Only lagged growth metrics... {'✓ PASS' if test5 else '✗ FAIL: ' + str(non_lag_growth)}")

# Save verification results
with open('results/leak_verification.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DATA LEAK VERIFICATION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    for test_name, passed, details in verification_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        f.write(f"{test_name}: {status}\n")
        f.write(f"   Details: {details}\n\n")
    
    all_passed = all(r[1] for r in verification_results)
    f.write("=" * 80 + "\n")
    if all_passed:
        f.write("✅ ALL TESTS PASSED - NO DATA LEAKAGE DETECTED\n")
    else:
        f.write("❌ SOME TESTS FAILED - REVIEW REQUIRED\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ Saved: results/leak_verification.txt")

# Final check
all_passed = all(r[1] for r in verification_results)
if not all_passed:
    print(f"\n   ⚠️  WARNING: Some verification tests failed!")
else:
    print(f"\n   ✅ All verification tests passed!")

# =============================================================================
# COMPREHENSIVE REPORT
# =============================================================================

print(f"\n📄 Generating evaluation report...")

with open('results/evaluation_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GLOBAL ML MODEL - EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total observations: {total_samples:,}\n")
    f.write(f"Companies: {combined_sorted['symbol'].nunique()}\n")
    f.write(f"Date range: {dates.min().date()} to {dates.max().date()}\n\n")
    
    f.write("TRAIN/TEST SPLIT (Temporal)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Train: {len(X_train):,} samples ({len(X_train)/total_samples:.1%})\n")
    f.write(f"  Date range: {train_dates.min().date()} to {train_dates.max().date()}\n")
    f.write(f"  Beat rate: {y_train.mean():.2%}\n")
    f.write(f"  Companies: {train_data['symbol'].nunique()}\n\n")
    
    f.write(f"Test: {len(X_test):,} samples ({len(X_test)/total_samples:.1%})\n")
    f.write(f"  Date range: {test_dates.min().date()} to {test_dates.max().date()}\n")
    f.write(f"  Beat rate: {y_test.mean():.2%}\n")
    f.write(f"  Companies: {test_data['symbol'].nunique()}\n\n")
    
    f.write("MODEL PERFORMANCE (on test set)\n")
    f.write("-" * 80 + "\n")
    f.write(perf_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("FEATURE IMPORTANCE - TOP 20 (Random Forest)\n")
    f.write("-" * 80 + "\n")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    feature_names = np.array(SAFE_FEATURES)
    for i, idx in enumerate(indices, 1):
        f.write(f"{i:2d}. {feature_names[idx]:45s} {importances[idx]:.4f}\n")
    
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ Saved: results/evaluation_report.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "=" * 80)
print(f"✅ GLOBAL MODEL TRAINING COMPLETE")
print(f"=" * 80)

print(f"\nDataset:")
print(f"   • Total observations: {total_samples:,}")
print(f"   • Companies: {combined_sorted['symbol'].nunique()}")
print(f"   • Date range: {dates.min().date()} to {dates.max().date()}")

print(f"\nSplit (Temporal - 35%/65%):")
print(f"   • Train: {len(X_train):,} ({len(X_train)/total_samples:.1%})")
print(f"   • Test:  {len(X_test):,} ({len(X_test)/total_samples:.1%})")
print(f"   • Cutoff date: {cutoff_date.date()}")

print(f"\nBest Model Performance:")
best_model = perf_df.loc[perf_df['Accuracy'].idxmax()]
print(f"   • Model: {best_model['Model']}")
print(f"   • Accuracy: {best_model['Accuracy']:.4f}")
print(f"   • ROC-AUC: {best_model['ROC-AUC']:.4f}")

print(f"\nLeak Verification:")
if all_passed:
    print(f"   ✅ All tests passed - NO DATA LEAKAGE")
else:
    print(f"   ⚠️  Some tests failed - Review leak_verification.txt")

print(f"\nOutputs:")
print(f"   📁 Data: data/")
print(f"   🤖 Models: models/")
print(f"   📊 Results: results/")

print(f"\n" + "=" * 80)

