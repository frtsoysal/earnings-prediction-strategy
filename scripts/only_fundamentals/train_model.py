"""
Model Training Script for EPS Beat Prediction
Trains Random Forest, XGBoost, and Logistic Regression with hyperparameter tuning
Usage: python3 train_model.py --symbol IBM
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import argparse
import sys
import os
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_fundamentals_paths as get_paths, ensure_fundamentals_directories as ensure_directories

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train ML models')
parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., IBM, TSLA)')
args = parser.parse_args()

symbol = args.symbol.upper()
paths = ensure_directories(symbol)

print("=" * 80)
print(f"EPS BEAT PREDICTION - MODEL TRAINING - {symbol}")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPARED DATA
# ============================================================================
print("\n1. Loading prepared data...")
X_train = pd.read_csv(paths['X_train'])
y_train = pd.read_csv(paths['y_train'])['eps_beat']

print(f"   Train samples: {len(X_train)}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Class balance: {y_train.value_counts().to_dict()}")

# ============================================================================
# 2. CREATE PREPROCESSING PIPELINE
# ============================================================================
print("\n2. Creating preprocessing pipeline...")

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

print("   Pipeline: Imputer (median) → StandardScaler")

# ============================================================================
# 3. MODEL 1: RANDOM FOREST
# ============================================================================
print("\n3. Training Random Forest...")

# Calculate class weights for imbalance
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts else 1

rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"   Hyperparameter search space: {len(rf_param_dist)} parameters")
print(f"   Max iterations: 20")

rf_search = RandomizedSearchCV(
    rf_model,
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit with preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
rf_search.fit(X_train_processed, y_train)

print(f"\n   Best parameters: {rf_search.best_params_}")
print(f"   Best CV ROC-AUC score: {rf_search.best_score_:.4f}")

rf_best = rf_search.best_estimator_

# ============================================================================
# 4. MODEL 2: XGBOOST
# ============================================================================
print("\n4. Training XGBoost...")

xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

print(f"   Hyperparameter search space: {len(xgb_param_dist)} parameters")
print(f"   Max iterations: 20")

xgb_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train_processed, y_train)

print(f"\n   Best parameters: {xgb_search.best_params_}")
print(f"   Best CV ROC-AUC score: {xgb_search.best_score_:.4f}")

xgb_best = xgb_search.best_estimator_

# ============================================================================
# 5. MODEL 3: LOGISTIC REGRESSION
# ============================================================================
print("\n5. Training Logistic Regression...")

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

lr_param_dist = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

print(f"   Hyperparameter search space: {len(lr_param_dist)} parameters")
print(f"   Max iterations: 10")

lr_search = RandomizedSearchCV(
    lr_model,
    param_distributions=lr_param_dist,
    n_iter=10,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

lr_search.fit(X_train_processed, y_train)

print(f"\n   Best parameters: {lr_search.best_params_}")
print(f"   Best CV ROC-AUC score: {lr_search.best_score_:.4f}")

lr_best = lr_search.best_estimator_

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n6. Saving trained models...")

joblib.dump(rf_best, paths['rf_model'])
print(f"   ✓ Saved {paths['rf_model']}")

joblib.dump(xgb_best, paths['xgb_model'])
print(f"   ✓ Saved {paths['xgb_model']}")

joblib.dump(lr_best, paths['lr_model'])
print(f"   ✓ Saved {paths['lr_model']}")

joblib.dump(preprocessor, paths['preprocessor'])
print(f"   ✓ Saved {paths['preprocessor']}")

# Save training results
training_results = {
    'symbol': symbol,
    'random_forest': {
        'best_params': rf_search.best_params_,
        'best_cv_score': float(rf_search.best_score_),
        'cv_mean_scores': [float(x) for x in rf_search.cv_results_['mean_test_score']],
        'cv_std_scores': [float(x) for x in rf_search.cv_results_['std_test_score']]
    },
    'xgboost': {
        'best_params': xgb_search.best_params_,
        'best_cv_score': float(xgb_search.best_score_),
        'cv_mean_scores': [float(x) for x in xgb_search.cv_results_['mean_test_score']],
        'cv_std_scores': [float(x) for x in xgb_search.cv_results_['std_test_score']]
    },
    'logistic_regression': {
        'best_params': lr_search.best_params_,
        'best_cv_score': float(lr_search.best_score_),
        'cv_mean_scores': [float(x) for x in lr_search.cv_results_['mean_test_score']],
        'cv_std_scores': [float(x) for x in lr_search.cv_results_['std_test_score']]
    }
}

with open(paths['training_results'], 'w') as f:
    json.dump(training_results, f, indent=2)

print(f"   ✓ Saved {paths['training_results']}")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING SUMMARY")
print("=" * 80)
print(f"\nRandom Forest:")
print(f"  Best CV ROC-AUC: {rf_search.best_score_:.4f}")
print(f"  Best params: {rf_search.best_params_}")

print(f"\nXGBoost:")
print(f"  Best CV ROC-AUC: {xgb_search.best_score_:.4f}")
print(f"  Best params: {xgb_search.best_params_}")

print(f"\nLogistic Regression:")
print(f"  Best CV ROC-AUC: {lr_search.best_score_:.4f}")
print(f"  Best params: {lr_search.best_params_}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

