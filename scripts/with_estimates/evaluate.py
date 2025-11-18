"""
Model Evaluation Script for EPS Beat Prediction
Evaluates models, creates visualizations, and generates comprehensive reports
Usage: python3 evaluate.py --symbol IBM
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import sys
import os
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_paths, ensure_directories

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate ML models')
parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., IBM, TSLA)')
args = parser.parse_args()

symbol = args.symbol.upper()
paths = ensure_directories(symbol)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print(f"EPS BEAT PREDICTION - MODEL EVALUATION - {symbol}")
print("=" * 80)

# ============================================================================
# 1. LOAD MODELS AND TEST DATA
# ============================================================================
print("\n1. Loading models and test data...")

rf_model = joblib.load(paths['rf_model'])
preprocessor = joblib.load(paths['preprocessor'])

# Try loading optional models (may not exist if training failed)
try:
    xgb_model = joblib.load(paths['xgb_model'])
except FileNotFoundError:
    print("   ⚠️  XGBoost model not found (skipped during training)")
    xgb_model = None

try:
    lr_model = joblib.load(paths['lr_model'])
except FileNotFoundError:
    print("   ⚠️  Logistic Regression model not found (skipped during training)")
    lr_model = None

X_test = pd.read_csv(paths['X_test'])
y_test = pd.read_csv(paths['y_test'])['eps_beat']

print(f"   Test samples: {len(X_test)}")
print(f"   Class distribution: {y_test.value_counts().to_dict()}")

# Preprocess test data
X_test_processed = preprocessor.transform(X_test)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Logistic Regression': lr_model
}

# ============================================================================
# 2. GENERATE PREDICTIONS
# ============================================================================
print("\n2. Generating predictions...")

predictions = {}
probabilities = {}

for name, model in models.items():
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    predictions[name] = y_pred
    probabilities[name] = y_proba
    
    print(f"   {name}: {sum(y_pred)} predicted beats out of {len(y_pred)}")

# ============================================================================
# 3. CALCULATE METRICS
# ============================================================================
print("\n3. Calculating metrics...")

metrics_results = []

for name in models.keys():
    y_pred = predictions[name]
    y_proba = probabilities[name]
    
    # Calculate metrics (handle edge cases for single class)
    accuracy = accuracy_score(y_test, y_pred)
    
    # For precision, recall, f1: handle case where only one class exists
    try:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    except:
        precision = recall = f1 = np.nan
    
    # ROC-AUC requires both classes
    try:
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = np.nan
    except:
        roc_auc = np.nan
    
    metrics_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

metrics_df = pd.DataFrame(metrics_results)

print("\nMetrics Summary:")
print(metrics_df.to_string(index=False))

# ============================================================================
# 4. FEATURE IMPORTANCE (RF and XGBoost)
# ============================================================================
print("\n4. Creating feature importance plots...")

feature_names = X_test.columns

# Random Forest Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:20]

plt.barh(range(len(indices_rf)), importances_rf[indices_rf])
plt.yticks(range(len(indices_rf)), [feature_names[i] for i in indices_rf])
plt.xlabel('Feature Importance')
plt.title(f'Random Forest - Top 20 Feature Importances - {symbol}')
plt.tight_layout()
plt.savefig(f"{paths['results_dir']}/feature_importance_rf.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved {paths['results_dir']}/feature_importance_rf.png")

# XGBoost Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1][:20]

plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb])
plt.yticks(range(len(indices_xgb)), [feature_names[i] for i in indices_xgb])
plt.xlabel('Feature Importance')
plt.title(f'XGBoost - Top 20 Feature Importances - {symbol}')
plt.tight_layout()
plt.savefig(f"{paths['results_dir']}/feature_importance_xgb.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved {paths['results_dir']}/feature_importance_xgb.png")

# ============================================================================
# 5. CONFUSION MATRICES
# ============================================================================
print("\n5. Creating confusion matrices...")

for name in models.keys():
    y_pred = predictions[name]
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Miss', 'Beat'],
                yticklabels=['Miss', 'Beat'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    
    filename = f"{paths['results_dir']}/confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {filename}")

# ============================================================================
# 6. ROC CURVES
# ============================================================================
print("\n6. Creating ROC curves...")

fig, ax = plt.subplots(figsize=(8, 6))

if len(np.unique(y_test)) > 1:
    for name in models.keys():
        y_proba = probabilities[name]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'ROC Curve not available\n(Test set has only one class)', 
             ha='center', va='center', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')

plt.tight_layout()
plt.savefig(f"{paths['results_dir']}/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved results/roc_curves_comparison.png")

# ============================================================================
# 7. PRECISION-RECALL CURVES
# ============================================================================
print("\n7. Creating precision-recall curves...")

fig, ax = plt.subplots(figsize=(8, 6))

if len(np.unique(y_test)) > 1:
    for name in models.keys():
        y_proba = probabilities[name]
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        plt.plot(recall_curve, precision_curve, 
                label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'PR Curve not available\n(Test set has only one class)', 
             ha='center', va='center', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')

plt.tight_layout()
plt.savefig(f"{paths['results_dir']}/pr_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved results/pr_curves_comparison.png")

# ============================================================================
# 8. CALIBRATION CURVES
# ============================================================================
print("\n8. Creating calibration curves...")

fig, ax = plt.subplots(figsize=(8, 6))

if len(np.unique(y_test)) > 1:
    for name in models.keys():
        y_proba = probabilities[name]
        
        try:
            fraction_positives, mean_predicted = calibration_curve(
                y_test, y_proba, n_bins=5
            )
            plt.plot(mean_predicted, fraction_positives, 's-', label=name)
        except:
            pass
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Calibration curves not available\n(Test set has only one class)', 
             ha='center', va='center', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')

plt.tight_layout()
plt.savefig(f"{paths['results_dir']}/calibration_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved results/calibration_curves.png")

# ============================================================================
# 9. PREDICTION ANALYSIS
# ============================================================================
print("\n9. Creating prediction analysis plots...")

for name in models.keys():
    y_proba = probabilities[name]
    y_pred = predictions[name]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot predictions
    correct = (y_pred == y_test)
    plt.scatter(range(len(y_test)), y_proba, 
               c=['green' if c else 'red' for c in correct],
               alpha=0.6, s=100)
    
    plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability (Beat)')
    plt.title(f'Prediction Analysis - {name}\nGreen=Correct, Red=Incorrect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{paths['results_dir']}/prediction_analysis_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {filename}")

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================
print("\n10. Saving model comparison...")

metrics_df.to_csv(f"{paths['results_dir']}/model_comparison.csv", index=False)
print("   ✓ Saved results/model_comparison.csv")

# ============================================================================
# 11. ERROR ANALYSIS
# ============================================================================
print("\n11. Performing error analysis...")

# Reload original test data with dates
df_test_full = pd.read_csv(paths['raw_csv'])
df_test_full = df_test_full[~df_test_full['horizon'].str.contains('fiscal year', case=False, na=False)]
df_test_full = df_test_full[df_test_full['date'] < '2025-12-31']
df_test_full = df_test_full[df_test_full['actual_eps'].notna()]
df_test_full = df_test_full[pd.to_datetime(df_test_full['date']) >= '2023-01-01']

misclassified_data = []

for name in models.keys():
    y_pred = predictions[name]
    y_proba = probabilities[name]
    
    for i, (pred, actual, proba) in enumerate(zip(y_pred, y_test, y_proba)):
        if pred != actual:
            misclassified_data.append({
                'Model': name,
                'Date': df_test_full.iloc[i]['date'],
                'Horizon': df_test_full.iloc[i]['horizon'],
                'Actual': int(actual),
                'Predicted': int(pred),
                'Probability': proba,
                'Actual_EPS': df_test_full.iloc[i]['actual_eps'],
                'Estimate_EPS': df_test_full.iloc[i]['eps_estimate_average']
            })

if misclassified_data:
    misclassified_df = pd.DataFrame(misclassified_data)
    misclassified_df.to_csv(f"{paths['results_dir']}/misclassified_samples.csv", index=False)
    print(f"   ✓ Found {len(misclassified_df)} misclassified samples")
    print(f"   ✓ Saved results/misclassified_samples.csv")
else:
    print("   ✓ No misclassified samples (perfect predictions!)")
    pd.DataFrame().to_csv(f"{paths['results_dir']}/misclassified_samples.csv", index=False)

# ============================================================================
# 12. COMPREHENSIVE REPORT
# ============================================================================
print("\n12. Generating comprehensive report...")

with open(f"{paths['results_dir']}/evaluation_report.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EPS BEAT PREDICTION - EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TEST SET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {len(y_test)}\n")
    f.write(f"Class distribution: {y_test.value_counts().to_dict()}\n")
    f.write(f"Beat rate: {y_test.mean():.2%}\n\n")
    
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("-" * 80 + "\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("FEATURE IMPORTANCE - TOP 10 (Random Forest)\n")
    f.write("-" * 80 + "\n")
    importances_rf = rf_model.feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1][:10]
    for i, idx in enumerate(indices_rf, 1):
        f.write(f"{i:2d}. {feature_names[idx]:40s} : {importances_rf[idx]:.4f}\n")
    f.write("\n")
    
    f.write("FEATURE IMPORTANCE - TOP 10 (XGBoost)\n")
    f.write("-" * 80 + "\n")
    importances_xgb = xgb_model.feature_importances_
    indices_xgb = np.argsort(importances_xgb)[::-1][:10]
    for i, idx in enumerate(indices_xgb, 1):
        f.write(f"{i:2d}. {feature_names[idx]:40s} : {importances_xgb[idx]:.4f}\n")
    f.write("\n")
    
    f.write("CLASSIFICATION REPORTS\n")
    f.write("-" * 80 + "\n")
    for name in models.keys():
        f.write(f"\n{name}:\n")
        y_pred = predictions[name]
        try:
            report = classification_report(y_test, y_pred, 
                                          target_names=['Miss', 'Beat'],
                                          zero_division=0)
            f.write(report)
        except:
            f.write("Classification report not available\n")
        f.write("\n")
    
    if misclassified_data:
        f.write("MISCLASSIFIED SAMPLES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total misclassified: {len(misclassified_data)}\n")
        f.write("See misclassified_samples.csv for details\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print("   ✓ Saved results/evaluation_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print("\nModel Comparison:")
print(metrics_df.to_string(index=False))

print("\nTop 5 Features (Random Forest):")
indices_rf = np.argsort(rf_model.feature_importances_)[::-1][:5]
for i, idx in enumerate(indices_rf, 1):
    print(f"  {i}. {feature_names[idx]}: {rf_model.feature_importances_[idx]:.4f}")

print("\nTop 5 Features (XGBoost):")
indices_xgb = np.argsort(xgb_model.feature_importances_)[::-1][:5]
for i, idx in enumerate(indices_xgb, 1):
    print(f"  {i}. {feature_names[idx]}: {xgb_model.feature_importances_[idx]:.4f}")

print("\nAll outputs saved to results/ directory")
print("=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)

