#!/usr/bin/env python3
"""
Aggregate ML Results Analyzer
==============================

Tüm şirketlerin evaluation_report.txt ve model_comparison.csv dosyalarını okur,
toplu analiz yapar ve kapsamlı özet rapor oluşturur.

Output:
    excel_data/ml_results_summary.csv
    excel_data/ml_performance_analysis.txt
    excel_data/feature_importance_aggregate.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from collections import defaultdict

print("=" * 80)
print("AGGREGATE ML RESULTS ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. FIND ALL EVALUATION REPORTS
# =============================================================================

print(f"\n📊 1/4 Evaluation raporları bulunuyor...")

report_pattern = "../results/*/evaluation_report.txt"
report_files = glob.glob(report_pattern)

print(f"   • {len(report_files)} evaluation report bulundu")

# =============================================================================
# 2. PARSE ALL REPORTS
# =============================================================================

print(f"\n📖 2/4 Raporlar okunuyor ve parse ediliyor...")

all_metrics = []
all_feature_importance = defaultdict(lambda: {'rf': [], 'xgb': []})

for report_file in report_files:
    # Symbol çıkar: results/AAPL/evaluation_report.txt -> AAPL
    symbol = report_file.split('/')[-2]
    
    try:
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Test set info parse et
        test_samples = None
        beat_rate = None
        
        test_match = re.search(r'Total samples: (\d+)', content)
        if test_match:
            test_samples = int(test_match.group(1))
        
        beat_match = re.search(r'Beat rate: ([\d.]+)%', content)
        if beat_match:
            beat_rate = float(beat_match.group(1))
        
        # Model metrics'leri parse et (model_comparison.csv'den daha kolay)
        csv_path = report_file.replace('evaluation_report.txt', 'model_comparison.csv')
        if os.path.exists(csv_path):
            metrics_df = pd.read_csv(csv_path)
            
            for _, row in metrics_df.iterrows():
                all_metrics.append({
                    'symbol': symbol,
                    'model': row['Model'],
                    'accuracy': row['Accuracy'],
                    'precision': row['Precision'],
                    'recall': row['Recall'],
                    'f1_score': row['F1-Score'],
                    'roc_auc': row['ROC-AUC'],
                    'test_samples': test_samples,
                    'test_beat_rate': beat_rate
                })
        
        # Feature importance parse et (RF)
        rf_section = re.search(r'FEATURE IMPORTANCE - TOP 10 \(Random Forest\)\n-+\n(.*?)\n\n', content, re.DOTALL)
        if rf_section:
            for line in rf_section.group(1).split('\n'):
                match = re.search(r'\d+\.\s+([\w_]+)\s*:\s*([\d.]+)', line)
                if match:
                    feat_name = match.group(1)
                    importance = float(match.group(2))
                    all_feature_importance[feat_name]['rf'].append(importance)
        
        # Feature importance parse et (XGBoost)
        xgb_section = re.search(r'FEATURE IMPORTANCE - TOP 10 \(XGBoost\)\n-+\n(.*?)\n\n', content, re.DOTALL)
        if xgb_section:
            for line in xgb_section.group(1).split('\n'):
                match = re.search(r'\d+\.\s+([\w_]+)\s*:\s*([\d.]+)', line)
                if match:
                    feat_name = match.group(1)
                    importance = float(match.group(2))
                    all_feature_importance[feat_name]['xgb'].append(importance)
    
    except Exception as e:
        print(f"   ⚠️  {symbol}: {str(e)[:50]}")
        continue

print(f"   • {len(all_metrics)} model sonucu toplandı")
print(f"   • {len(all_feature_importance)} unique feature bulundu")

# =============================================================================
# 3. CREATE SUMMARY DATAFRAMES
# =============================================================================

print(f"\n📈 3/4 Özet tablolar oluşturuluyor...")

# Metrics summary
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.dropna(subset=['accuracy'])  # NaN'ları çıkar

print(f"   • {metrics_df['symbol'].nunique()} şirket")
print(f"   • {len(metrics_df)} model × şirket kombinasyonu")

# Feature importance summary
feat_importance_data = []
for feat_name, values in all_feature_importance.items():
    rf_scores = values['rf']
    xgb_scores = values['xgb']
    
    feat_importance_data.append({
        'feature': feat_name,
        'rf_mean': np.mean(rf_scores) if rf_scores else 0,
        'rf_median': np.median(rf_scores) if rf_scores else 0,
        'rf_count': len(rf_scores),
        'xgb_mean': np.mean(xgb_scores) if xgb_scores else 0,
        'xgb_median': np.median(xgb_scores) if xgb_scores else 0,
        'xgb_count': len(xgb_scores),
        'combined_mean': (np.mean(rf_scores) if rf_scores else 0) + (np.mean(xgb_scores) if xgb_scores else 0)
    })

feat_importance_df = pd.DataFrame(feat_importance_data).sort_values('combined_mean', ascending=False)

# =============================================================================
# 4. SAVE OUTPUTS
# =============================================================================

print(f"\n💾 4/4 Çıktılar kaydediliyor...")

# Save detailed metrics
metrics_df.to_csv('excel_data/ml_results_summary.csv', index=False)
print(f"   ✓ excel_data/ml_results_summary.csv")

# Save feature importance
feat_importance_df.to_csv('excel_data/feature_importance_aggregate.csv', index=False)
print(f"   ✓ excel_data/feature_importance_aggregate.csv")

# =============================================================================
# 5. GENERATE COMPREHENSIVE ANALYSIS REPORT
# =============================================================================

print(f"\n📄 Kapsamlı analiz raporu oluşturuluyor...")

with open('excel_data/ml_performance_analysis.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("S&P 500 ML PIPELINE - AGGREGATE PERFORMANCE ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    # Overall statistics
    f.write("1. OVERALL STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total companies analyzed: {metrics_df['symbol'].nunique()}\n")
    f.write(f"Total model evaluations: {len(metrics_df)}\n")
    f.write(f"Average test samples per company: {metrics_df['test_samples'].mean():.1f}\n")
    f.write(f"Average test beat rate: {metrics_df['test_beat_rate'].mean():.2f}%\n\n")
    
    # Model performance comparison
    f.write("2. MODEL PERFORMANCE COMPARISON (AVERAGE ACROSS ALL COMPANIES)\n")
    f.write("-" * 80 + "\n")
    
    model_summary = metrics_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].agg(['mean', 'std', 'median'])
    f.write(model_summary.to_string())
    f.write("\n\n")
    
    # Best performing models per metric
    f.write("3. BEST PERFORMING MODELS\n")
    f.write("-" * 80 + "\n")
    
    for metric in ['accuracy', 'roc_auc', 'f1_score']:
        best_model = metrics_df.groupby('model')[metric].mean().idxmax()
        best_score = metrics_df.groupby('model')[metric].mean().max()
        f.write(f"{metric.upper():20s}: {best_model:25s} ({best_score:.4f})\n")
    
    f.write("\n")
    
    # Top companies (best accuracy)
    f.write("4. TOP 10 COMPANIES (Best Average Accuracy)\n")
    f.write("-" * 80 + "\n")
    
    top_companies = metrics_df.groupby('symbol')['accuracy'].mean().sort_values(ascending=False).head(10)
    for i, (symbol, acc) in enumerate(top_companies.items(), 1):
        f.write(f"{i:2d}. {symbol:6s} - Accuracy: {acc:.2%}\n")
    
    f.write("\n")
    
    # Worst companies (lowest accuracy)
    f.write("5. BOTTOM 10 COMPANIES (Lowest Average Accuracy)\n")
    f.write("-" * 80 + "\n")
    
    bottom_companies = metrics_df.groupby('symbol')['accuracy'].mean().sort_values().head(10)
    for i, (symbol, acc) in enumerate(bottom_companies.items(), 1):
        f.write(f"{i:2d}. {symbol:6s} - Accuracy: {acc:.2%}\n")
    
    f.write("\n")
    
    # Feature importance aggregate
    f.write("6. TOP 20 MOST IMPORTANT FEATURES (Aggregate Across All Companies)\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Rank':<6}{'Feature':<45}{'RF Mean':<12}{'XGB Mean':<12}{'Combined':<12}\n")
    f.write("-" * 80 + "\n")
    
    for i, row in feat_importance_df.head(20).iterrows():
        f.write(f"{i+1:<6}{row['feature']:<45}{row['rf_mean']:<12.4f}{row['xgb_mean']:<12.4f}{row['combined_mean']:<12.4f}\n")
    
    f.write("\n")
    
    # Model reliability (consistency across companies)
    f.write("7. MODEL RELIABILITY (Consistency Metrics)\n")
    f.write("-" * 80 + "\n")
    
    for model_name in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model_name]
        f.write(f"\n{model_name}:\n")
        f.write(f"  Mean Accuracy:  {model_data['accuracy'].mean():.4f} ± {model_data['accuracy'].std():.4f}\n")
        f.write(f"  Mean ROC-AUC:   {model_data['roc_auc'].mean():.4f} ± {model_data['roc_auc'].std():.4f}\n")
        f.write(f"  Coefficient of Variation (Accuracy): {(model_data['accuracy'].std() / model_data['accuracy'].mean()):.2%}\n")
    
    f.write("\n")
    
    # Distribution analysis
    f.write("8. PERFORMANCE DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    
    for model_name in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model_name]['accuracy']
        f.write(f"\n{model_name} Accuracy Distribution:\n")
        f.write(f"  Min:     {model_data.min():.4f}\n")
        f.write(f"  25th %:  {model_data.quantile(0.25):.4f}\n")
        f.write(f"  Median:  {model_data.median():.4f}\n")
        f.write(f"  75th %:  {model_data.quantile(0.75):.4f}\n")
        f.write(f"  Max:     {model_data.max():.4f}\n")
        f.write(f"  Companies with >80% accuracy: {(model_data > 0.8).sum()}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF ANALYSIS\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ excel_data/ml_performance_analysis.txt")

# =============================================================================
# 6. SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Özet İstatistikler:")
print("=" * 80)

print(f"\nŞirket Başına Ortalama Model Performansı:")
print(metrics_df.groupby('model')[['accuracy', 'roc_auc', 'f1_score']].mean().to_string())

print(f"\n\nEn İyi 10 Feature (Combined RF + XGBoost):")
print(feat_importance_df.head(10)[['feature', 'rf_mean', 'xgb_mean', 'combined_mean']].to_string(index=False))

print(f"\n\nModel Karşılaştırması:")
print(f"   • Random Forest:        Ortalama Accuracy: {metrics_df[metrics_df['model']=='Random Forest']['accuracy'].mean():.2%}")
print(f"   • XGBoost:              Ortalama Accuracy: {metrics_df[metrics_df['model']=='XGBoost']['accuracy'].mean():.2%}")
print(f"   • Logistic Regression:  Ortalama Accuracy: {metrics_df[metrics_df['model']=='Logistic Regression']['accuracy'].mean():.2%}")

print("\n" + "=" * 80)
print("✅ AGGREGATE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs:")
print(f"   • excel_data/ml_results_summary.csv")
print(f"   • excel_data/ml_performance_analysis.txt")
print(f"   • excel_data/feature_importance_aggregate.csv")
print("=" * 80)

