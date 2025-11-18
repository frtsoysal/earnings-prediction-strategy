#!/usr/bin/env python3
"""
S&P 500 Scatter Matrix & Feature Analysis
==========================================

500 şirketin earnings verileri ile EPS beat prediction için:
- Feature selection (MI + Spearman)
- Decile charts (tüm şirketler)
- Scatter matrix (stratified sample)
- Point-in-time safe (leak prevention)

Usage:
    python scatter_matrix_analysis.py
    
Output:
    reports/feature_deciles_all.png
    reports/pairgrid_global.png
    reports/feature_signal_summary.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_GLOB = "../data/raw/*_earnings_with_q4.csv"
TARGET = "eps_beat"

# Leak prevention: rapor sonrası kolonlar
LEAK_COLS = [
    "actual_eps", "eps_delta", 
    "elo_after", "elo_change",
    "price_change_1m_pct", "price_change_3m_pct",
    "price_at_report",  # rapor günü fiyatı
]

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Stratified sampling
PER_SYMBOL_CAP = 10
RANDOM_SEED = 42

# Feature selection
MIN_COVERAGE = 0.85  # %85+ doluluk
TOP_N_FEATURES = 10  # Using top 10 from Global Model

# Decile chart
N_DECILES = 10
WINSORIZE_Q = (0.01, 0.99)  # Outlier kesme

print("=" * 80)
print("S&P 500 SCATTER MATRIX & FEATURE ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING & MERGING
# =============================================================================

print(f"\n📊 1/6 Veri yükleniyor...")

csv_files = glob.glob(DATA_GLOB)
print(f"   • {len(csv_files)} CSV dosyası bulundu")

dfs = []
for fp in csv_files:
    # Symbol çıkar: AAPL_earnings_with_q4.csv -> AAPL
    filename = os.path.basename(fp)
    symbol = filename.replace("_earnings_with_q4.csv", "")
    
    try:
        df = pd.read_csv(fp, low_memory=False)
        df['symbol'] = symbol
        dfs.append(df)
    except Exception as e:
        print(f"   ⚠️  {symbol}: {e}")
        continue

print(f"   • {len(dfs)} şirket yüklendi")

full = pd.concat(dfs, ignore_index=True)
print(f"   • Toplam satır: {len(full):,}")
print(f"   • Toplam kolon: {len(full.columns)}")

# =============================================================================
# 2. LEAK PREVENTION
# =============================================================================

print(f"\n🔒 2/6 Leak prevention...")

# Hedef kolonunu kontrol et
if TARGET not in full.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

# Leak kolonlarını çıkar
leak_found = [c for c in LEAK_COLS if c in full.columns]
if leak_found:
    full = full.drop(columns=leak_found)
    print(f"   • {len(leak_found)} leak kolonu çıkarıldı: {leak_found[:5]}...")

# Keep important elo features (momentum, decay, vol_4q, before)
# These are critical features in global model
print(f"   • Elo features kept for analysis (critical for global model)")

# Hedefi olmayan satırları çıkar
full = full.dropna(subset=[TARGET])
print(f"   • Target olmayan satırlar çıkarıldı: {len(full):,} satır kaldı")

# Hedefi int'e çevir
full[TARGET] = full[TARGET].astype(int)

# =============================================================================
# 3. FEATURE SELECTION (from Global Model)
# =============================================================================

print(f"\n🎯 3/6 Feature selection (from Global Model evaluation)...")

# Top 10 features from Global Model (Random Forest feature importance)
# Source: graphtest/global_model/results/evaluation_report.txt
top_features = [
    'elo_momentum',
    'elo_before',
    'elo_decay',
    'elo_vol_4q',
    'actual_eps_qoq_growth_lag1',
    'price_1m_before',
    'price_3m_before',
    'total_revenue_qoq_growth_lag1',
    'revenue_estimate_average',
    'eps_estimate_average',
]

# Check which features exist in data
available_features = [f for f in top_features if f in full.columns]
missing_features = [f for f in top_features if f not in full.columns]

print(f"   • Top {len(top_features)} features from Global Model")
print(f"   • Available: {len(available_features)}")
if missing_features:
    print(f"   • Missing: {missing_features}")

# Use available features
top_features = available_features

print(f"\n   ✓ Using {len(top_features)} features:")
for i, feat in enumerate(top_features, 1):
    coverage = full[feat].notna().mean()
    print(f"      {i:2d}. {feat:35s} (coverage: {coverage:.1%})")

# =============================================================================
# 4. STRATIFIED SAMPLING (for pairplot)
# =============================================================================

print(f"\n📦 4/6 Stratified sampling...")

# Pairplot için dengeli örnek
balanced = (
    full[top_features + [TARGET, 'symbol']]
    .dropna()
    .groupby('symbol', group_keys=False)
    .apply(lambda g: g.sample(min(PER_SYMBOL_CAP, len(g)), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)

print(f"   • Stratified sample: {len(balanced):,} satır")
print(f"   • Şirket sayısı: {balanced['symbol'].nunique()}")
print(f"   • Ortalama satır/şirket: {len(balanced) / balanced['symbol'].nunique():.1f}")

# =============================================================================
# 5. DECILE CHARTS (tüm veri)
# =============================================================================

print(f"\n📊 5/6 Decile charts oluşturuluyor...")

def decile_panel(df, features, target=TARGET, n_deciles=N_DECILES, 
                 winsorize=WINSORIZE_Q, out="reports/feature_deciles_all.png"):
    """
    Her feature için decile-based beat rate grafiği.
    
    Args:
        df: DataFrame
        features: Feature listesi
        target: Target kolonu
        n_deciles: Decile sayısı
        winsorize: (lower, upper) quantile için outlier kesme
        out: Çıktı dosyası
    """
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig = plt.figure(figsize=(16, 4 * n_rows))
    
    for i, feat in enumerate(features, 1):
        ax = plt.subplot(n_rows, n_cols, i)
        
        # Feature ve target verisi
        data = df[[feat, target]].dropna()
        
        if len(data) < 100:
            ax.text(0.5, 0.5, f"Insufficient data\n({len(data)} rows)", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feat)
            continue
        
        # Winsorize (outlier kesme)
        if winsorize:
            q_low, q_high = data[feat].quantile(list(winsorize))
            data = data[(data[feat] >= q_low) & (data[feat] <= q_high)]
        
        # Decile bins
        try:
            data['bin'] = pd.qcut(data[feat], q=n_deciles, duplicates='drop')
            grouped = data.groupby('bin')[target].agg(['mean', 'count']).reset_index()
        except Exception as e:
            ax.text(0.5, 0.5, f"Binning error\n{str(e)[:30]}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feat)
            continue
        
        # Beat rate line (primary y-axis)
        x_pos = range(len(grouped))
        ax.plot(x_pos, grouped['mean'], marker='o', linewidth=2, 
               markersize=6, color='#4CAF50', label='Beat rate')
        ax.set_ylabel('Beat rate', color='#4CAF50', fontsize=10)
        ax.tick_params(axis='y', labelcolor='#4CAF50')
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, linestyle='--')
        
        # Count bars (secondary y-axis)
        ax2 = ax.twinx()
        ax2.bar(x_pos, grouped['count'], alpha=0.2, color='#999', label='Count')
        ax2.set_ylabel('Count', color='#999', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#999')
        
        # Title & labels
        ax.set_title(f"{feat}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Decile (1=lowest, 10=highest)', fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{j+1}" for j in x_pos], fontsize=8)
        
    plt.suptitle(f"Feature → Beat Rate Analysis (All {df['symbol'].nunique()} companies)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches='tight')
    print(f"   ✓ Saved: {out}")
    plt.close()

# Generate decile charts
decile_panel(full, top_features, out=f"{REPORTS_DIR}/feature_deciles_all.png")

# =============================================================================
# 6. SCATTER MATRIX (stratified sample)
# =============================================================================

print(f"\n🎨 6/6 Scatter matrix oluşturuluyor...")

# Dark theme setup
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#000",
    "axes.facecolor": "#000",
    "axes.labelcolor": "#DDD",
    "xtick.color": "#BBB",
    "ytick.color": "#BBB",
    "grid.color": "#333",
    "legend.frameon": False,
    "font.size": 9,
    "text.color": "#DDD"
})

# Color palette: 0=miss (orange), 1=beat (blue)
palette = {0: "#FFA64D", 1: "#6EC1FF"}

# Sample for pairplot
sample = balanced[[TARGET] + top_features].dropna()
print(f"   • Sample size: {len(sample):,} rows")
print(f"   • Beat rate: {sample[TARGET].mean():.2%}")

# PairGrid
g = sns.PairGrid(
    sample, 
    vars=top_features,
    hue=TARGET,
    corner=True,
    diag_sharey=False,
    height=1.5,
    palette=palette,
    despine=False
)

# Scatter plots (lower triangle)
g.map_lower(sns.scatterplot, s=7, alpha=0.6, edgecolor=None)

# KDE plots (diagonal)
g.map_diag(sns.kdeplot, fill=True, alpha=0.5, linewidth=1.5)

# Legend
g.add_legend(title="EPS Beat", loc='upper right', frameon=True, 
            facecolor='#111', edgecolor='#444')

# Title
plt.suptitle(
    f"Global Scatter Matrix - Stratified Sample ({len(sample):,} observations, {balanced['symbol'].nunique()} companies)",
    color="#FFF",
    fontsize=14,
    fontweight='bold',
    y=1.00
)

plt.tight_layout()
out_path = f"{REPORTS_DIR}/pairgrid_global.png"
plt.savefig(out_path, dpi=200, facecolor='#000', edgecolor='none')
print(f"   ✓ Saved: {out_path}")
plt.close()

# =============================================================================
# 7. SUMMARY EXPORT
# =============================================================================

print(f"\n📄 Exporting summary...")

# Feature summary (from Global Model rankings)
summary = pd.DataFrame({
    'rank': range(1, len(top_features) + 1),
    'feature': top_features,
    'source': 'Global Model RF Feature Importance'
})

summary_path = f"{REPORTS_DIR}/feature_signal_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"   ✓ Saved: {summary_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nData:")
print(f"   • Companies: {full['symbol'].nunique()}")
print(f"   • Total observations: {len(full):,}")
print(f"   • Top features (from Global Model): {len(top_features)}")
print(f"   • Overall beat rate: {full[TARGET].mean():.2%}")

print(f"\nTop 10 Features (Global Model RF Importance):")
for i, feat in enumerate(top_features, 1):
    print(f"   {i:2d}. {feat}")

print(f"\nOutputs:")
print(f"   • {REPORTS_DIR}/feature_deciles_all.png")
print(f"   • {REPORTS_DIR}/pairgrid_global.png")
print(f"   • {REPORTS_DIR}/feature_signal_summary.csv")

print("\n" + "=" * 80)

