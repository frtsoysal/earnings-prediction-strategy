#!/usr/bin/env python3
"""
Professional Figure Generation - Goldman Sachs / Morgan Stanley Style
======================================================================

Regenerates all key figures with consistent professional styling:
- Navy blue primary color (#002D72)
- Gold accents (#C5A572)
- Clean axes, professional grid
- Source citations
- High DPI (300)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING PROFESSIONAL FIGURES - GOLDMAN SACHS STYLE")
print("=" * 80)

# =============================================================================
# PROFESSIONAL STYLE CONFIGURATION
# =============================================================================

# Professional Blue (matching Polymarket mechanism)
BLUE_HEX = '#2295DC'  # RGB(34, 149, 220)

# Color Palette
GS_COLORS = {
    'navy': BLUE_HEX,       # Primary blue
    'gold': BLUE_HEX,       # Using same blue
    'green': '#0A6E4E',     # Success/positive
    'red': '#8B0000',       # Danger/negative
    'gray': '#4A4A4A',      # Neutral
    'light_gray': '#CCCCCC' # Grid/background
}

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

def add_source_citation(fig, source_text, analysis_date="November 2025"):
    """Add professional source citation to figure"""
    fig.text(0.99, 0.01, f"Source: {source_text} | Analysis: {analysis_date}",
            ha='right', va='bottom', fontsize=8, style='italic', color=GS_COLORS['gray'])

# =============================================================================
# FIGURE 1: YEARLY BEAT RATE TREND (2017-2025)
# =============================================================================

print("\n📊 Figure 1: Yearly Beat Rate Trend...")

yearly_df = pd.read_csv('../excel_data/yearly_beat_miss.csv')

fig, ax = plt.subplots(figsize=(12, 7))

# Line plot with markers
years = yearly_df['year'].values
beat_pct = yearly_df['beat_pct'].values

ax.plot(years, beat_pct, 'o-', linewidth=3, markersize=10,
       color=GS_COLORS['navy'], markeredgecolor='white', markeredgewidth=2)

# Add value labels
for x, y in zip(years, beat_pct):
    ax.text(x, y + 1.5, f'{y:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Trend line
z = np.polyfit(years, beat_pct, 1)
p = np.poly1d(z)
ax.plot(years, p(years), '--', linewidth=2, color=GS_COLORS['gold'], 
       label=f'Trend (slope: {z[0]:+.2f}% per year)', alpha=0.7)

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Beat Rate (%)', fontsize=13, fontweight='bold')

ax.set_ylim([60, 85])
ax.set_xticks(years)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=GS_COLORS['gray'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GS_COLORS['gray'])
ax.spines['bottom'].set_color(GS_COLORS['gray'])

add_source_citation(fig, f"Alpha Vantage API, {yearly_df['total_companies'].sum():,} quarterly reports")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig1_yearly_beat_rate_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig1_yearly_beat_rate_trend.png")
plt.close()

# =============================================================================
# FIGURE 2: GLOBAL MODEL PERFORMANCE
# =============================================================================

print("\n📊 Figure 2: Global Model Performance...")

model_perf = pd.read_csv('../global_model/results/model_performance.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Accuracy & ROC-AUC
models = model_perf['Model'].values
accuracy = model_perf['Accuracy'].values * 100
roc_auc = model_perf['ROC-AUC'].values * 100

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy',
               color=GS_COLORS['navy'], edgecolor='white', linewidth=2)
bars2 = ax1.bar(x + width/2, roc_auc, width, label='ROC-AUC',
               color=GS_COLORS['gold'], edgecolor='white', linewidth=2)

# Add values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Accuracy & ROC-AUC', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([75, 90])

# Panel B: Precision & Recall
precision = model_perf['Precision'].values * 100
recall = model_perf['Recall'].values * 100

bars3 = ax2.bar(x - width/2, precision, width, label='Precision',
               color=GS_COLORS['green'], edgecolor='white', linewidth=2)
bars4 = ax2.bar(x + width/2, recall, width, label='Recall',
               color=GS_COLORS['gold'], edgecolor='white', linewidth=2)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: Precision & Recall', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15, ha='right')
ax2.legend(loc='lower right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([80, 92])

plt.suptitle('Global ML Model Performance\nOut-of-Sample Test Set (N=9,256 observations, 2020-2025)',
            fontsize=15, fontweight='bold', y=0.98)

add_source_citation(fig, "Global model evaluation, 468 S&P 500 companies")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig2_global_model_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig2_global_model_performance.png")
plt.close()

# =============================================================================
# FIGURE 3: FEATURE IMPORTANCE (ELO DOMINANCE)
# =============================================================================

print("\n📊 Figure 3: Feature Importance...")

# Top 10 features
features = ['elo_momentum', 'elo_before', 'elo_decay', 'elo_vol_4q',
           'actual_eps_qoq_growth_lag1', 'price_1m_before', 'price_3m_before',
           'total_revenue_qoq_growth_lag1', 'revenue_estimate_average', 'eps_estimate_average']
importances = [36.17, 6.38, 6.16, 5.27, 2.66, 2.43, 2.38, 2.31, 2.18, 2.16]
categories = ['Elo', 'Elo', 'Elo', 'Elo', 'Growth', 'Price', 'Price', 'Growth', 'Estimates', 'Estimates']

fig, ax = plt.subplots(figsize=(12, 8))

# Color by category
colors = [GS_COLORS['navy'] if c == 'Elo' else 
         (GS_COLORS['green'] if c == 'Growth' else 
         (GS_COLORS['gold'] if c == 'Price' else GS_COLORS['gray']))
         for c in categories]

bars = ax.barh(range(len(features)), importances, color=colors, 
              edgecolor='white', linewidth=2, alpha=0.9)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importances)):
    ax.text(val, i, f'  {val:.2f}%', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=11)
ax.set_xlabel('Feature Importance (%)', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Predictive Features (Random Forest)\nElo Metrics Dominate Model Performance',
            fontsize=15, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=GS_COLORS['navy'], label='Elo System (53.98%)'),
    Patch(facecolor=GS_COLORS['green'], label='Growth Metrics'),
    Patch(facecolor=GS_COLORS['gold'], label='Price Momentum'),
    Patch(facecolor=GS_COLORS['gray'], label='Analyst Estimates')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
         facecolor='white', edgecolor=GS_COLORS['gray'])

ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

add_source_citation(fig, "Global Random Forest, N=14,239 observations")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig3_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig3_feature_importance.png")
plt.close()

# =============================================================================
# FIGURE 4: ANALYST CONSENSUS SPREAD ANALYSIS
# =============================================================================

print("\n📊 Figure 4: Consensus Spread Analysis...")

spread_df = pd.read_csv('../research_paper/tables/beat_rate_by_spread.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Beat rate by quintile
quintiles = range(len(spread_df))
beat_rates = spread_df['eps_beat_mean'].values * 100

bars = ax1.bar(quintiles, beat_rates, color=GS_COLORS['navy'], 
              edgecolor='white', linewidth=2, alpha=0.9)

# Trend line
ax1.plot(quintiles, beat_rates, 'o-', color=GS_COLORS['red'], linewidth=2.5, 
        markersize=10, label='Monotonic decline')

for i, (bar, val) in enumerate(zip(bars, beat_rates)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Consensus Spread Quintile', fontsize=12, fontweight='bold')
ax1.set_ylabel('Beat Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Beat Rates by Analyst Disagreement', fontsize=13, fontweight='bold')
ax1.set_xticks(quintiles)
ax1.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
ax1.set_ylim([55, 85])
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Panel B: Sample distribution
counts = spread_df['eps_beat_count'].values

ax2.bar(quintiles, counts, color=GS_COLORS['gray'], 
       edgecolor='white', linewidth=2, alpha=0.7)

for i, val in enumerate(counts):
    ax2.text(i, val, f'{int(val):,}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

ax2.set_xlabel('Consensus Spread Quintile', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: Sample Distribution', fontsize=13, fontweight='bold')
ax2.set_xticks(quintiles)
ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Analyst Consensus Uncertainty and Beat Probability\n15.1pp Spread | χ²=186.80, p<0.001',
            fontsize=15, fontweight='bold', y=0.98)

add_source_citation(fig, "N=13,856 observations, 467 S&P 500 companies, 2017-2025")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig4_consensus_spread.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig4_consensus_spread.png")
plt.close()

# =============================================================================
# FIGURE 5: STRATEGY ROI COMPARISON
# =============================================================================

print("\n📊 Figure 5: Strategy ROI Comparison...")

strategy_df = pd.read_csv('../polymarket_backtest/tables/strategy_comparison_full.csv')
strategy_df = strategy_df.sort_values('roi_net', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))

# Color code: positive=green, negative=red
colors = [GS_COLORS['green'] if x > 0 else GS_COLORS['red'] 
         for x in strategy_df['roi_net']]

bars = ax.barh(range(len(strategy_df)), strategy_df['roi_net'],
              color=colors, edgecolor='white', linewidth=2, alpha=0.9)

# Highlight Sweet Spot
sweet_spot_idx = strategy_df[strategy_df['Strategy'].str.contains('Sweet Spot')].index[0]
bars[sweet_spot_idx].set_edgecolor(GS_COLORS['gold'])
bars[sweet_spot_idx].set_linewidth(4)

# Add value labels
for i, (val, trades) in enumerate(zip(strategy_df['roi_net'], strategy_df['trades'])):
    label = f'{val:+.1f}% (n={int(trades)})'
    ax.text(val, i, f'  {label}', va='center', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_yticks(range(len(strategy_df)))
ax.set_yticklabels(strategy_df['Strategy'], fontsize=11)
ax.set_xlabel('ROI (Net, After Fees) %', fontsize=13, fontweight='bold')
ax.set_title('Trading Strategy Performance Comparison\nKelly 25%, Polymarket 2% Fees',
            fontsize=15, fontweight='bold', pad=20)

ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add Sweet Spot annotation
sweet_spot_row = strategy_df.loc[sweet_spot_idx]
ax.text(0.98, 0.98, f'🏆 Optimal Strategy:\n{sweet_spot_row["Strategy"]}\nROI: {sweet_spot_row["roi_net"]:+.1f}%',
       transform=ax.transAxes, ha='right', va='top',
       bbox=dict(boxstyle='round', facecolor=GS_COLORS['gold'], alpha=0.3, edgecolor=GS_COLORS['gold']),
       fontsize=11, fontweight='bold')

add_source_citation(fig, "Polymarket backtest, 248 events, Q3 2025")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig5_strategy_roi_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig5_strategy_roi_comparison.png")
plt.close()

# =============================================================================
# FIGURE 6: EDGE DISTRIBUTION & SWEET SPOT
# =============================================================================

print("\n📊 Figure 6: Edge Distribution...")

# Load backtest data
backtest = pd.read_csv('../polymarket_backtest/backtest_results.csv')
poly = pd.read_csv('../../../earnings_history_1year.csv')
df = backtest.merge(poly[['marketId', 'avgPrice']], left_on='market_id', right_on='marketId', how='left')
df['edge_lr'] = df['lr_probability'] - df['avgPrice']

fig, ax = plt.subplots(figsize=(14, 7))

# Histogram
n, bins, patches = ax.hist(df['edge_lr'].dropna(), bins=40, 
                           color=GS_COLORS['navy'], alpha=0.7, edgecolor='white')

# Color Sweet Spot bins (0.10-0.15)
for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
    if 0.10 <= bin_edge < 0.15:
        patch.set_facecolor(GS_COLORS['gold'])
        patch.set_alpha(1.0)
        patch.set_edgecolor(GS_COLORS['gold'])
        patch.set_linewidth(3)

# Zero line
ax.axvline(x=0, color=GS_COLORS['red'], linestyle='--', linewidth=2.5, label='Zero Edge')

# Sweet Spot boundaries
ax.axvline(x=0.10, color=GS_COLORS['gold'], linestyle='--', linewidth=2.5, label='Sweet Spot (10-15%)')
ax.axvline(x=0.15, color=GS_COLORS['gold'], linestyle='--', linewidth=2.5)

# Shaded region
ax.axvspan(0.10, 0.15, alpha=0.2, color=GS_COLORS['gold'])

ax.set_xlabel('Edge (p_model - p_market)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Edge Distribution with Sweet Spot Highlighted\n93.75% Win Rate in 10-15% Range',
            fontsize=15, fontweight='bold', pad=20)

ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=GS_COLORS['gray'])
ax.grid(axis='y', alpha=0.3)

add_source_citation(fig, "N=248 Polymarket events, Logistic Regression probabilities")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig6_edge_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig6_edge_distribution.png")
plt.close()

# =============================================================================
# FIGURE 7: KELLY FRACTION OPTIMIZATION
# =============================================================================

print("\n📊 Figure 7: Kelly Optimization...")

kelly_df = pd.read_csv('../polymarket_backtest/tables/kelly_fraction_comparison.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: ROI (constant)
kelly_pct = kelly_df['kelly_fraction'].values * 100
roi = kelly_df['roi_net'].values

ax1.plot(kelly_pct, roi, 'o-', linewidth=3, markersize=12,
        color=GS_COLORS['navy'], markeredgecolor='white', markeredgewidth=2)

for x, y in zip(kelly_pct, roi):
    ax1.text(x, y+0.3, f'{y:.2f}%', ha='center', fontsize=10, fontweight='bold')

ax1.axhline(y=roi[0], color=GS_COLORS['gold'], linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Kelly Fraction (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: ROI vs Kelly Fraction\n(Constant Across Fractions)', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_ylim([12, 14])

# Panel B: Capital requirement
capital = kelly_df['total_invested'].values

ax2.plot(kelly_pct, capital, 'o-', linewidth=3, markersize=12,
        color=GS_COLORS['green'], markeredgecolor='white', markeredgewidth=2)

for x, y in zip(kelly_pct, capital):
    ax2.text(x, y+50, f'${y:,.0f}', ha='center', fontsize=9, fontweight='bold')

# Highlight recommended
recommended_idx = 1  # 25%
ax2.scatter([kelly_pct[recommended_idx]], [capital[recommended_idx]], 
           s=400, facecolors='none', edgecolors=GS_COLORS['gold'], linewidths=4,
           label='Recommended (25%)')

ax2.set_xlabel('Kelly Fraction (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Capital Required ($)', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: Capital Requirement (Linear Scaling)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.suptitle('Kelly Criterion Optimization\nROI Independent of Fraction | 25% Recommended',
            fontsize=15, fontweight='bold', y=0.98)

add_source_citation(fig, "N=90 positive edge events")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig7_kelly_optimization.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ fig7_kelly_optimization.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("✅ PROFESSIONAL FIGURES GENERATED")
print("=" * 80)

print("\nGenerated Figures:")
print("   1. Yearly Beat Rate Trend (2017-2025)")
print("   2. Global Model Performance (Accuracy, Precision, Recall, ROC-AUC)")
print("   3. Feature Importance (Elo dominance visualization)")
print("   4. Analyst Consensus Spread Analysis")
print("   5. Strategy ROI Comparison (Sweet Spot highlighted)")
print("   6. Edge Distribution with Sweet Spot zone")
print("   7. Kelly Fraction Optimization curves")

print("\nStyle: Goldman Sachs / Morgan Stanley")
print("   • Navy blue primary (#002D72)")
print("   • Gold accents (#C5A572)")
print("   • Professional grid & axes")
print("   • Source citations")
print("   • 300 DPI (print-ready)")

print("\nLocation: graphtest/final_report/figures/")
print("=" * 80)

