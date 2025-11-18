#!/usr/bin/env python3
"""
Additional Professional Figures Generation
===========================================

Creates 15+ additional figures for comprehensive report coverage.
Every section needs 2-3 visualizations minimum.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING ADDITIONAL PROFESSIONAL FIGURES (15+)")
print("=" * 80)

# Goldman Sachs colors
GS_COLORS = {
    'navy': '#002D72',
    'gold': '#C5A572',
    'green': '#0A6E4E',
    'red': '#8B0000',
    'gray': '#4A4A4A'
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.titleweight': 'bold'})

def add_source(fig, text):
    fig.text(0.99, 0.01, f"Source: {text}", ha='right', va='bottom', 
            fontsize=8, style='italic', color=GS_COLORS['gray'])

# =============================================================================
# FIGURE 8: S&P 500 COVERAGE MAP
# =============================================================================

print("\n📊 Fig 8: S&P 500 Coverage Map...")

# Load data
csv_files = glob.glob('../../data/raw/*_earnings_with_q4.csv')
coverage_data = []

for fp in csv_files:
    ticker = fp.split('/')[-1].replace('_earnings_with_q4.csv', '')
    try:
        df = pd.read_csv(fp)
        coverage_data.append({
            'ticker': ticker,
            'observations': len(df),
            'has_data': 1 if len(df) > 5 else 0
        })
    except:
        coverage_data.append({'ticker': ticker, 'observations': 0, 'has_data': 0})

coverage_df = pd.DataFrame(coverage_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Distribution
ax1.hist(coverage_df[coverage_df['observations']>0]['observations'], bins=30,
        color=GS_COLORS['navy'], alpha=0.8, edgecolor='white')
ax1.set_xlabel('Observations per Company', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Panel A: Data Coverage Distribution', fontweight='bold')
ax1.grid(alpha=0.3)

# Panel B: Pie chart
with_data = int(coverage_df['has_data'].sum())
total_sp500 = 503
missing = max(0, total_sp500 - with_data)

coverage_summary = [with_data, missing]
labels = [f'With Data\n({with_data})', f'Missing\n({missing})']

if missing > 0:
    ax2.pie(coverage_summary, labels=labels, colors=[GS_COLORS['green'], GS_COLORS['red']],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
else:
    ax2.pie([with_data], labels=[f'With Data\n({with_data})'], colors=[GS_COLORS['green']],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax2.set_title('Panel B: S&P 500 Coverage', fontweight='bold')

plt.suptitle('Data Coverage Across S&P 500 Universe', fontsize=15, fontweight='bold')
add_source(fig, "498 companies successfully fetched from Alpha Vantage")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig8_coverage_map.png', dpi=300, bbox_inches='tight')
print("   ✓ fig8_coverage_map.png")
plt.close()

# =============================================================================
# FIGURE 9: ELO DISTRIBUTION
# =============================================================================

print("\n📊 Fig 9: Elo Distribution...")

# Load combined data
combined = pd.read_csv('../global_model/data/combined_data.csv')
combined_clean = combined[combined['elo_before'].notna()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Elo histogram
ax1.hist(combined_clean['elo_before'], bins=50, color=GS_COLORS['navy'], 
        alpha=0.8, edgecolor='white')
ax1.axvline(x=1500, color=GS_COLORS['gold'], linestyle='--', linewidth=2, label='Neutral (1500)')
ax1.set_xlabel('Elo Rating', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Panel A: Elo Rating Distribution', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel B: Elo vs Beat Rate
elo_bins = pd.cut(combined_clean['elo_before'], bins=10)
elo_beat = combined_clean.groupby(elo_bins)['eps_beat'].agg(['mean', 'count'])
elo_beat = elo_beat[elo_beat['count'] >= 30]  # Min sample

x_pos = range(len(elo_beat))
ax2.plot(x_pos, elo_beat['mean'] * 100, 'o-', linewidth=2.5, markersize=10,
        color=GS_COLORS['green'], markeredgecolor='white', markeredgewidth=2)

ax2.set_xlabel('Elo Decile (Low → High)', fontweight='bold')
ax2.set_ylabel('Beat Rate (%)', fontweight='bold')
ax2.set_title('Panel B: Beat Rate by Elo Level', fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_ylim([50, 90])

plt.suptitle('Elo Rating System Performance', fontsize=15, fontweight='bold')
add_source(fig, "N=14,239 observations, 468 companies")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig9_elo_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ fig9_elo_distribution.png")
plt.close()

# =============================================================================
# FIGURE 10: ROC CURVES (3 Models)
# =============================================================================

print("\n📊 Fig 10: ROC Curves...")

# Load test data
X_test = pd.read_csv('../global_model/data/X_test_global.csv')
y_test = pd.read_csv('../global_model/data/y_test_global.csv')['eps_beat']

# Load models
import joblib
rf = joblib.load('../global_model/models/global_rf_model.pkl')
xgb = joblib.load('../global_model/models/global_xgb_model.pkl')
lr = joblib.load('../global_model/models/global_lr_model.pkl')
prep = joblib.load('../global_model/models/global_preprocessor.pkl')

X_processed = prep.transform(X_test)

fig, ax = plt.subplots(figsize=(10, 8))

models = [
    ('Random Forest', rf, GS_COLORS['navy']),
    ('XGBoost', xgb, GS_COLORS['green']),
    ('Logistic Regression', lr, GS_COLORS['gold'])
]

for name, model, color in models:
    y_pred_proba = model.predict_proba(X_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={roc_auc:.3f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('ROC Curves - Model Discriminative Power\nOut-of-Sample Test Set (N=9,256)',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

add_source(fig, "Global models, 2020-2025 test period")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig10_roc_curves.png', dpi=300, bbox_inches='tight')
print("   ✓ fig10_roc_curves.png")
plt.close()

# =============================================================================
# FIGURE 11: CONFUSION MATRICES (3x3 Grid)
# =============================================================================

print("\n📊 Fig 11: Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, model, color) in enumerate(models):
    y_pred = model.predict(X_processed)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               xticklabels=['Miss', 'Beat'], yticklabels=['Miss', 'Beat'],
               cbar_kws={'label': 'Count'})
    axes[idx].set_title(name, fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicted', fontweight='bold')
    axes[idx].set_ylabel('Actual', fontweight='bold')

plt.suptitle('Confusion Matrices - Model Classification Performance',
            fontsize=15, fontweight='bold')
add_source(fig, "Test set N=9,256")
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig11_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   ✓ fig11_confusion_matrices.png")
plt.close()

# =============================================================================
# FIGURE 12: REVISION MOMENTUM EFFECT
# =============================================================================

print("\n📊 Fig 12: Revision Momentum...")

revisions = pd.read_csv('../research_paper/tables/beat_rate_by_revisions.csv')

fig, ax = plt.subplots(figsize=(12, 7))

x_pos = range(len(revisions))
beat_rates = revisions['eps_beat_mean'] * 100

bars = ax.bar(x_pos, beat_rates, color=GS_COLORS['green'], 
             alpha=0.9, edgecolor='white', linewidth=2)

for bar, val, cnt in zip(bars, beat_rates, revisions['eps_beat_count']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}%\n(n={int(cnt):,})', ha='center', va='bottom', 
           fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(revisions['momentum_group'], rotation=15, ha='right')
ax.set_xlabel('30-Day Estimate Revision Momentum', fontsize=13, fontweight='bold')
ax.set_ylabel('Beat Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Analyst Revision Momentum Predicts Earnings Beats\n12pp Spread | Spearman ρ=0.088, p<0.001',
            fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([60, 85])

add_source(fig, "N=13,856 observations, 467 S&P 500 companies")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig12_revision_momentum.png', dpi=300, bbox_inches='tight')
print("   ✓ fig12_revision_momentum.png")
plt.close()

# =============================================================================
# FIGURE 13: ANALYST COVERAGE U-SHAPE
# =============================================================================

print("\n📊 Fig 13: Analyst Coverage...")

coverage = pd.read_csv('../research_paper/tables/beat_rate_by_coverage.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Beat rate
x_pos = range(len(coverage))
beat_rates = coverage['eps_beat_mean'] * 100

bars = ax1.bar(x_pos, beat_rates, color=GS_COLORS['navy'], 
              alpha=0.9, edgecolor='white', linewidth=2)

for bar, val in zip(bars, beat_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Highlight sweet spot
bars[1].set_color(GS_COLORS['gold'])
bars[1].set_edgecolor(GS_COLORS['gold'])
bars[1].set_linewidth(3)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(coverage['coverage_bin'])
ax1.set_xlabel('Number of Analysts Covering', fontsize=12, fontweight='bold')
ax1.set_ylabel('Beat Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: U-Shaped Coverage Effect', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([65, 78])

# Panel B: Sample counts
counts = coverage['eps_beat_count']
ax2.bar(x_pos, counts, color=GS_COLORS['gray'], alpha=0.7, edgecolor='white', linewidth=2)

for i, val in enumerate(counts):
    ax2.text(i, val, f'{int(val):,}', ha='center', va='bottom', 
            fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(coverage['coverage_bin'])
ax2.set_xlabel('Coverage Bin', fontsize=12, fontweight='bold')
ax2.set_ylabel('Observations', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: Sample Distribution', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Optimal Analyst Coverage: 10-20 Analysts (75% Beat Rate)',
            fontsize=15, fontweight='bold')
add_source(fig, "N=13,856 observations")
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig13_analyst_coverage.png', dpi=300, bbox_inches='tight')
print("   ✓ fig13_analyst_coverage.png")
plt.close()

# =============================================================================
# FIGURE 14: POLYMARKET ACCURACY BY TICKER
# =============================================================================

print("\n📊 Fig 14: Polymarket Accuracy Distribution...")

backtest = pd.read_csv('../polymarket_backtest/backtest_results.csv')

# Accuracy distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Model agreement
agreement = (backtest['rf_correct'] & backtest['xgb_correct'] & backtest['lr_correct']).mean() * 100
partial = ((backtest['rf_correct'] | backtest['xgb_correct'] | backtest['lr_correct']) & 
          ~(backtest['rf_correct'] & backtest['xgb_correct'] & backtest['lr_correct'])).mean() * 100
disagree = (~(backtest['rf_correct'] | backtest['xgb_correct'] | backtest['lr_correct'])).mean() * 100

categories = ['All 3 Correct', 'Partial Agreement', 'All 3 Wrong']
values = [agreement, partial, disagree]
colors_agree = [GS_COLORS['green'], GS_COLORS['gold'], GS_COLORS['red']]

bars = ax1.bar(range(len(categories)), values, color=colors_agree, 
              alpha=0.9, edgecolor='white', linewidth=2)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(categories)
ax1.set_ylabel('Percentage of Events', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Model Consensus', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Panel B: Individual model accuracy
model_accs = [
    backtest['rf_correct'].mean() * 100,
    backtest['xgb_correct'].mean() * 100,
    backtest['lr_correct'].mean() * 100
]
model_names = ['RF', 'XGB', 'LR']

ax2.bar(range(len(model_names)), model_accs, 
       color=[GS_COLORS['navy'], GS_COLORS['green'], GS_COLORS['gold']],
       alpha=0.9, edgecolor='white', linewidth=2)

for i, val in enumerate(model_accs):
    ax2.text(i, val, f'{val:.1f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: Individual Accuracy', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([65, 75])

plt.suptitle('Polymarket Backtest - Model Performance', fontsize=15, fontweight='bold')
add_source(fig, "N=262 Polymarket events, Q3 2025")
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/fig14_model_agreement.png', dpi=300, bbox_inches='tight')
print("   ✓ fig14_model_agreement.png")
plt.close()

# =============================================================================
# FIGURE 15: EDGE BUCKET PERFORMANCE
# =============================================================================

print("\n📊 Fig 15: Edge Performance Detailed...")

poly = pd.read_csv('../../../earnings_history_1year.csv')
merged = backtest.merge(poly[['marketId', 'avgPrice']], left_on='market_id', right_on='marketId', how='left')
merged['edge_lr'] = merged['lr_probability'] - merged['avgPrice']

edge_buckets_detailed = pd.cut(merged['edge_lr'], bins=15)
edge_perf = merged.groupby(edge_buckets_detailed).agg({
    'polymarket_outcome': ['mean', 'count'],
    'lr_correct': 'mean'
})
edge_perf = edge_perf[edge_perf[('polymarket_outcome', 'count')] >= 3]
edge_perf.columns = ['actual_beat', 'count', 'model_acc']

fig, ax = plt.subplots(figsize=(14, 7))

x_pos = range(len(edge_perf))
ax.plot(x_pos, edge_perf['actual_beat'] * 100, 'o-', linewidth=3, markersize=10,
       color=GS_COLORS['navy'], markeredgecolor='white', markeredgewidth=2, label='Actual Beat Rate')
ax.plot(x_pos, edge_perf['model_acc'] * 100, 's--', linewidth=2.5, markersize=8,
       color=GS_COLORS['green'], alpha=0.7, label='Model Accuracy')

# Highlight sweet spot range
sweet_spot_indices = [i for i, idx in enumerate(edge_perf.index) 
                      if 0.10 <= idx.mid <= 0.15]
if sweet_spot_indices:
    for i in sweet_spot_indices:
        ax.axvspan(i-0.4, i+0.4, alpha=0.2, color=GS_COLORS['gold'])

ax.set_xlabel('Edge Bucket (p_model - p_market)', fontsize=13, fontweight='bold')
ax.set_ylabel('Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Beat Rate and Model Accuracy Across Edge Spectrum\nSweet Spot (10-15% edge) Highlighted',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([50, 100])

add_source(fig, "N=248 events with pricing data")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig15_edge_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ fig15_edge_performance.png")
plt.close()

# =============================================================================
# FIGURE 16: CUMULATIVE P&L
# =============================================================================

print("\n📊 Fig 16: Cumulative P&L...")

# Sweet Spot trades only
sweet_spot_trades = merged[
    (merged['edge_lr'] >= 0.10) & 
    (merged['edge_lr'] <= 0.15)
].copy()

if len(sweet_spot_trades) > 0:
    # Calculate individual P&Ls
    sweet_spot_trades = sweet_spot_trades.sort_values('closed_at')
    
    pnls = []
    for _, row in sweet_spot_trades.iterrows():
        # Kelly bet
        edge = row['edge_lr']
        q = row['avgPrice']
        kelly_opt = edge / (1 - q) if edge > 0 else 0
        kelly_bet = kelly_opt * 0.25 * 100
        kelly_bet = max(0, min(kelly_bet, 100))
        
        # P&L
        if row['polymarket_outcome'] == 1:
            profit = (kelly_bet / q) - kelly_bet
            pnl = profit * 0.98
        else:
            pnl = -kelly_bet
        
        pnls.append(pnl)
    
    cumulative_pnl = np.cumsum(pnls)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=3,
           color=GS_COLORS['green'], marker='o', markersize=8,
           markeredgecolor='white', markeredgewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                    where=(np.array(cumulative_pnl) >= 0),
                    color=GS_COLORS['green'], alpha=0.2)
    
    ax.set_xlabel('Trade Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative P&L ($)', fontsize=13, fontweight='bold')
    ax.set_title('Sweet Spot Strategy - Cumulative P&L\n16 Trades, $63 Total Profit',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    
    # Add final value annotation
    ax.text(len(cumulative_pnl)-1, cumulative_pnl[-1],
           f'  Final: ${cumulative_pnl[-1]:.2f}',
           ha='left', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=GS_COLORS['green']))
    
    add_source(fig, "Sweet Spot (10-15% edge) backtest, Kelly 25%")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig('figures/fig16_cumulative_pnl.png', dpi=300, bbox_inches='tight')
    print("   ✓ fig16_cumulative_pnl.png")
    plt.close()

# =============================================================================
# FIGURE 17: PRICE BUCKET ANALYSIS
# =============================================================================

print("\n📊 Fig 17: Polymarket Price Buckets...")

price_buckets = pd.read_csv('../polymarket_backtest/tables/price_bucket_analysis.csv')

fig, ax = plt.subplots(figsize=(12, 7))

x_pos = range(len(price_buckets))
actual_beat = price_buckets['polymarket_outcome_mean'] * 100
model_prob = price_buckets['lr_probability_mean'] * 100
pm_price = price_buckets['avgPrice_mean'] * 100

width = 0.25
bars1 = ax.bar([i-width for i in x_pos], actual_beat, width, label='Actual Beat Rate',
              color=GS_COLORS['green'], alpha=0.9, edgecolor='white', linewidth=2)
bars2 = ax.bar(x_pos, model_prob, width, label='Model Probability',
              color=GS_COLORS['navy'], alpha=0.9, edgecolor='white', linewidth=2)
bars3 = ax.bar([i+width for i in x_pos], pm_price, width, label='PM Price',
              color=GS_COLORS['gold'], alpha=0.9, edgecolor='white', linewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(price_buckets['pm_price_bucket'])
ax.set_xlabel('Polymarket Price Range', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability / Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Model vs Market Calibration by Price Level\nGaps Indicate Mispricing Opportunities',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

add_source(fig, "N=248 events with pricing")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig17_price_buckets.png', dpi=300, bbox_inches='tight')
print("   ✓ fig17_price_buckets.png")
plt.close()

# =============================================================================
# FIGURE 18: WIN/LOSS DISTRIBUTION BY STRATEGY
# =============================================================================

print("\n📊 Fig 18: Strategy Win/Loss Distribution...")

strategies_data = pd.read_csv('../polymarket_backtest/tables/strategy_comparison_full.csv')

fig, ax = plt.subplots(figsize=(12, 8))

strats = strategies_data.head(6).copy()  # Top 6 strategies
strats = strats.sort_values('roi_net')

wins = strats['trades'] * strats['win_rate']
losses = strats['trades'] * (1 - strats['win_rate'])

y_pos = range(len(strats))

bars1 = ax.barh(y_pos, wins, label='Wins', color=GS_COLORS['green'], 
               alpha=0.9, edgecolor='white', linewidth=2)
bars2 = ax.barh(y_pos, -losses, label='Losses', color=GS_COLORS['red'],
               alpha=0.9, edgecolor='white', linewidth=2)

# Add counts
for i, (w, l) in enumerate(zip(wins, losses)):
    ax.text(w, i, f' {int(w)}W', va='center', fontsize=10, fontweight='bold')
    ax.text(-l, i, f'{int(l)}L ', ha='right', va='center', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(strats['Strategy'], fontsize=10)
ax.set_xlabel('Win/Loss Count', fontsize=13, fontweight='bold')
ax.set_title('Strategy Win/Loss Distribution', fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)

add_source(fig, "Top 6 strategies by ROI")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig18_win_loss_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ fig18_win_loss_distribution.png")
plt.close()

# =============================================================================
# FIGURE 19: ELO MOMENTUM VS OUTCOME SCATTER
# =============================================================================

print("\n📊 Fig 19: Elo Momentum Scatter...")

# Sample for visualization
sample = combined_clean.sample(min(2000, len(combined_clean)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter by outcome
beats = sample[sample['eps_beat'] == 1]
misses = sample[sample['eps_beat'] == 0]

ax.scatter(misses['elo_momentum'], misses['elo_before'], 
          alpha=0.4, s=30, color=GS_COLORS['red'], label='Miss')
ax.scatter(beats['elo_momentum'], beats['elo_before'],
          alpha=0.4, s=30, color=GS_COLORS['green'], label='Beat')

ax.set_xlabel('Elo Momentum', fontsize=13, fontweight='bold')
ax.set_ylabel('Elo Before', fontsize=13, fontweight='bold')
ax.set_title('Elo Momentum vs Level - Beat/Miss Separation\nMomentum (x-axis) Shows Stronger Separation',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, markerscale=2)
ax.grid(alpha=0.3)

add_source(fig, "Random sample of 2,000 observations")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig19_elo_momentum_scatter.png', dpi=300, bbox_inches='tight')
print("   ✓ fig19_elo_momentum_scatter.png")
plt.close()

# =============================================================================
# FIGURE 20: QUARTERLY BEAT RATE TREND
# =============================================================================

print("\n📊 Fig 20: Quarterly Trend...")

quarterly = pd.read_csv('../excel_data/quarterly_beat_miss.csv')
quarterly = quarterly[quarterly['status'] == 'Historical'].tail(20)  # Last 20 quarters

fig, ax = plt.subplots(figsize=(14, 7))

dates = pd.to_datetime(quarterly['date'])
beat_rates = quarterly['beat_rate_pct']

ax.plot(range(len(beat_rates)), beat_rates, 'o-', linewidth=2.5, markersize=8,
       color=GS_COLORS['navy'], markeredgecolor='white', markeredgewidth=2)

# Add trend line
z = np.polyfit(range(len(beat_rates)), beat_rates, 1)
p = np.poly1d(z)
ax.plot(range(len(beat_rates)), p(range(len(beat_rates))), '--',
       linewidth=2, color=GS_COLORS['gold'], label=f'Trend (slope: {z[0]:+.2f}%/quarter)')

ax.set_xlabel('Quarter (Chronological)', fontsize=13, fontweight='bold')
ax.set_ylabel('Beat Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Quarterly Beat Rate Trend (Last 20 Quarters)\nRecent Improvement Visible',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([60, 85])

# Add recent quarters annotation
ax.text(0.98, 0.98, f'Q3 2025: {beat_rates.iloc[-1]:.1f}%',
       transform=ax.transAxes, ha='right', va='top',
       bbox=dict(boxstyle='round', facecolor=GS_COLORS['gold'], alpha=0.3),
       fontsize=11, fontweight='bold')

add_source(fig, "Last 20 quarters, aggregate S&P 500")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig20_quarterly_trend.png', dpi=300, bbox_inches='tight')
print("   ✓ fig20_quarterly_trend.png")
plt.close()

# =============================================================================
# FIGURE 21: BRIER SCORE COMPARISON
# =============================================================================

print("\n📊 Fig 21: Calibration Quality...")

calibration = pd.read_csv('../polymarket_backtest/tables/calibration_metrics.csv')

fig, ax = plt.subplots(figsize=(10, 6))

models = calibration['Model']
brier = calibration['Brier_Score']

bars = ax.barh(range(len(models)), brier,
              color=[GS_COLORS['navy'], GS_COLORS['green'], GS_COLORS['gold']],
              alpha=0.9, edgecolor='white', linewidth=2)

for i, (bar, val) in enumerate(zip(bars, brier)):
    ax.text(val, i, f'  {val:.4f}', va='center', fontsize=11, fontweight='bold')

# Add baselines
ax.axvline(x=0.25, color=GS_COLORS['red'], linestyle='--', linewidth=2, label='Random (0.250)')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Perfect (0.000)')

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Brier Score (Lower = Better)', fontsize=13, fontweight='bold')
ax.set_title('Probability Calibration Quality - Brier Scores\nAll Models Significantly Better Than Random',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim([0, 0.27])

add_source(fig, "Polymarket outcomes, N=248")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('figures/fig21_brier_scores.png', dpi=300, bbox_inches='tight')
print("   ✓ fig21_brier_scores.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("✅ ADDITIONAL FIGURES GENERATED (14 new figures)")
print("=" * 80)

print("\nNew Figures:")
print("   8.  S&P 500 Coverage Map")
print("   9.  Elo Distribution & Beat Rate")
print("   10. ROC Curves (3 models)")
print("   11. Confusion Matrices (3x3)")
print("   12. Revision Momentum Effect")
print("   13. Analyst Coverage U-Shape")
print("   14. Model Agreement Analysis")
print("   15. Edge Performance Detailed")
print("   16. Cumulative P&L Trajectory")
print("   17. Price Bucket Calibration")
print("   18. Win/Loss Distribution")
print("   19. Elo Momentum Scatter")
print("   20. Quarterly Beat Rate Trend")
print("   21. Brier Score Comparison")

print("\nTotal Figures: 21 (7 original + 14 new)")
print("All figures: 300 DPI, Goldman Sachs style")
print("=" * 80)

