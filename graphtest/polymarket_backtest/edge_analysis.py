#!/usr/bin/env python3
"""
Polymarket Edge Analysis & EV Optimization
===========================================

Analyzes the edge between ML model probabilities and Polymarket prices.
Identifies profitable trading opportunities with positive expected value.

Key Metrics:
- Edge = p_model - p_market
- Brier Score = Probability accuracy
- Expected Value (EV) = Kelly criterion based
- ROI = Return on investment with fees

Output:
    tables/edge_bucket_analysis.csv
    tables/price_bucket_analysis.csv
    tables/calibration_metrics.csv
    tables/tradeable_opportunities.csv
    tables/pnl_simulation.csv
    figures/edge_distribution.png
    figures/calibration_curves.png
    figures/ev_by_edge_bucket.png
    figures/strategy_comparison.png
    edge_analysis_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

print("=" * 80)
print("POLYMARKET EDGE ANALYSIS & EV OPTIMIZATION")
print("=" * 80)

# Create directories
import os
os.makedirs('tables', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print(f"\n📊 1/7 Loading data...")

# Backtest results (ML predictions)
backtest_df = pd.read_csv('backtest_results.csv')

# Original Polymarket data (for avgPrice)
poly_df = pd.read_csv('/Users/ibrahimfiratsoysal/Documents/earnings_history_1year.csv')

# Merge to get avgPrice
merged = backtest_df.merge(
    poly_df[['marketId', 'avgPrice']], 
    left_on='market_id', 
    right_on='marketId', 
    how='left'
)

print(f"   • Backtest events: {len(backtest_df)}")
print(f"   • Merged with avgPrice: {len(merged)}")
print(f"   • Events with avgPrice: {merged['avgPrice'].notna().sum()}")

# Use merged data
df = merged.copy()

# =============================================================================
# 2. CALCULATE EDGE (p_model - p_market)
# =============================================================================

print(f"\n📈 2/7 Calculating edge...")

# Edge for each model
df['edge_rf'] = df['rf_probability'] - df['avgPrice']
df['edge_xgb'] = df['xgb_probability'] - df['avgPrice']
df['edge_lr'] = df['lr_probability'] - df['avgPrice']

print(f"   • Edge calculated for all 3 models")
print(f"\n   Edge statistics (LR):")
print(f"      Mean: {df['edge_lr'].mean():+.4f}")
print(f"      Std:  {df['edge_lr'].std():.4f}")
print(f"      Min:  {df['edge_lr'].min():+.4f}")
print(f"      Max:  {df['edge_lr'].max():+.4f}")

# Edge buckets
edge_buckets = [-1, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 1]
edge_labels = ['<-20%', '-20to-15%', '-15to-10%', '-10to-5%', '-5to0%', 
               '0to5%', '5to10%', '10to15%', '15to20%', '>20%']

df['edge_bucket_rf'] = pd.cut(df['edge_rf'], bins=edge_buckets, labels=edge_labels)
df['edge_bucket_xgb'] = pd.cut(df['edge_xgb'], bins=edge_buckets, labels=edge_labels)
df['edge_bucket_lr'] = pd.cut(df['edge_lr'], bins=edge_buckets, labels=edge_labels)

# Analyze by edge bucket (LR - best model)
edge_analysis = df.groupby('edge_bucket_lr', observed=True).agg({
    'polymarket_outcome': ['mean', 'count', 'std'],
    'lr_probability': 'mean',
    'avgPrice': 'mean',
    'edge_lr': 'mean',
    'lr_correct': 'mean'
}).round(4)

edge_analysis.columns = ['_'.join(col).strip() for col in edge_analysis.columns.values]
edge_analysis = edge_analysis.reset_index()

edge_analysis.to_csv('tables/edge_bucket_analysis.csv', index=False)
print(f"   ✓ Saved: tables/edge_bucket_analysis.csv")

# =============================================================================
# 3. CALIBRATION ANALYSIS (Brier Score)
# =============================================================================

print(f"\n🎯 3/7 Calibration analysis...")

y_true = df['polymarket_outcome'].values

# Brier scores
brier_rf = brier_score_loss(y_true, df['rf_probability'])
brier_xgb = brier_score_loss(y_true, df['xgb_probability'])
brier_lr = brier_score_loss(y_true, df['lr_probability'])

print(f"   Brier Scores (lower = better):")
print(f"      Random Forest:        {brier_rf:.4f}")
print(f"      XGBoost:              {brier_xgb:.4f}")
print(f"      Logistic Regression:  {brier_lr:.4f}")

# Calibration curves
cal_metrics = []

for model_name, prob_col in [('Random Forest', 'rf_probability'),
                              ('XGBoost', 'xgb_probability'),
                              ('Logistic Regression', 'lr_probability')]:
    
    fraction_pos, mean_pred = calibration_curve(
        y_true, df[prob_col], n_bins=10, strategy='quantile'
    )
    
    cal_metrics.append({
        'Model': model_name,
        'Brier_Score': brier_rf if model_name == 'Random Forest' else (brier_xgb if model_name == 'XGBoost' else brier_lr),
        'Calibration_Points': len(fraction_pos)
    })

cal_df = pd.DataFrame(cal_metrics)
cal_df.to_csv('tables/calibration_metrics.csv', index=False)
print(f"   ✓ Saved: tables/calibration_metrics.csv")

# =============================================================================
# 4. PRICE BUCKET ANALYSIS
# =============================================================================

print(f"\n💰 4/7 Price bucket analysis...")

# Polymarket price buckets
pm_buckets = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
pm_labels = ['0.0-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-1.0']

df['pm_price_bucket'] = pd.cut(df['avgPrice'], bins=pm_buckets, labels=pm_labels)

# Analyze by price bucket
price_analysis = df.groupby('pm_price_bucket', observed=True).agg({
    'polymarket_outcome': ['mean', 'count'],
    'lr_probability': 'mean',
    'avgPrice': 'mean',
    'edge_lr': 'mean',
    'lr_correct': 'mean'
}).round(4)

price_analysis.columns = ['_'.join(col).strip() for col in price_analysis.columns.values]
price_analysis = price_analysis.reset_index()

price_analysis.to_csv('tables/price_bucket_analysis.csv', index=False)
print(f"   ✓ Saved: tables/price_bucket_analysis.csv")

# =============================================================================
# 5. EXPECTED VALUE (EV) & KELLY CRITERION
# =============================================================================

print(f"\n💵 5/7 Calculating EV and optimal bet sizes...")

def calculate_ev_kelly(p_model, q_market, bet_size=100, kelly_fraction=0.25):
    """
    Calculate expected value and Kelly bet size
    
    Args:
        p_model: Model probability (0-1)
        q_market: Market price / probability (0-1)
        bet_size: Base stake
        kelly_fraction: Fraction of Kelly (default 0.25 = quarter Kelly)
    
    Returns:
        (ev, optimal_bet, payout_if_win, loss_if_lose)
    """
    if q_market >= 0.99 or q_market <= 0.01:
        return 0, 0, 0, 0
    
    # Number of shares we can buy
    shares = bet_size / q_market
    
    # Payout if win ($1 per share)
    payout = shares * 1.0
    profit = payout - bet_size
    
    # Kelly criterion
    edge = p_model - q_market
    kelly_optimal = edge / (1 - q_market) if edge > 0 else 0
    kelly_bet = kelly_optimal * kelly_fraction * bet_size
    kelly_bet = max(0, min(kelly_bet, bet_size))  # Clamp to [0, bet_size]
    
    # EV with Kelly bet
    if kelly_bet > 0:
        kelly_shares = kelly_bet / q_market
        kelly_payout = kelly_shares * 1.0
        kelly_profit = kelly_payout - kelly_bet
        ev = (p_model * kelly_profit) - ((1 - p_model) * kelly_bet)
    else:
        ev = 0
    
    return ev, kelly_bet, profit, bet_size

# Calculate for LR (best model)
df['ev_lr'], df['optimal_bet_lr'], df['win_profit_lr'], df['lose_loss_lr'] = zip(
    *df.apply(lambda x: calculate_ev_kelly(x['lr_probability'], x['avgPrice']), axis=1)
)

print(f"   • EV calculated for all events")
print(f"   • Average EV (LR): ${df['ev_lr'].mean():.2f} per event")
print(f"   • Positive EV events: {(df['ev_lr'] > 0).sum()}/{len(df)}")

# =============================================================================
# 6. TRADING STRATEGY FILTERS
# =============================================================================

print(f"\n🔍 6/7 Applying trading filters...")

# Filter criteria
MIN_EDGE = 0.12
PROB_RANGE = (0.25, 0.90)

# Apply filters
tradeable = df[
    (df['edge_lr'].abs() >= MIN_EDGE) &
    (df['lr_probability'] >= PROB_RANGE[0]) &
    (df['lr_probability'] <= PROB_RANGE[1])
].copy()

print(f"   • Filter: |edge| >= {MIN_EDGE:.0%}")
print(f"   • Filter: probability in {PROB_RANGE}")
print(f"   • Tradeable events: {len(tradeable)}/{len(df)} ({len(tradeable)/len(df):.1%})")

if len(tradeable) > 0:
    print(f"   • Tradeable accuracy: {tradeable['lr_correct'].mean():.2%}")
    print(f"   • Tradeable avg EV: ${tradeable['ev_lr'].mean():.2f}")

tradeable.to_csv('tables/tradeable_opportunities.csv', index=False)
print(f"   ✓ Saved: tables/tradeable_opportunities.csv")

# =============================================================================
# 7. P&L SIMULATION
# =============================================================================

print(f"\n💰 7/7 P&L simulation...")

# Strategy 1: Trade all events with Kelly-weighted bets
all_trades = df.copy()
all_trades['bet'] = all_trades['optimal_bet_lr']  # Use Kelly criterion!

# Calculate actual P&L based on Kelly bet
all_trades['pnl_gross'] = all_trades.apply(
    lambda x: (x['bet'] / x['avgPrice'] - x['bet']) if x['polymarket_outcome'] == 1 else -x['bet'],
    axis=1
)

# Polymarket fees: 2% on winnings
all_trades['pnl_net'] = all_trades.apply(
    lambda x: x['pnl_gross'] * 0.98 if x['pnl_gross'] > 0 else x['pnl_gross'],
    axis=1
)

# Strategy 2: Trade only filtered (high edge) with Kelly-weighted bets
if len(tradeable) > 0:
    tradeable_sim = tradeable.copy()
    tradeable_sim['bet'] = tradeable_sim['optimal_bet_lr']  # Use Kelly criterion!
    
    # Calculate actual P&L based on Kelly bet
    tradeable_sim['pnl_gross'] = tradeable_sim.apply(
        lambda x: (x['bet'] / x['avgPrice'] - x['bet']) if x['polymarket_outcome'] == 1 else -x['bet'],
        axis=1
    )
    tradeable_sim['pnl_net'] = tradeable_sim.apply(
        lambda x: x['pnl_gross'] * 0.98 if x['pnl_gross'] > 0 else x['pnl_gross'],
        axis=1
    )
else:
    tradeable_sim = pd.DataFrame()

# Results
strategies = []

# All trades
strategies.append({
    'Strategy': 'All Events',
    'Trades': len(all_trades),
    'Total_Invested': all_trades['bet'].sum(),
    'Gross_PnL': all_trades['pnl_gross'].sum(),
    'Net_PnL': all_trades['pnl_net'].sum(),
    'ROI_Gross': (all_trades['pnl_gross'].sum() / all_trades['bet'].sum()) * 100,
    'ROI_Net': (all_trades['pnl_net'].sum() / all_trades['bet'].sum()) * 100,
    'Win_Rate': (all_trades['polymarket_outcome'] == 1).mean(),
    'Avg_Edge': all_trades['edge_lr'].mean()
})

# Filtered trades
if len(tradeable_sim) > 0:
    strategies.append({
        'Strategy': f'Filtered (|edge| >= {MIN_EDGE:.0%})',
        'Trades': len(tradeable_sim),
        'Total_Invested': tradeable_sim['bet'].sum(),
        'Gross_PnL': tradeable_sim['pnl_gross'].sum(),
        'Net_PnL': tradeable_sim['pnl_net'].sum(),
        'ROI_Gross': (tradeable_sim['pnl_gross'].sum() / tradeable_sim['bet'].sum()) * 100,
        'ROI_Net': (tradeable_sim['pnl_net'].sum() / tradeable_sim['bet'].sum()) * 100,
        'Win_Rate': (tradeable_sim['polymarket_outcome'] == 1).mean(),
        'Avg_Edge': tradeable_sim['edge_lr'].mean()
    })

# Baseline: Always bet on beat
baseline_beat = df.copy()
baseline_beat['pnl_gross'] = baseline_beat.apply(
    lambda x: (100/x['avgPrice'] - 100) if x['polymarket_outcome'] == 1 else -100,
    axis=1
)
baseline_beat['pnl_net'] = baseline_beat.apply(
    lambda x: x['pnl_gross'] * 0.98 if x['pnl_gross'] > 0 else x['pnl_gross'],
    axis=1
)

strategies.append({
    'Strategy': 'Baseline (Always Beat)',
    'Trades': len(baseline_beat),
    'Total_Invested': len(baseline_beat) * 100,
    'Gross_PnL': baseline_beat['pnl_gross'].sum(),
    'Net_PnL': baseline_beat['pnl_net'].sum(),
    'ROI_Gross': (baseline_beat['pnl_gross'].sum() / (len(baseline_beat) * 100)) * 100,
    'ROI_Net': (baseline_beat['pnl_net'].sum() / (len(baseline_beat) * 100)) * 100,
    'Win_Rate': baseline_beat['polymarket_outcome'].mean(),
    'Avg_Edge': 0
})

pnl_df = pd.DataFrame(strategies)
pnl_df.to_csv('tables/pnl_simulation.csv', index=False)
print(f"   ✓ Saved: tables/pnl_simulation.csv")

print(f"\n   Strategy Comparison:")
for _, row in pnl_df.iterrows():
    print(f"      {row['Strategy']:30s}: ROI={row['ROI_Net']:+6.1f}%, Trades={int(row['Trades']):3d}")

# =============================================================================
# 8. GENERATE FIGURES
# =============================================================================

print(f"\n🎨 Generating figures...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10})

COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#06A77D', 'danger': '#D90429'}

# Figure 1: Edge Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['edge_lr'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Edge')
ax1.set_xlabel('Edge (p_model - p_market)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Edge Distribution (Logistic Regression)', fontweight='bold', pad=15)
ax1.legend()
ax1.grid(alpha=0.3)

# Cumulative distribution
sorted_edges = np.sort(df['edge_lr'])
cumulative = np.arange(1, len(sorted_edges) + 1) / len(sorted_edges)
ax2.plot(sorted_edges, cumulative, linewidth=2, color=COLORS['primary'])
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=MIN_EDGE, color='green', linestyle='--', linewidth=2, label=f'Trade threshold ({MIN_EDGE:.0%})')
ax2.axvline(x=-MIN_EDGE, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Edge', fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontweight='bold')
ax2.set_title('Cumulative Edge Distribution', fontweight='bold', pad=15)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/edge_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/edge_distribution.png")
plt.close()

# Figure 2: Calibration Curves
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, prob_col, color in [('Random Forest', 'rf_probability', COLORS['primary']),
                                      ('XGBoost', 'xgb_probability', COLORS['secondary']),
                                      ('Logistic Regression', 'lr_probability', COLORS['success'])]:
    
    fraction_pos, mean_pred = calibration_curve(y_true, df[prob_col], n_bins=10, strategy='quantile')
    ax.plot(mean_pred, fraction_pos, 's-', linewidth=2, markersize=8, label=model_name, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives (Actual Beat Rate)', fontsize=12, fontweight='bold')
ax.set_title('Probability Calibration Curves\n(Polymarket Outcomes)', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/calibration_curves.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/calibration_curves.png")
plt.close()

# Figure 3: EV by Edge Bucket
fig, ax = plt.subplots(figsize=(12, 6))

# Aggregate EV by edge bucket
ev_by_bucket = df.groupby('edge_bucket_lr', observed=True).agg({
    'ev_lr': ['sum', 'mean', 'count'],
    'polymarket_outcome': 'mean'
}).round(2)

ev_by_bucket.columns = ['_'.join(col).strip() for col in ev_by_bucket.columns.values]
ev_by_bucket = ev_by_bucket.reset_index()

x_pos = range(len(ev_by_bucket))
bars = ax.bar(x_pos, ev_by_bucket['ev_lr_sum'], color=COLORS['success'], alpha=0.8, edgecolor='white', linewidth=2)

for bar, val, cnt in zip(bars, ev_by_bucket['ev_lr_sum'], ev_by_bucket['ev_lr_count']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${val:.0f}\n(n={int(cnt)})',
           ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Edge Bucket (p_model - p_market)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total EV ($)', fontsize=12, fontweight='bold')
ax.set_title('Expected Value by Edge Bucket (Kelly 25%)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(edge_labels, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/ev_by_edge_bucket.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/ev_by_edge_bucket.png")
plt.close()

# Figure 4: Strategy Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROI comparison
strategies_plot = pnl_df.sort_values('ROI_Net', ascending=False)
bars = ax1.barh(range(len(strategies_plot)), strategies_plot['ROI_Net'], 
               color=[COLORS['success'] if x > 0 else COLORS['danger'] for x in strategies_plot['ROI_Net']],
               alpha=0.8, edgecolor='white', linewidth=2)

for i, (bar, val) in enumerate(zip(bars, strategies_plot['ROI_Net'])):
    ax1.text(val, i, f'  {val:+.1f}%', va='center', fontsize=11, fontweight='bold')

ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_yticks(range(len(strategies_plot)))
ax1.set_yticklabels(strategies_plot['Strategy'])
ax1.set_xlabel('ROI (Net, After Fees)', fontsize=12, fontweight='bold')
ax1.set_title('Strategy ROI Comparison', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

# Win rate comparison
ax2.barh(range(len(strategies_plot)), strategies_plot['Win_Rate'] * 100,
        color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=2)

for i, val in enumerate(strategies_plot['Win_Rate'] * 100):
    ax2.text(val, i, f'  {val:.1f}%', va='center', fontsize=11, fontweight='bold')

ax2.set_yticks(range(len(strategies_plot)))
ax2.set_yticklabels(strategies_plot['Strategy'])
ax2.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Strategy Win Rate', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/strategy_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/strategy_comparison.png")
plt.close()

# =============================================================================
# 9. COMPREHENSIVE REPORT
# =============================================================================

print(f"\n📄 Generating comprehensive report...")

with open('edge_analysis_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("POLYMARKET EDGE ANALYSIS & EV OPTIMIZATION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total events analyzed: {len(df)}\n")
    f.write(f"Best ML model: Logistic Regression (70.23% accuracy)\n")
    f.write(f"Average edge (LR): {df['edge_lr'].mean():+.4f}\n")
    f.write(f"Average EV per event: ${df['ev_lr'].mean():.2f}\n\n")
    
    f.write("EDGE BUCKET ANALYSIS (Logistic Regression)\n")
    f.write("-" * 80 + "\n")
    f.write(edge_analysis.to_string(index=False))
    f.write("\n\n")
    
    f.write("CALIBRATION METRICS\n")
    f.write("-" * 80 + "\n")
    f.write(cal_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("PRICE BUCKET ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(price_analysis.to_string(index=False))
    f.write("\n\n")
    
    f.write("STRATEGY PERFORMANCE (After 2% Fees)\n")
    f.write("-" * 80 + "\n")
    f.write(pnl_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("TRADING RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    
    # Find best edge bucket
    positive_edge = edge_analysis[edge_analysis['edge_lr_mean'] > 0]
    if len(positive_edge) > 0:
        best_bucket = positive_edge.loc[positive_edge['lr_correct_mean'].idxmax()]
        f.write(f"Best Edge Bucket: {best_bucket['edge_bucket_lr']}\n")
        f.write(f"  Actual beat rate: {best_bucket['polymarket_outcome_mean']:.2%}\n")
        f.write(f"  Model accuracy: {best_bucket['lr_correct_mean']:.2%}\n")
        f.write(f"  Sample size: {int(best_bucket['polymarket_outcome_count'])}\n\n")
    
    # Overall recommendation
    best_strat = pnl_df.loc[pnl_df['ROI_Net'].idxmax()]
    f.write(f"\nRecommended Strategy: {best_strat['Strategy']}\n")
    f.write(f"  Expected ROI: {best_strat['ROI_Net']:+.2f}%\n")
    f.write(f"  Win rate: {best_strat['Win_Rate']:.2%}\n")
    f.write(f"  Total trades: {int(best_strat['Trades'])}\n")
    
    if best_strat['ROI_Net'] > 0:
        f.write(f"\n✅ POSITIVE EXPECTED VALUE DETECTED\n")
    else:
        f.write(f"\n⚠️  NEGATIVE/NEUTRAL EXPECTED VALUE\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF ANALYSIS\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ edge_analysis_report.txt")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "=" * 80)
print(f"✅ EDGE ANALYSIS COMPLETE")
print(f"=" * 80)

print(f"\nKey Findings:")
print(f"   • Brier Score (LR): {brier_lr:.4f} (lower = better)")
print(f"   • Average Edge: {df['edge_lr'].mean():+.4f}")
print(f"   • Positive EV events: {(df['ev_lr'] > 0).sum()}/{len(df)}")

print(f"\nBest Strategy:")
print(f"   • {best_strat['Strategy']}")
print(f"   • ROI (after fees): {best_strat['ROI_Net']:+.2f}%")
print(f"   • Total P&L: ${best_strat['Net_PnL']:+,.2f}")

print(f"\nOutputs:")
print(f"   📋 Tables: tables/")
print(f"   📊 Figures: figures/")
print(f"   📄 Report: edge_analysis_report.txt")

print(f"\n" + "=" * 80)

