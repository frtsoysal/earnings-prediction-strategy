#!/usr/bin/env python3
"""
Strategy Optimization - Multiple Approach Testing
==================================================

Tests various trading strategies to maximize ROI:
1. Sweet Spot (10-15% edge)
2. High Conviction
3. Different Kelly fractions (10%, 25%, 50%, 75%)
4. Counter-trade (negative edge → bet NO)
5. Combination strategies

Output:
    tables/strategy_comparison.csv
    tables/kelly_fraction_comparison.csv
    figures/strategy_roi_comparison.png
    figures/kelly_optimization.png
    strategy_optimization_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("STRATEGY OPTIMIZATION - ROI MAXIMIZATION")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print(f"\n📊 Loading data...")

# Backtest + Polymarket merged
backtest = pd.read_csv('backtest_results.csv')
poly = pd.read_csv('/Users/ibrahimfiratsoysal/Documents/earnings_history_1year.csv')

df = backtest.merge(poly[['marketId', 'avgPrice']], left_on='market_id', right_on='marketId', how='left')

# Calculate edge
df['edge_lr'] = df['lr_probability'] - df['avgPrice']

# NO price (Polymarket: NO = 1 - YES)
df['no_price'] = 1 - df['avgPrice']

print(f"   ✓ {len(df)} events loaded")
print(f"   ✓ avgPrice: {df['avgPrice'].notna().sum()} events")

# =============================================================================
# 2. KELLY FRACTION COMPARISON (10%, 25%, 50%, 75%)
# =============================================================================

print(f"\n🎲 Testing different Kelly fractions...")

def calculate_kelly_pnl(df, kelly_fraction=0.25, base_bet=100):
    """Calculate P&L with given Kelly fraction"""
    results = []
    
    for _, row in df.iterrows():
        p_model = row['lr_probability']
        q_market = row['avgPrice']
        outcome = row['polymarket_outcome']
        
        if pd.isna(q_market) or q_market >= 0.99 or q_market <= 0.01:
            continue
        
        # Kelly bet
        edge = p_model - q_market
        kelly_optimal = edge / (1 - q_market) if edge > 0 else 0
        kelly_bet = kelly_optimal * kelly_fraction * base_bet
        kelly_bet = max(0, min(kelly_bet, base_bet))
        
        # P&L
        if kelly_bet > 0:
            if outcome == 1:  # Beat
                shares = kelly_bet / q_market
                payout = shares * 1.0
                pnl_gross = payout - kelly_bet
                pnl_net = pnl_gross * 0.98  # 2% fee
            else:  # Miss
                pnl_gross = -kelly_bet
                pnl_net = pnl_gross
            
            results.append({
                'bet': kelly_bet,
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'outcome': outcome,
                'edge': edge
            })
    
    results_df = pd.DataFrame(results)
    
    return {
        'trades': len(results_df),
        'total_invested': results_df['bet'].sum(),
        'gross_pnl': results_df['pnl_gross'].sum(),
        'net_pnl': results_df['pnl_net'].sum(),
        'roi_net': (results_df['pnl_net'].sum() / results_df['bet'].sum()) * 100 if results_df['bet'].sum() > 0 else 0,
        'win_rate': results_df['outcome'].mean()
    }

kelly_fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
kelly_results = []

for kf in kelly_fractions:
    result = calculate_kelly_pnl(df, kelly_fraction=kf)
    result['kelly_fraction'] = kf
    kelly_results.append(result)
    print(f"   Kelly {kf:.0%}: ROI={result['roi_net']:+6.2f}%, Invested=${result['total_invested']:.0f}")

kelly_df = pd.DataFrame(kelly_results)
kelly_df.to_csv('tables/kelly_fraction_comparison.csv', index=False)
print(f"   ✓ Saved: tables/kelly_fraction_comparison.csv")

# =============================================================================
# 3. EDGE-BASED STRATEGIES
# =============================================================================

print(f"\n🎯 Testing edge-based strategies...")

strategies = []

# Strategy 1: All events (current)
all_result = calculate_kelly_pnl(df, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'All Events (Kelly 25%)',
    **all_result
})

# Strategy 2: Sweet Spot (10-15% edge)
sweet_spot = df[(df['edge_lr'] >= 0.10) & (df['edge_lr'] < 0.15)]
ss_result = calculate_kelly_pnl(sweet_spot, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'Sweet Spot (10-15% edge)',
    **ss_result
})

# Strategy 3: Positive edge only (>0%)
positive_edge = df[df['edge_lr'] > 0]
pe_result = calculate_kelly_pnl(positive_edge, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'Positive Edge Only (>0%)',
    **pe_result
})

# Strategy 4: High edge (>8%)
high_edge = df[df['edge_lr'] >= 0.08]
he_result = calculate_kelly_pnl(high_edge, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'High Edge (>=8%)',
    **he_result
})

# Strategy 5: Moderate edge (5-20%)
moderate = df[(df['edge_lr'] >= 0.05) & (df['edge_lr'] <= 0.20)]
mod_result = calculate_kelly_pnl(moderate, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'Moderate Edge (5-20%)',
    **mod_result
})

# Strategy 6: High Conviction (edge>10% AND prob>0.70)
high_conv = df[(df['edge_lr'] >= 0.10) & (df['lr_probability'] >= 0.70) & (df['lr_probability'] <= 0.85)]
hc_result = calculate_kelly_pnl(high_conv, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'High Conviction (edge>10%, prob 0.7-0.85)',
    **hc_result
})

# Strategy 7: Avoid overconfidence (edge>0 AND prob<0.80)
avoid_overconf = df[(df['edge_lr'] > 0) & (df['lr_probability'] < 0.80)]
ao_result = calculate_kelly_pnl(avoid_overconf, kelly_fraction=0.25)
strategies.append({
    'Strategy': 'Avoid Overconfidence (edge>0, prob<0.80)',
    **ao_result
})

# =============================================================================
# 4. COUNTER-TRADE STRATEGY (Bet NO on negative edge)
# =============================================================================

print(f"\n🔄 Testing counter-trade strategy (bet NO)...")

def calculate_no_bet_pnl(df_negative, kelly_fraction=0.25, base_bet=100):
    """Bet NO when model has negative edge"""
    results = []
    
    for _, row in df_negative.iterrows():
        p_model_miss = 1 - row['lr_probability']  # Probability of MISS
        q_no = row['no_price']  # NO price (1 - avgPrice)
        outcome = row['polymarket_outcome']
        
        if pd.isna(q_no) or q_no >= 0.99 or q_no <= 0.01:
            continue
        
        # Edge for NO bet
        edge_no = p_model_miss - q_no
        
        if edge_no > 0:  # Only bet if we have edge on NO
            kelly_optimal = edge_no / (1 - q_no)
            kelly_bet = kelly_optimal * kelly_fraction * base_bet
            kelly_bet = max(0, min(kelly_bet, base_bet))
            
            if kelly_bet > 0:
                if outcome == 0:  # Miss - NO wins!
                    shares = kelly_bet / q_no
                    payout = shares * 1.0
                    pnl_gross = payout - kelly_bet
                    pnl_net = pnl_gross * 0.98
                else:  # Beat - NO loses
                    pnl_gross = -kelly_bet
                    pnl_net = pnl_gross
                
                results.append({
                    'bet': kelly_bet,
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'outcome': outcome,
                    'edge_no': edge_no
                })
    
    if len(results) == 0:
        return {'trades': 0, 'total_invested': 0, 'net_pnl': 0, 'roi_net': 0, 'win_rate': 0}
    
    results_df = pd.DataFrame(results)
    return {
        'trades': len(results_df),
        'total_invested': results_df['bet'].sum(),
        'gross_pnl': results_df['pnl_gross'].sum(),
        'net_pnl': results_df['pnl_net'].sum(),
        'roi_net': (results_df['pnl_net'].sum() / results_df['bet'].sum()) * 100,
        'win_rate': (results_df['outcome'] == 0).mean()  # NO wins when outcome=0
    }

# Negative edge events (model pessimistic)
negative_edge = df[df['edge_lr'] < -0.10]
counter_result = calculate_no_bet_pnl(negative_edge, kelly_fraction=0.25)

strategies.append({
    'Strategy': 'Counter-Trade (bet NO on negative edge)',
    **counter_result
})

print(f"   Counter-trade: {counter_result['trades']} trades, ROI={counter_result['roi_net']:+.2f}%")

# =============================================================================
# 5. COMBINATION STRATEGY (YES on positive, NO on negative)
# =============================================================================

print(f"\n🎯 Testing combination strategy...")

# Positive edge: bet YES
positive_yes = calculate_kelly_pnl(df[df['edge_lr'] >= 0.10], kelly_fraction=0.25)

# Negative edge: bet NO
negative_no = calculate_no_bet_pnl(df[df['edge_lr'] <= -0.10], kelly_fraction=0.25)

# Combine
combo_result = {
    'Strategy': 'Combination (YES if edge>10%, NO if edge<-10%)',
    'trades': positive_yes['trades'] + negative_no['trades'],
    'total_invested': positive_yes['total_invested'] + negative_no['total_invested'],
    'gross_pnl': positive_yes['gross_pnl'] + negative_no['gross_pnl'],
    'net_pnl': positive_yes['net_pnl'] + negative_no['net_pnl'],
    'roi_net': ((positive_yes['net_pnl'] + negative_no['net_pnl']) / 
                (positive_yes['total_invested'] + negative_no['total_invested']) * 100) 
                if (positive_yes['total_invested'] + negative_no['total_invested']) > 0 else 0,
    'win_rate': (positive_yes['trades'] * positive_yes['win_rate'] + 
                 negative_no['trades'] * negative_no['win_rate']) / 
                (positive_yes['trades'] + negative_no['trades']) 
                if (positive_yes['trades'] + negative_no['trades']) > 0 else 0
}

strategies.append(combo_result)

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

print(f"\n💾 Saving strategy comparison...")

strat_df = pd.DataFrame(strategies)
strat_df = strat_df.sort_values('roi_net', ascending=False)
strat_df.to_csv('tables/strategy_comparison_full.csv', index=False)
print(f"   ✓ tables/strategy_comparison_full.csv")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

print(f"\n🎨 Creating visualizations...")

# Figure 1: ROI Comparison
fig, ax = plt.subplots(figsize=(12, 8))

strategies_plot = strat_df.sort_values('roi_net', ascending=True)
colors = ['#06A77D' if x > 0 else '#D90429' for x in strategies_plot['roi_net']]

bars = ax.barh(range(len(strategies_plot)), strategies_plot['roi_net'],
              color=colors, alpha=0.8, edgecolor='white', linewidth=2)

for i, (bar, val, trades) in enumerate(zip(bars, strategies_plot['roi_net'], strategies_plot['trades'])):
    label_text = f'{val:+.1f}% (n={int(trades)})'
    ax.text(val, i, f'  {label_text}', va='center', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_yticks(range(len(strategies_plot)))
ax.set_yticklabels(strategies_plot['Strategy'], fontsize=10)
ax.set_xlabel('ROI (Net, After 2% Fees)', fontsize=13, fontweight='bold')
ax.set_title('Strategy ROI Comparison (Kelly 25%)\nPolymarket Earnings Backtest', 
            fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/strategy_roi_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/strategy_roi_comparison.png")
plt.close()

# Figure 2: Kelly Fraction Optimization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROI vs Kelly fraction
ax1.plot(kelly_df['kelly_fraction'] * 100, kelly_df['roi_net'], 
        'o-', linewidth=2.5, markersize=10, color='#2E86AB')

for x, y in zip(kelly_df['kelly_fraction'] * 100, kelly_df['roi_net']):
    ax1.text(x, y, f'{y:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_xlabel('Kelly Fraction (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('ROI (Net, %)', fontsize=12, fontweight='bold')
ax1.set_title('ROI vs Kelly Fraction', fontsize=13, fontweight='bold', pad=15)
ax1.grid(alpha=0.3)

# Total invested vs Kelly
ax2.plot(kelly_df['kelly_fraction'] * 100, kelly_df['total_invested'],
        'o-', linewidth=2.5, markersize=10, color='#A23B72')

for x, y in zip(kelly_df['kelly_fraction'] * 100, kelly_df['total_invested']):
    ax2.text(x, y, f'${y:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Kelly Fraction (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Capital Required ($)', fontsize=12, fontweight='bold')
ax2.set_title('Capital Required vs Kelly Fraction', fontsize=13, fontweight='bold', pad=15)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/kelly_optimization.png', dpi=300, bbox_inches='tight')
print(f"   ✓ figures/kelly_optimization.png")
plt.close()

# =============================================================================
# 8. GENERATE REPORT
# =============================================================================

print(f"\n📄 Generating optimization report...")

with open('strategy_optimization_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("STRATEGY OPTIMIZATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("KELLY FRACTION COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(kelly_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("STRATEGY COMPARISON (Kelly 25%)\n")
    f.write("-" * 80 + "\n")
    f.write(strat_df.to_string(index=False))
    f.write("\n\n")
    
    # Best strategy
    best = strat_df.iloc[0]
    f.write("OPTIMAL STRATEGY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Strategy: {best['Strategy']}\n")
    f.write(f"ROI (Net): {best['roi_net']:+.2f}%\n")
    f.write(f"Total P&L: ${best['net_pnl']:+,.2f}\n")
    f.write(f"Capital Required: ${best['total_invested']:,.2f}\n")
    f.write(f"Number of Trades: {int(best['trades'])}\n")
    f.write(f"Win Rate: {best['win_rate']:.2%}\n\n")
    
    # Best Kelly
    best_kelly = kelly_df.loc[kelly_df['roi_net'].idxmax()]
    f.write("OPTIMAL KELLY FRACTION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Kelly Fraction: {best_kelly['kelly_fraction']:.0%}\n")
    f.write(f"ROI: {best_kelly['roi_net']:+.2f}%\n")
    f.write(f"Capital Required: ${best_kelly['total_invested']:,.2f}\n\n")
    
    f.write("RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    
    if best['roi_net'] > 10:
        f.write(f"✅ STRONG POSITIVE EV DETECTED ({best['roi_net']:+.1f}% ROI)\n\n")
        f.write(f"Recommended approach:\n")
        f.write(f"1. Use {best['Strategy']}\n")
        f.write(f"2. Kelly fraction: {best_kelly['kelly_fraction']:.0%}\n")
        f.write(f"3. Expected return: {best['roi_net']:+.1f}% per cycle\n")
        f.write(f"4. Capital allocation: ${best['total_invested']:,.0f}\n")
    elif best['roi_net'] > 0:
        f.write(f"✓ MODERATE POSITIVE EV ({best['roi_net']:+.1f}% ROI)\n\n")
        f.write(f"Consider conservative Kelly (10-25%) and larger sample size.\n")
    else:
        f.write(f"⚠️  NEGATIVE/NEUTRAL EV ({best['roi_net']:+.1f}% ROI)\n\n")
        f.write(f"Current data does not support profitable trading.\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF OPTIMIZATION REPORT\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ strategy_optimization_report.txt")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "=" * 80)
print(f"✅ STRATEGY OPTIMIZATION COMPLETE")
print(f"=" * 80)

print(f"\nKelly Fraction Results:")
for _, row in kelly_df.iterrows():
    print(f"   {row['kelly_fraction']:.0%}: ROI={row['roi_net']:+6.2f}%, Capital=${row['total_invested']:>7,.0f}")

print(f"\nTop 3 Strategies:")
for i, (_, row) in enumerate(strat_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['Strategy']:50s} ROI={row['roi_net']:+6.2f}% (n={int(row['trades'])})")

best = strat_df.iloc[0]
print(f"\n🏆 BEST STRATEGY:")
print(f"   • {best['Strategy']}")
print(f"   • ROI: {best['roi_net']:+.2f}%")
print(f"   • P&L: ${best['net_pnl']:+,.2f}")
print(f"   • Capital: ${best['total_invested']:,.2f}")
print(f"   • Trades: {int(best['trades'])}")

print(f"\n" + "=" * 80)

