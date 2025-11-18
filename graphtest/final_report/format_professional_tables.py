#!/usr/bin/env python3
"""
Professional Table Formatting
==============================

Creates beautifully formatted tables for the investment report:
- Clean borders
- Alternating row shading
- Right-aligned numbers
- Comma separators
- Source citations
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("FORMATTING PROFESSIONAL TABLES")
print("=" * 80)

# Helper function for professional table formatting
def format_table_professional(df, title, source, filename):
    """Format dataframe as professional table and save"""
    
    # Create formatted string
    output = []
    output.append("=" * 100)
    output.append(f" {title}")
    output.append("=" * 100)
    output.append("")
    
    # Header
    col_widths = [max(len(str(col)), df[col].astype(str).str.len().max() + 2) for col in df.columns]
    
    header_line = " | ".join([str(col).ljust(w) for col, w in zip(df.columns, col_widths)])
    output.append(header_line)
    output.append("-" * len(header_line))
    
    # Rows with alternating shading indicator
    for i, (idx, row) in enumerate(df.iterrows()):
        shade_marker = "█" if i % 2 == 0 else " "
        row_strs = []
        for col, width in zip(df.columns, col_widths):
            val = row[col]
            # Format numbers
            if isinstance(val, (int, np.integer)):
                formatted = f"{val:,}".rjust(width)
            elif isinstance(val, (float, np.floating)):
                if abs(val) < 1:
                    formatted = f"{val:.4f}".rjust(width)
                elif abs(val) < 100:
                    formatted = f"{val:.2f}".rjust(width)
                else:
                    formatted = f"{val:,.2f}".rjust(width)
            else:
                formatted = str(val).ljust(width)
            row_strs.append(formatted)
        
        output.append(f"{shade_marker}| " + " | ".join(row_strs) + " |")
    
    output.append("=" * 100)
    output.append(f"Source: {source}")
    output.append(f"Table: {filename}")
    output.append("")
    
    # Save
    with open(f'tables/{filename}', 'w') as f:
        f.write('\n'.join(output))
    
    print(f"   ✓ {filename}")

# =============================================================================
# TABLE 1: GLOBAL MODEL PERFORMANCE
# =============================================================================

print("\n📋 Table 1: Global Model Performance...")

model_perf = pd.read_csv('../global_model/results/model_performance.csv')

# Format percentages
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    model_perf[col] = (model_perf[col] * 100).round(2)

# Rename for clarity
model_perf = model_perf[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

format_table_professional(
    model_perf,
    "Global ML Model Performance (Out-of-Sample Test Set)",
    "N=9,256 observations, 2020-2025 | Analysis: November 2025",
    "table1_global_model_performance.txt"
)

# =============================================================================
# TABLE 2: YEARLY BEAT RATES
# =============================================================================

print("\n📋 Table 2: Yearly Beat Rates...")

yearly = pd.read_csv('../excel_data/yearly_beat_miss.csv')

# Rename columns
yearly_formatted = yearly.copy()
yearly_formatted.columns = ['Year', 'Total Reports', 'Beats', 'Misses', 'Beat %', 'Miss %']

format_table_professional(
    yearly_formatted,
    "S&P 500 Earnings Beat/Miss Statistics by Year (2017-2025)",
    "Alpha Vantage API | 14,601 quarterly reports | 468 companies",
    "table2_yearly_beat_rates.txt"
)

# =============================================================================
# TABLE 3: CONSENSUS SPREAD ANALYSIS
# =============================================================================

print("\n📋 Table 3: Consensus Spread...")

spread = pd.read_csv('../research_paper/tables/beat_rate_by_spread.csv')

# Format
spread_formatted = pd.DataFrame({
    'Quintile': spread['spread_quintile'],
    'Avg Spread %': spread['estimate_spread_pct_mean'].round(2),
    'Beat Rate %': (spread['eps_beat_mean'] * 100).round(2),
    'Count': spread['eps_beat_count'].astype(int),
    'Std Dev': spread['eps_beat_std'].round(4)
})

format_table_professional(
    spread_formatted,
    "Beat Rate by Analyst Consensus Spread Quintile",
    "N=13,856 observations | Chi-square: χ²(4)=186.80, p<0.0001",
    "table3_consensus_spread.txt"
)

# =============================================================================
# TABLE 4: STRATEGY COMPARISON
# =============================================================================

print("\n📋 Table 4: Strategy Comparison...")

strategies = pd.read_csv('../polymarket_backtest/tables/strategy_comparison_full.csv')

# Format
strat_formatted = pd.DataFrame({
    'Strategy': strategies['Strategy'],
    'Trades': strategies['trades'].astype(int),
    'Capital $': strategies['total_invested'].round(2),
    'Net P&L $': strategies['net_pnl'].round(2),
    'ROI %': strategies['roi_net'].round(2),
    'Win Rate %': (strategies['win_rate'] * 100).round(2)
})

# Sort by ROI
strat_formatted = strat_formatted.sort_values('ROI %', ascending=False)

format_table_professional(
    strat_formatted,
    "Trading Strategy Performance Comparison (Kelly 25%, After Fees)",
    "Polymarket backtest | N=248 events | Q3 2025",
    "table4_strategy_comparison.txt"
)

# =============================================================================
# TABLE 5: EDGE BUCKET ANALYSIS
# =============================================================================

print("\n📋 Table 5: Edge Bucket Analysis...")

edge = pd.read_csv('../polymarket_backtest/tables/edge_bucket_analysis.csv')

edge_formatted = pd.DataFrame({
    'Edge Range': edge['edge_bucket_lr'],
    'Actual Beat %': (edge['polymarket_outcome_mean'] * 100).round(2),
    'Model Accuracy %': (edge['lr_correct_mean'] * 100).round(2),
    'Count': edge['polymarket_outcome_count'].astype(int),
    'Avg Model Prob': edge['lr_probability_mean'].round(4),
    'Avg PM Price': edge['avgPrice_mean'].round(4)
})

format_table_professional(
    edge_formatted,
    "Performance by Edge Bucket (p_model - p_market)",
    "Logistic Regression | N=248 Polymarket events",
    "table5_edge_bucket_analysis.txt"
)

# =============================================================================
# TABLE 6: KELLY FRACTION COMPARISON
# =============================================================================

print("\n📋 Table 6: Kelly Fraction...")

kelly = pd.read_csv('../polymarket_backtest/tables/kelly_fraction_comparison.csv')

kelly_formatted = pd.DataFrame({
    'Kelly %': (kelly['kelly_fraction'] * 100).astype(int),
    'Trades': kelly['trades'].astype(int),
    'Capital $': kelly['total_invested'].round(2),
    'Net P&L $': kelly['net_pnl'].round(2),
    'ROI %': kelly['roi_net'].round(2),
    'Win Rate %': (kelly['win_rate'] * 100).round(2)
})

format_table_professional(
    kelly_formatted,
    "Kelly Fraction Optimization Analysis",
    "N=90 positive edge events | ROI independent of fraction",
    "table6_kelly_fraction.txt"
)

# =============================================================================
# TABLE 7: FEATURE IMPORTANCE TOP 20
# =============================================================================

print("\n📋 Table 7: Feature Importance...")

# Reconstruct from report data
features_data = {
    'Rank': list(range(1, 11)),
    'Feature': ['elo_momentum', 'elo_before', 'elo_decay', 'elo_vol_4q',
               'actual_eps_qoq_growth_lag1', 'price_1m_before', 'price_3m_before',
               'total_revenue_qoq_growth_lag1', 'revenue_estimate_average', 'eps_estimate_average'],
    'Importance %': [36.17, 6.38, 6.16, 5.27, 2.66, 2.43, 2.38, 2.31, 2.18, 2.16],
    'Category': ['Elo', 'Elo', 'Elo', 'Elo', 'Growth', 'Price', 'Price', 'Growth', 'Estimates', 'Estimates']
}

features_df = pd.DataFrame(features_data)

format_table_professional(
    features_df,
    "Top 10 Predictive Features (Random Forest Global Model)",
    "N=14,239 observations | Elo metrics: 53.98% total importance",
    "table7_feature_importance.txt"
)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("✅ PROFESSIONAL TABLES FORMATTED")
print("=" * 80)

print("\nGenerated Tables:")
print("   1. Global Model Performance")
print("   2. Yearly Beat Rates (2017-2025)")
print("   3. Consensus Spread Analysis")
print("   4. Strategy Comparison")
print("   5. Edge Bucket Analysis")
print("   6. Kelly Fraction Comparison")
print("   7. Feature Importance Top 10")

print("\nFormat:")
print("   • Professional borders")
print("   • Alternating row shading (█ marker)")
print("   • Right-aligned numbers")
print("   • Comma separators")
print("   • Source citations")

print("\nLocation: graphtest/final_report/tables/")
print("=" * 80)

