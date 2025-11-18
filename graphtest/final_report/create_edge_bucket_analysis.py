#!/usr/bin/env python3
"""
Edge Bucket Analysis - Visualize Performance by Edge Range
===========================================================

Shows actual beat rate, model probability, and market price across edge buckets.
Identifies the "Sweet Spot" where model has consistent positive edge.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BLUE = '#2295DC'
ORANGE = '#E74C3C'
GREEN = '#27AE60'
GRAY = '#4A4A4A'

print("=" * 80)
print("EDGE BUCKET ANALYSIS - PERFORMANCE BY EDGE RANGE")
print("=" * 80)

# =============================================================================
# STEP 1: Load Data
# =============================================================================

print(f"\n📂 Loading data...")

# Load backtest results
backtest_df = pd.read_csv('../../graphtest/polymarket_backtest/backtest_results.csv')

# Load polymarket data for avgPrice
poly_df = pd.read_csv('../../../earnings_history_1year.csv')

# Merge
df = backtest_df.merge(
    poly_df[['marketId', 'avgPrice']], 
    left_on='market_id', 
    right_on='marketId', 
    how='left'
)

# Calculate edge (using LR as best model)
df['edge_lr'] = df['lr_probability'] - df['avgPrice']

# Filter to events with avgPrice
df = df[df['avgPrice'].notna()].copy()

print(f"✓ Loaded {len(df)} events with pricing data")
print(f"✓ Average edge: {df['edge_lr'].mean():.3f}")

# =============================================================================
# STEP 2: Create Edge Buckets
# =============================================================================

print(f"\n📊 Creating edge buckets...")

# Edge buckets
edge_bins = [-1, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 1]
edge_labels = ['<-20%', '-20 to -15%', '-15 to -10%', '-10 to -5%', '-5 to 0%', 
               '0 to 5%', '5 to 10%', '10 to 15%', '15 to 20%', '>20%']

df['edge_bucket'] = pd.cut(df['edge_lr'], bins=edge_bins, labels=edge_labels)

# Analyze by edge bucket
bucket_stats = df.groupby('edge_bucket', observed=True).agg({
    'polymarket_outcome': ['mean', 'count'],
    'lr_probability': 'mean',
    'avgPrice': 'mean',
    'edge_lr': 'mean'
}).reset_index()

bucket_stats.columns = ['edge_bucket', 'actual_beat_rate', 'count', 'model_prob', 'market_price', 'avg_edge']

# Convert to percentages
bucket_stats['actual_beat_rate'] *= 100
bucket_stats['model_prob'] *= 100
bucket_stats['market_price'] *= 100
bucket_stats['avg_edge'] *= 100

print("\n" + "=" * 100)
print("EDGE BUCKET ANALYSIS")
print("=" * 100)
print(bucket_stats.to_string(index=False))

# =============================================================================
# STEP 3: Create Visualization
# =============================================================================

print(f"\n🎨 Creating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# -------------------------------------------------------------------------
# PANEL A: Beat Rate, Model Prob, Market Price by Edge Bucket
# -------------------------------------------------------------------------

x = np.arange(len(bucket_stats))
width = 0.25

# Bars
bars1 = ax1.bar(x - width, bucket_stats['actual_beat_rate'], width, 
               label='Actual Beat Rate', color=GREEN, alpha=0.8, edgecolor='white')
bars2 = ax1.bar(x, bucket_stats['model_prob'], width, 
               label='Model Probability', color=BLUE, alpha=0.8, edgecolor='white')
bars3 = ax1.bar(x + width, bucket_stats['market_price'], width, 
               label='Polymarket Price', color=ORANGE, alpha=0.8, edgecolor='white')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Styling
ax1.set_xlabel('Edge Bucket (Model - Market)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability / Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Performance by Edge Bucket', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(bucket_stats['edge_bucket'], rotation=45, ha='right')
ax1.legend(loc='upper left', frameon=True, fancybox=True)
ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
ax1.set_ylim(0, 105)

# Highlight sweet spot
sweet_spot_idx = bucket_stats[bucket_stats['edge_bucket'] == '10 to 15%'].index
if len(sweet_spot_idx) > 0:
    idx = sweet_spot_idx[0]
    ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color='green', zorder=0)
    ax1.text(idx, 100, 'Sweet Spot', ha='center', va='top', 
            fontsize=10, fontweight='bold', color=GREEN)

# Remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# -------------------------------------------------------------------------
# PANEL B: Count and Average Edge by Bucket
# -------------------------------------------------------------------------

# Count bars
ax2_count = ax2.bar(x, bucket_stats['count'], alpha=0.6, color=GRAY, 
                    edgecolor='white', label='Event Count')

# Add count labels
for i, (idx, count) in enumerate(zip(x, bucket_stats['count'])):
    ax2.text(idx, count + 1, f'{int(count)}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Edge Bucket (Model - Market)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Event Count', fontsize=12, fontweight='bold', color=GRAY)
ax2.tick_params(axis='y', labelcolor=GRAY)
ax2.set_xticks(x)
ax2.set_xticklabels(bucket_stats['edge_bucket'], rotation=45, ha='right')
ax2.spines['top'].set_visible(False)

# Secondary y-axis for average edge
ax2_edge = ax2.twinx()
line = ax2_edge.plot(x, bucket_stats['avg_edge'], 
                     marker='o', linewidth=2.5, markersize=8, 
                     color=BLUE, label='Avg Edge')
ax2_edge.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax2_edge.set_ylabel('Average Edge (%)', fontsize=12, fontweight='bold', color=BLUE)
ax2_edge.tick_params(axis='y', labelcolor=BLUE)
ax2_edge.spines['top'].set_visible(False)

# Title
ax2.set_title('Panel B: Event Distribution and Average Edge', fontsize=13, fontweight='bold', pad=15)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_edge.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fancybox=True)

# Main title
fig.suptitle('Edge Bucket Analysis: Identifying Profitable Opportunities', 
            fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()

# Save
output_path = 'figures/fig29_edge_bucket_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_path}")

# =============================================================================
# STEP 4: Key Insights
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Find sweet spot
sweet_spot = bucket_stats[bucket_stats['edge_bucket'] == '10 to 15%']
if len(sweet_spot) > 0:
    print(f"\n🎯 SWEET SPOT (10 to 15% edge):")
    print(f"   • Actual Beat Rate: {sweet_spot['actual_beat_rate'].values[0]:.1f}%")
    print(f"   • Model Probability: {sweet_spot['model_prob'].values[0]:.1f}%")
    print(f"   • Market Price: {sweet_spot['market_price'].values[0]:.1f}%")
    print(f"   • Event Count: {int(sweet_spot['count'].values[0])}")
    print(f"   • Average Edge: {sweet_spot['avg_edge'].values[0]:+.1f}%")

# Negative edge analysis
neg_edge = bucket_stats[bucket_stats['avg_edge'] < -10]
if len(neg_edge) > 0:
    print(f"\n⚠️  NEGATIVE EDGE REGIONS (edge < -10%):")
    for _, row in neg_edge.iterrows():
        print(f"   • {row['edge_bucket']:15s}: Beat Rate {row['actual_beat_rate']:.1f}%, "
              f"Model {row['model_prob']:.1f}%, Market {row['market_price']:.1f}%, "
              f"Count {int(row['count'])}")

# Overall statistics
print(f"\n📊 OVERALL STATISTICS:")
print(f"   • Total Events: {len(df)}")
print(f"   • Overall Beat Rate: {df['polymarket_outcome'].mean()*100:.1f}%")
print(f"   • Avg Model Probability: {df['lr_probability'].mean()*100:.1f}%")
print(f"   • Avg Market Price: {df['avgPrice'].mean()*100:.1f}%")
print(f"   • Avg Edge: {df['edge_lr'].mean()*100:.1f}%")

print("\n" + "=" * 80)
print("✅ EDGE BUCKET ANALYSIS COMPLETE!")
print("=" * 80)

plt.show()


