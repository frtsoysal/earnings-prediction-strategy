#!/usr/bin/env python3
"""
Elo Rating Distribution - Corrected Version
============================================

Create proper Elo rating distribution across all companies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BLUE = '#2295DC'
LIGHT_BLUE = '#89CFF0'
GRAY = '#4A4A4A'

print("=" * 80)
print("ELO RATING DISTRIBUTION - ALL COMPANIES")
print("=" * 80)

# =============================================================================
# STEP 1: Load all Elo data from CSV files
# =============================================================================

data_dir = Path("../../data/raw")
all_elo_data = []

print(f"\n📂 Loading Elo data from all CSV files...")

csv_files = list(data_dir.glob("*_earnings_with_q4.csv"))
loaded_count = 0
skipped_count = 0

for csv_path in csv_files:
    ticker = csv_path.stem.replace('_earnings_with_q4', '')
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if necessary columns exist
        if 'elo_after' not in df.columns or 'eps_beat' not in df.columns:
            skipped_count += 1
            continue
        
        # Extract relevant data
        elo_data = df[['elo_after', 'eps_beat']].copy()
        elo_data['ticker'] = ticker
        
        # Remove invalid values
        elo_data = elo_data[elo_data['elo_after'].notna() & (elo_data['elo_after'] > 0)]
        elo_data = elo_data[elo_data['eps_beat'].notna()]
        
        if len(elo_data) > 0:
            all_elo_data.append(elo_data)
            loaded_count += 1
    
    except Exception as e:
        skipped_count += 1
        continue

if not all_elo_data:
    print("\n❌ No data loaded!")
    exit(1)

# Combine all data
df_all = pd.concat(all_elo_data, ignore_index=True)

print(f"✓ Loaded {loaded_count} companies")
print(f"✓ Skipped {skipped_count} files")
print(f"✓ Total observations: {len(df_all):,}")
print(f"✓ Elo range: {df_all['elo_after'].min():.0f} - {df_all['elo_after'].max():.0f}")
print(f"✓ Mean Elo: {df_all['elo_after'].mean():.0f}")
print(f"✓ Median Elo: {df_all['elo_after'].median():.0f}")

# =============================================================================
# STEP 2: Create the visualization
# =============================================================================

# Filter outliers for better visualization (keep 99th percentile)
elo_99 = df_all['elo_after'].quantile(0.99)
df_filtered = df_all[df_all['elo_after'] <= elo_99].copy()

print(f"\n📊 Filtering for visualization:")
print(f"✓ 99th percentile: {elo_99:.0f}")
print(f"✓ Filtered data: {len(df_filtered):,} observations ({len(df_filtered)/len(df_all)*100:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------------------------------------------------------
# PANEL A: Elo Rating Distribution (Histogram)
# -------------------------------------------------------------------------

# Use filtered data for cleaner visualization
ax1.hist(df_filtered['elo_after'], bins=60, color=BLUE, alpha=0.7, edgecolor='white', linewidth=0.5)

# Add vertical line for neutral rating (1500)
ax1.axvline(x=1500, color='orange', linestyle='--', linewidth=2, label='Neutral (1500)', alpha=0.8)

# Add mean and median lines (from filtered data for visualization)
mean_elo_filtered = df_filtered['elo_after'].mean()
median_elo_filtered = df_filtered['elo_after'].median()
ax1.axvline(x=mean_elo_filtered, color='red', linestyle='--', linewidth=2, 
           label=f'Mean ({mean_elo_filtered:.0f})', alpha=0.8)
ax1.axvline(x=median_elo_filtered, color='green', linestyle=':', linewidth=2, 
           label=f'Median ({median_elo_filtered:.0f})', alpha=0.7)

ax1.set_xlabel('Elo Rating', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Panel A: Elo Rating Distribution', fontsize=12, fontweight='bold', pad=15)
ax1.legend(frameon=True, fancybox=True)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# -------------------------------------------------------------------------
# PANEL B: Beat Rate by Elo Level (Deciles)
# -------------------------------------------------------------------------

# Create Elo deciles
df_all['elo_decile'] = pd.qcut(df_all['elo_after'], q=10, labels=False, duplicates='drop')

# Calculate beat rate by decile
decile_stats = df_all.groupby('elo_decile').agg({
    'eps_beat': 'mean',
    'elo_after': ['mean', 'count']
}).reset_index()

decile_stats.columns = ['decile', 'beat_rate', 'mean_elo', 'count']
decile_stats['beat_rate_pct'] = decile_stats['beat_rate'] * 100

# Normalize decile for x-axis (0 to 1)
decile_stats['decile_norm'] = decile_stats['decile'] / 9.0

# Plot
ax2.plot(decile_stats['decile_norm'], decile_stats['beat_rate_pct'], 
        marker='o', linewidth=2.5, markersize=8, color=BLUE)

ax2.set_xlabel('Elo Decile (Low → High)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Beat Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Panel B: Beat Rate by Elo Level', fontsize=12, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Set y-axis range
y_min = max(0, decile_stats['beat_rate_pct'].min() - 5)
y_max = min(100, decile_stats['beat_rate_pct'].max() + 5)
ax2.set_ylim(y_min, y_max)

# Add annotation
ax2.text(0.98, 0.02, f'Source: N={len(df_all):,} observations, {loaded_count} companies',
        transform=ax2.transAxes, ha='right', va='bottom',
        fontsize=8, style='italic', color=GRAY)

# Main title
fig.suptitle('Elo Rating System Performance', fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()

# Save
output_path = 'figures/fig9_elo_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_path}")

# =============================================================================
# STEP 3: Summary Statistics
# =============================================================================

print("\n" + "=" * 80)
print("ELO DISTRIBUTION SUMMARY")
print("=" * 80)

print("\nDistribution Statistics:")
print(f"  Min:     {df_all['elo_after'].min():.0f}")
print(f"  25th %:  {df_all['elo_after'].quantile(0.25):.0f}")
print(f"  Median:  {df_all['elo_after'].median():.0f}")
print(f"  Mean:    {df_all['elo_after'].mean():.0f}")
print(f"  75th %:  {df_all['elo_after'].quantile(0.75):.0f}")
print(f"  Max:     {df_all['elo_after'].max():.0f}")
print(f"  Std Dev: {df_all['elo_after'].std():.0f}")

print("\nElo Categories:")
categories = [
    ("< 1000 (Very Weak)", (df_all['elo_after'] < 1000).sum()),
    ("1000-1500 (Weak)", ((df_all['elo_after'] >= 1000) & (df_all['elo_after'] < 1500)).sum()),
    ("1500-2000 (Average)", ((df_all['elo_after'] >= 1500) & (df_all['elo_after'] < 2000)).sum()),
    ("2000-2500 (Strong)", ((df_all['elo_after'] >= 2000) & (df_all['elo_after'] < 2500)).sum()),
    ("2500+ (Very Strong)", (df_all['elo_after'] >= 2500).sum()),
]

for cat_name, count in categories:
    pct = (count / len(df_all)) * 100
    print(f"  {cat_name:25s}: {count:5,} ({pct:5.1f}%)")

print("\nBeat Rate by Elo Level:")
print("\nDecile | Mean Elo | Beat Rate | Count")
print("-------|----------|-----------|-------")
for _, row in decile_stats.iterrows():
    print(f"  {int(row['decile']):2d}   | {row['mean_elo']:8.0f} | {row['beat_rate_pct']:6.1f}%  | {int(row['count']):5,}")

print("\n" + "=" * 80)
print("✅ ELO DISTRIBUTION ANALYSIS COMPLETE!")
print("=" * 80)

plt.show()

