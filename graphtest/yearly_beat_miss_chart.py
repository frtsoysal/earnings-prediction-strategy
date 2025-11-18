#!/usr/bin/env python3
"""
Yearly Beat/Miss Analysis Chart
================================

2017+ yılları için beat/miss oranlarını profesyonel bar chart ile görselleştirir.
Big 4 consulting firma tarzında.

Output:
    excel_data/yearly_beat_miss.csv
    excel_data/yearly_beat_miss_chart.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read data
df = pd.read_csv('excel_data/quarterly_beat_miss.csv')

print("=" * 80)
print("YEARLY BEAT/MISS ANALYSIS - PROFESSIONAL CHART")
print("=" * 80)

# Parse date and extract year
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Filter: 2017+ and Historical only
df_filtered = df[(df['year'] >= 2017) & (df['status'] == 'Historical')].copy()

print(f"\n📊 Veri filtrelendi: {df_filtered['year'].min()}-{df_filtered['year'].max()}")

# Group by year
yearly = df_filtered.groupby('year').agg({
    'total_companies': 'sum',
    'beat_count': 'sum',
    'miss_count': 'sum'
}).reset_index()

# Calculate percentages
yearly['beat_pct'] = (yearly['beat_count'] / yearly['total_companies'] * 100).round(1)
yearly['miss_pct'] = (yearly['miss_count'] / yearly['total_companies'] * 100).round(1)

print(f"\n📈 Yıllık özet:")
print(yearly[['year', 'total_companies', 'beat_count', 'miss_count', 'beat_pct', 'miss_pct']].to_string(index=False))

# Save to CSV
yearly.to_csv('excel_data/yearly_beat_miss.csv', index=False)
print(f"\n💾 CSV kaydedildi: excel_data/yearly_beat_miss.csv")

# =============================================================================
# PROFESSIONAL CHART - BIG 4 STYLE
# =============================================================================

print(f"\n🎨 Profesyonel grafik oluşturuluyor...")

# Big 4 Corporate Colors
BEAT_COLOR = '#00A3E0'    # Deloitte Blue
MISS_COLOR = '#ED8B00'    # EY Yellow/Orange
GRID_COLOR = '#E5E5E5'
TEXT_COLOR = '#333333'
LABEL_COLOR = '#666666'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

# Data
years = yearly['year'].astype(str)
beat_pct = yearly['beat_pct']
miss_pct = yearly['miss_pct']

# Bar positions
x = np.arange(len(years))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, beat_pct, width, label='Beat %', 
               color=BEAT_COLOR, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, miss_pct, width, label='Miss %',
               color=MISS_COLOR, edgecolor='white', linewidth=1.5)

# Add value labels on bars
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=TEXT_COLOR)

add_labels(bars1, beat_pct)
add_labels(bars2, miss_pct)

# Styling
ax.set_xlabel('Year', fontsize=13, fontweight='bold', color=TEXT_COLOR)
ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold', color=TEXT_COLOR)
ax.set_title('S&P 500 Earnings Beat vs Miss Rate by Year (2017-2025)',
            fontsize=16, fontweight='bold', color=TEXT_COLOR, pad=20)

# X-axis
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11, color=TEXT_COLOR)

# Y-axis
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_tick_params(labelsize=11, colors=LABEL_COLOR)

# Grid
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color=GRID_COLOR)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID_COLOR)
ax.spines['bottom'].set_color(GRID_COLOR)

# Legend
legend = ax.legend(loc='upper left', frameon=True, fontsize=12,
                  edgecolor=GRID_COLOR, facecolor='white', framealpha=0.95)
legend.get_frame().set_linewidth(1.5)

# Add subtle background
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# Add footer note
footer_text = f"Data: {yearly['total_companies'].sum():,} quarterly earnings reports | Historical data through {yearly['year'].max()}"
fig.text(0.5, 0.02, footer_text,
        ha='center', fontsize=9, color=LABEL_COLOR, style='italic')

# Add average line
avg_beat = yearly['beat_pct'].mean()
ax.axhline(y=avg_beat, color=BEAT_COLOR, linestyle='--', linewidth=1.5, alpha=0.4)
ax.text(len(years) - 0.5, avg_beat + 2, f'Avg Beat: {avg_beat:.1f}%',
       fontsize=9, color=BEAT_COLOR, fontstyle='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save
output_file = 'excel_data/yearly_beat_miss_chart.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Grafik kaydedildi: {output_file}")

plt.close()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Özet İstatistikler (2017-{yearly['year'].max()}):")
print("=" * 80)
print(f"   • Toplam rapor: {yearly['total_companies'].sum():,}")
print(f"   • Toplam beat: {yearly['beat_count'].sum():,}")
print(f"   • Toplam miss: {yearly['miss_count'].sum():,}")
print(f"   • Ortalama beat rate: {yearly['beat_pct'].mean():.1f}%")
print(f"   • En yüksek beat rate: {yearly['beat_pct'].max():.1f}% ({yearly.loc[yearly['beat_pct'].idxmax(), 'year']})")
print(f"   • En düşük beat rate: {yearly['beat_pct'].min():.1f}% ({yearly.loc[yearly['beat_pct'].idxmin(), 'year']})")
print("=" * 80)
print(f"\n✅ TAMAMLANDI!")
print(f"   CSV: excel_data/yearly_beat_miss.csv")
print(f"   Chart: excel_data/yearly_beat_miss_chart.png")
print("=" * 80)

