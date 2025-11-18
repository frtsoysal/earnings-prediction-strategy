#!/usr/bin/env python3
"""
Figure 1 - Yearly Beat/Miss Bar Chart
======================================

Clean bar chart showing beat and miss rates by year.
Blue tones, white background, no grid.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Blue tones
BLUE_BEAT = '#2295DC'  # RGB(34, 149, 220) - Beat
BLUE_MISS = '#5AADE8'  # Lighter blue - Miss
GRAY_TEXT = '#4A4A4A'

# Clean style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5
})

# Load data
yearly_df = pd.read_csv('../excel_data/yearly_beat_miss.csv')

fig, ax = plt.subplots(figsize=(12, 7))

years = yearly_df['year'].values
beat_pct = yearly_df['beat_pct'].values
miss_pct = yearly_df['miss_pct'].values

x = np.arange(len(years))
width = 0.35

# Beat bars
bars1 = ax.bar(x - width/2, beat_pct, width, 
              color=BLUE_BEAT, edgecolor='white', linewidth=2,
              label='Beat %')

# Miss bars
bars2 = ax.bar(x + width/2, miss_pct, width,
              color=BLUE_MISS, edgecolor='white', linewidth=2,
              label='Miss %')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}%',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}%',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Rate (%)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylim([0, 85])

# Legend
ax.legend(loc='upper left', frameon=False, fontsize=11)

# Clean axes - no grid
ax.grid(False)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRAY_TEXT)
ax.spines['bottom'].set_color(GRAY_TEXT)

plt.tight_layout()
plt.savefig('figures/fig1_yearly_beat_rate_trend.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print('✓ fig1 created - bar chart, blue tones, no grid')
plt.close()

