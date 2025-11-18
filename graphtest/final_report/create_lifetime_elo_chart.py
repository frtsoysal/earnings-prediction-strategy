#!/usr/bin/env python3
"""
LIFETIME ELO TRAJECTORIES - Chess Masters Style
================================================

Multi-line chart showing Elo rating evolution over time for 10 selected stocks.
Similar to the famous chess grandmasters lifetime rating chart.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BLUE = '#2295DC'
GRAY = '#4A4A4A'

# =============================================================================
# STEP 1: Load and combine data from multiple tickers
# =============================================================================

print("=" * 80)
print("LIFETIME ELO TRAJECTORIES - CHESS MASTERS STYLE")
print("=" * 80)

# Select tickers with good Elo history (diverse sectors)
# Note: Some tickers may have insufficient data and will be skipped
tickers = [
    "MSFT",   # Microsoft - consistent performer
    "META",   # Meta - social media
    "TSLA",   # Tesla - volatile innovator
    "JPM",    # JP Morgan - finance leader
    "V",      # Visa - payments
    "AMD",    # AMD - semiconductors
    "ADBE",   # Adobe - software
    "NFLX",   # Netflix - streaming
    "CRM",    # Salesforce - cloud
    "ORCL",   # Oracle - enterprise
    "COST",   # Costco - retail
    "HD",     # Home Depot - retail
    "UNH",    # UnitedHealth - healthcare
    "BA",     # Boeing - aerospace
    "CAT",    # Caterpillar - industrial
    "DIS",    # Disney - entertainment
    "NKE",    # Nike - consumer
    "INTC",   # Intel - semiconductors
    "QCOM",   # Qualcomm - semiconductors
    "SBUX",   # Starbucks - consumer
]

data_dir = Path("../../data/raw")
all_data = []

print(f"\n📂 Loading data for {len(tickers)} tickers...")

for ticker in tickers:
    csv_path = data_dir / f"{ticker}_earnings_with_q4.csv"
    
    if not csv_path.exists():
        print(f"   ⚠️  {ticker}: File not found, skipping")
        continue
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if we have necessary columns
        if 'elo_after' not in df.columns or 'date' not in df.columns:
            print(f"   ⚠️  {ticker}: Missing required columns, skipping")
            continue
        
        # Select relevant columns
        df = df[['date', 'elo_after']].copy()
        df['ticker'] = ticker
        df.columns = ['date', 'elo', 'ticker']
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove invalid Elo values
        df = df[df['elo'].notna() & (df['elo'] > 0)]
        
        if len(df) >= 10:  # Need at least 10 quarters for smooth trajectory
            all_data.append(df)
            print(f"   ✓ {ticker}: {len(df)} data points loaded")
        else:
            print(f"   ⚠️  {ticker}: Insufficient data ({len(df)} points), skipping")
    
    except Exception as e:
        print(f"   ✗ {ticker}: Error - {e}")

# Combine all data
if not all_data:
    print("\n❌ No data loaded! Check file paths.")
    exit(1)

df_combined = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined dataset: {len(df_combined)} total records")
print(f"✓ Date range: {df_combined['date'].min().date()} to {df_combined['date'].max().date()}")

# Update tickers list to only include those we loaded (limit to best 10)
loaded_tickers = df_combined['ticker'].unique().tolist()
if len(loaded_tickers) > 10:
    loaded_tickers = loaded_tickers[:10]
    df_combined = df_combined[df_combined['ticker'].isin(loaded_tickers)]
    print(f"✓ Successfully loaded tickers: {loaded_tickers} (limited to 10)")
else:
    print(f"✓ Successfully loaded tickers: {loaded_tickers}")

# =============================================================================
# STEP 2: Resample and smooth data (monthly averages)
# =============================================================================

print(f"\n📊 Resampling to monthly frequency for smoothness...")

def resample_smooth(group, freq="M"):
    """Resample to monthly averages and interpolate gaps"""
    ticker_name = group['ticker'].iloc[0]  # Preserve ticker
    g = group.set_index("date")[['elo']].resample(freq).mean()
    g = g.interpolate(method="time", limit=3)  # Fill small gaps
    g = g.reset_index()
    g['ticker'] = ticker_name  # Add ticker back
    return g

df_resampled = (df_combined
                .sort_values(['ticker', 'date'])
                .groupby('ticker', group_keys=False)
                .apply(lambda x: resample_smooth(x)))

print(f"✓ Resampled to {len(df_resampled)} monthly data points")

# =============================================================================
# STEP 3: Savitzky-Golay smoothing for chess-style curves
# =============================================================================

def smooth_trajectory(elo_values, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter for smooth trajectories
    
    Args:
        elo_values: Elo ratings
        window_length: smoothing window (must be odd)
        polyorder: polynomial order
    
    Returns:
        Smoothed Elo values
    """
    y = elo_values.to_numpy()
    
    # Ensure window length is appropriate
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window_length < 3:
            return y  # Too short, return as is
    
    # Apply Savitzky-Golay filter
    smoothed = savgol_filter(y, window_length, polyorder, mode='nearest')
    
    return smoothed

print(f"\n🎯 Applying Savitzky-Golay smoothing for smooth curves...")

# =============================================================================
# STEP 4: Create the chart (Chess Masters style)
# =============================================================================

print(f"\n🎨 Creating lifetime Elo chart...")

fig, ax = plt.subplots(figsize=(16, 10))

# Color palette - using diverse colors for 10 lines
colors = [
    '#2E86AB',  # Blue
    '#A23B72',  # Purple
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#6A994E',  # Green
    '#BC4B51',  # Dark red
    '#8CB369',  # Light green
    '#5B8E7D',  # Teal
    '#F4A259',  # Light orange
    '#BC96E6',  # Lavender
]

end_labels = []  # For direct labeling at the end

for i, ticker in enumerate(sorted(loaded_tickers)):
    group = df_resampled[df_resampled['ticker'] == ticker].copy()
    
    if group.empty or len(group) < 5:
        continue
    
    # Sort by date
    group = group.sort_values('date')
    
    # Apply Savitzky-Golay smoothing
    y_smooth = smooth_trajectory(group['elo'], window_length=11, polyorder=3)
    
    # Plot
    color = colors[i % len(colors)]
    ax.plot(group['date'], y_smooth, 
           linewidth=2.5, 
           alpha=0.9, 
           color=color,
           label=ticker)
    
    # Store last point for labeling
    end_labels.append({
        'x': group['date'].iloc[-1],
        'y': y_smooth[-1],
        'ticker': ticker,
        'color': color
    })

# =============================================================================
# STEP 5: Styling (Chess chart aesthetic)
# =============================================================================

# Title and labels
ax.set_title('LIFETIME ELO RATINGS OF SELECTED STOCKS', 
            fontsize=18, 
            fontweight='bold',
            pad=20,
            color=GRAY)

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Elo Rating', fontsize=13, fontweight='bold')

# Grid (subtle, like chess chart)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, color=GRAY)
ax.set_axisbelow(True)

# Remove top and right spines (cleaner look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Direct labels at the end (chess style)
for label in end_labels:
    ax.annotate(label['ticker'], 
               xy=(label['x'], label['y']),
               xytext=(8, 0),
               textcoords='offset points',
               va='center',
               fontsize=10,
               fontweight='bold',
               color=label['color'])

# Y-axis formatting
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

# Adjust layout
plt.tight_layout()

# Save
output_svg = 'figures/fig26_lifetime_elo_trajectories.svg'
output_png = 'figures/fig26_lifetime_elo_trajectories.png'

plt.savefig(output_svg, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n✓ Saved: {output_svg}")
print(f"✓ Saved: {output_png}")

# =============================================================================
# STEP 6: Summary statistics
# =============================================================================

print(f"\n" + "=" * 80)
print("TRAJECTORY SUMMARY")
print("=" * 80)

summary_data = []

for ticker in sorted(loaded_tickers):
    group = df_combined[df_combined['ticker'] == ticker]
    
    if group.empty:
        continue
    
    # Calculate statistics
    first_elo = group.sort_values('date')['elo'].iloc[0]
    last_elo = group.sort_values('date')['elo'].iloc[-1]
    max_elo = group['elo'].max()
    min_elo = group['elo'].min()
    change = last_elo - first_elo
    change_pct = (change / first_elo) * 100
    
    summary_data.append({
        'Ticker': ticker,
        'Start': f'{first_elo:.0f}',
        'End': f'{last_elo:.0f}',
        'Peak': f'{max_elo:.0f}',
        'Low': f'{min_elo:.0f}',
        'Change': f'{change:+.0f}',
        'Change %': f'{change_pct:+.1f}%'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("✅ LIFETIME ELO CHART COMPLETE!")
print("=" * 80)

plt.show()

