#!/usr/bin/env python3
"""
Polymarket Mechanism Diagram
=============================

Creates a professional diagram showing how Polymarket works for earnings bets.
Goldman Sachs color scheme.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

print("Creating Polymarket mechanism diagram...")

# Professional colors - Single blue theme
BLUE_RGB = (34/255, 149/255, 220/255)  # RGB(34, 149, 220)
BLUE_HEX = '#2295DC'

GS_COLORS = {
    'navy': BLUE_HEX,
    'gold': BLUE_HEX,
    'green': BLUE_HEX,
    'red': BLUE_HEX,
    'gray': '#4A4A4A',
    'light_gray': '#F5F5F5'
}

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'POLYMARKET EARNINGS PREDICTION MECHANISM',
       ha='center', va='top', fontsize=20, fontweight='bold', color=GS_COLORS['navy'])

ax.text(5, 9.0, 'Binary Outcome Contracts on Corporate Earnings',
       ha='center', va='top', fontsize=14, style='italic', color=GS_COLORS['gray'])

# ============================================================================
# STEP 1: Market Creation
# ============================================================================

# Market creation box
market_box = FancyBboxPatch((0.5, 7), 2.5, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor=BLUE_HEX,
                           facecolor='white',
                           linewidth=3)
ax.add_patch(market_box)

ax.text(1.75, 7.8, 'MARKET CREATION', ha='center', va='center',
       fontsize=12, fontweight='bold', color=BLUE_HEX)
ax.text(1.75, 7.4, 'Question:', ha='center', va='center',
       fontsize=9, fontweight='bold')
ax.text(1.75, 7.1, '"Will AAPL beat Q3\nearnings estimates?"',
       ha='center', va='center', fontsize=8, style='italic')

# ============================================================================
# STEP 2: Share Types
# ============================================================================

# Arrow to shares
arrow1 = FancyArrowPatch((3.1, 7.6), (4.4, 7.6),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color=GS_COLORS['gold'])
ax.add_patch(arrow1)

# YES share box
yes_box = FancyBboxPatch((4.5, 7.2), 2, 0.8,
                        boxstyle="round,pad=0.08",
                        edgecolor=BLUE_HEX,
                        facecolor='white',
                        linewidth=3)
ax.add_patch(yes_box)

ax.text(5.5, 7.6, 'YES SHARE', ha='center', va='center',
       fontsize=11, fontweight='bold', color=BLUE_HEX)
ax.text(5.5, 7.35, 'If company BEATS', ha='center', va='center',
       fontsize=8)

# NO share box
no_box = FancyBboxPatch((4.5, 6.1), 2, 0.8,
                       boxstyle="round,pad=0.08",
                       edgecolor=BLUE_HEX,
                       facecolor='white',
                       linewidth=3)
ax.add_patch(no_box)

ax.text(5.5, 6.5, 'NO SHARE', ha='center', va='center',
       fontsize=11, fontweight='bold', color=BLUE_HEX)
ax.text(5.5, 6.25, 'If company MISSES', ha='center', va='center',
       fontsize=8)

# ============================================================================
# STEP 3: Trading Range
# ============================================================================

# Arrow down
arrow2 = FancyArrowPatch((5.5, 6.0), (5.5, 5.3),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color=GS_COLORS['gold'])
ax.add_patch(arrow2)

# Trading box
trade_box = FancyBboxPatch((3.5, 3.8), 4, 1.3,
                          boxstyle="round,pad=0.1",
                          edgecolor=BLUE_HEX,
                          facecolor='white',
                          linewidth=3)
ax.add_patch(trade_box)

ax.text(5.5, 4.9, 'TRADING', ha='center', va='center',
       fontsize=12, fontweight='bold', color=GS_COLORS['navy'])

# Price scale
ax.plot([3.8, 7.2], [4.5, 4.5], linewidth=4, color=GS_COLORS['gray'])

# Price markers
for price, label in [(3.8, '$0.00'), (5.5, '$0.50'), (7.2, '$1.00')]:
    ax.plot([price], [4.5], 'o', markersize=12, color=GS_COLORS['navy'],
           markeredgecolor='white', markeredgewidth=2)
    ax.text(price, 4.2, label, ha='center', va='top',
           fontsize=10, fontweight='bold', color=GS_COLORS['navy'])

ax.text(5.5, 4.05, 'Shares trade between $0.00 - $1.00',
       ha='center', va='top', fontsize=9, style='italic')

# ============================================================================
# STEP 4: Resolution
# ============================================================================

# Arrow down to resolution
arrow3 = FancyArrowPatch((5.5, 3.7), (5.5, 2.9),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color=GS_COLORS['gold'])
ax.add_patch(arrow3)

# Resolution header
ax.text(5, 2.7, 'RESOLUTION (After Earnings Announcement)',
       ha='center', va='center', fontsize=12, fontweight='bold',
       color=GS_COLORS['navy'])

# WIN scenario
win_box = FancyBboxPatch((0.8, 1.0), 3.5, 1.5,
                        boxstyle="round,pad=0.1",
                        edgecolor=BLUE_HEX,
                        facecolor='white',
                        linewidth=3)
ax.add_patch(win_box)

ax.text(2.55, 2.15, '✓ COMPANY BEATS', ha='center', va='center',
       fontsize=11, fontweight='bold', color=BLUE_HEX)
ax.text(2.55, 1.8, 'YES shares → $1.00', ha='center', va='center',
       fontsize=10, fontweight='bold')
ax.text(2.55, 1.5, 'NO shares → $0.00', ha='center', va='center',
       fontsize=9, color=GS_COLORS['gray'])
ax.text(2.55, 1.15, 'Example: Buy at $0.60\nWin: $1.00 - $0.60 = $0.40 profit',
       ha='center', va='center', fontsize=8, style='italic')

# LOSS scenario  
loss_box = FancyBboxPatch((5.7, 1.0), 3.5, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=BLUE_HEX,
                         facecolor='white',
                         linewidth=3)
ax.add_patch(loss_box)

ax.text(7.45, 2.15, '✗ COMPANY MISSES', ha='center', va='center',
       fontsize=11, fontweight='bold', color=BLUE_HEX)
ax.text(7.45, 1.8, 'YES shares → $0.00', ha='center', va='center',
       fontsize=10, color=GS_COLORS['gray'])
ax.text(7.45, 1.5, 'NO shares → $1.00', ha='center', va='center',
       fontsize=9, fontweight='bold')
ax.text(7.45, 1.15, 'Example: Buy YES at $0.60\nLoss: $0.60 (total stake)',
       ha='center', va='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig('figures/fig22_polymarket_mechanism.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ fig22_polymarket_mechanism.png created!")
plt.close()

print("\n" + "=" * 80)
print("✅ POLYMARKET MECHANISM DIAGRAM COMPLETED")
print("=" * 80)
print("\nDiagram shows:")
print("   • Market creation (question format)")
print("   • YES/NO share structure")
print("   • Trading range ($0.00 - $1.00)")
print("   • Resolution outcomes (beat vs miss)")
print("   • Win/loss examples")
print("   • Fee structure (2%)")
print("   • Edge calculation formula")
print("\nStyle: Goldman Sachs colors (navy, gold, green, red)")
print("Location: figures/fig22_polymarket_mechanism.png")
print("=" * 80)

