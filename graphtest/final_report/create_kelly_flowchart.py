#!/usr/bin/env python3
"""
Kelly Criterion Flowchart
==========================

Visual flowchart showing Kelly calculation step-by-step.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BLUE = '#2295DC'
GRAY = '#4A4A4A'

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'KELLY CRITERION: OPTIMAL POSITION SIZING',
       ha='center', fontsize=16, fontweight='bold', color=BLUE)

# =============================================================================
# INPUTS
# =============================================================================

# Input box
input_box = FancyBboxPatch((1, 8), 8, 0.8,
                          boxstyle="round,pad=0.08",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(input_box)

ax.text(2.5, 8.4, 'INPUTS', ha='center', fontweight='bold', fontsize=11, color=BLUE)
ax.text(2.5, 8.15, 'p = 0.75', ha='center', fontsize=9)
ax.text(2.5, 7.95, '(Model Prob)', ha='center', fontsize=7, style='italic', color=GRAY)

ax.text(5, 8.4, '', ha='center')
ax.text(5, 8.15, 'q = 0.60', ha='center', fontsize=9)
ax.text(5, 7.95, '(Market Price)', ha='center', fontsize=7, style='italic', color=GRAY)

ax.text(7.5, 8.4, '', ha='center')
ax.text(7.5, 8.15, 'Capital = $100', ha='center', fontsize=9)
ax.text(7.5, 7.95, '(Base)', ha='center', fontsize=7, style='italic', color=GRAY)

# Arrow down
arrow1 = FancyArrowPatch((5, 7.9), (5, 7.4),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=BLUE)
ax.add_patch(arrow1)

# =============================================================================
# STEP 1: EDGE
# =============================================================================

step1_box = FancyBboxPatch((2, 6.5), 6, 0.8,
                          boxstyle="round,pad=0.08",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(step1_box)

ax.text(3, 7.05, 'STEP 1: Edge', ha='left', fontweight='bold', fontsize=10, color=BLUE)
ax.text(5, 6.85, 'Edge = p - q = 0.75 - 0.60 = 0.15 (15%)', 
       ha='center', fontsize=9, fontfamily='monospace')

# Arrow
arrow2 = FancyArrowPatch((5, 6.4), (5, 5.9),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=BLUE)
ax.add_patch(arrow2)

# =============================================================================
# STEP 2: KELLY OPTIMAL
# =============================================================================

step2_box = FancyBboxPatch((2, 5.0), 6, 0.8,
                          boxstyle="round,pad=0.08",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(step2_box)

ax.text(3, 5.55, 'STEP 2: Kelly Optimal', ha='left', fontweight='bold', fontsize=10, color=BLUE)
ax.text(5, 5.35, 'f* = Edge / (1 - q) = 0.15 / 0.40 = 0.375 (37.5%)',
       ha='center', fontsize=9, fontfamily='monospace')

# Arrow
arrow3 = FancyArrowPatch((5, 4.9), (5, 4.4),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=BLUE)
ax.add_patch(arrow3)

# =============================================================================
# STEP 3: QUARTER-KELLY
# =============================================================================

step3_box = FancyBboxPatch((2, 3.5), 6, 0.8,
                          boxstyle="round,pad=0.08",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(step3_box)

ax.text(3, 4.05, 'STEP 3: Quarter-Kelly (25%)', ha='left', fontweight='bold', fontsize=10, color=BLUE)
ax.text(5, 3.85, 'Bet = 0.375 × 0.25 × $100 = $9.38',
       ha='center', fontsize=9, fontfamily='monospace')

# Arrow
arrow4 = FancyArrowPatch((5, 3.4), (5, 2.9),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=BLUE)
ax.add_patch(arrow4)

# =============================================================================
# OUTCOMES
# =============================================================================

ax.text(5, 2.7, 'OUTCOMES', ha='center', fontweight='bold', fontsize=11, color=BLUE)

# Win box
win_box = FancyBboxPatch((1, 1.2), 3.5, 1.3,
                        boxstyle="round,pad=0.08",
                        edgecolor=BLUE,
                        facecolor='white',
                        linewidth=2.5)
ax.add_patch(win_box)

ax.text(2.75, 2.3, 'WIN (75% prob)', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(2.75, 2.0, 'Shares: 15.63', ha='center', fontsize=8)
ax.text(2.75, 1.75, 'Payout: $15.63', ha='center', fontsize=8)
ax.text(2.75, 1.5, 'Profit: $6.13', ha='center', fontsize=9, fontweight='bold')

# Lose box
lose_box = FancyBboxPatch((5.5, 1.2), 3.5, 1.3,
                         boxstyle="round,pad=0.08",
                         edgecolor=BLUE,
                         facecolor='white',
                         linewidth=2.5)
ax.add_patch(lose_box)

ax.text(7.25, 2.3, 'LOSE (25% prob)', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(7.25, 2.0, 'Shares: 15.63', ha='center', fontsize=8)
ax.text(7.25, 1.75, 'Worthless: $0.00', ha='center', fontsize=8)
ax.text(7.25, 1.5, 'Loss: -$9.38', ha='center', fontsize=9, fontweight='bold')

# =============================================================================
# EXPECTED VALUE
# =============================================================================

ev_box = FancyBboxPatch((2.5, 0.2), 5, 0.7,
                       boxstyle="round,pad=0.08",
                       edgecolor=BLUE,
                       facecolor='white',
                       linewidth=3)
ax.add_patch(ev_box)

ax.text(5, 0.7, 'EXPECTED VALUE', ha='center', fontweight='bold', fontsize=11, color=BLUE)
ax.text(5, 0.4, 'EV = 0.75 × $6.13 - 0.25 × $9.38 = $2.25',
       ha='center', fontsize=9, fontfamily='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig24_kelly_flowchart.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ fig24_kelly_flowchart.png created!")
plt.close()


