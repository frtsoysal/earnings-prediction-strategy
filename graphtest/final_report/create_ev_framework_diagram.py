#!/usr/bin/env python3
"""
Expected Value Framework - Conceptual Diagram
==============================================

Visual representation of the mathematical framework described in Section 2.4
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

BLUE = '#2295DC'
LIGHT_BLUE = '#89CFF0'
GRAY = '#4A4A4A'

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Title
ax.text(7, 10.5, 'EXPECTED VALUE FRAMEWORK FOR POLYMARKET TRADING',
       ha='center', fontsize=15, fontweight='bold', color=BLUE)

# =============================================================================
# LAYER 1: INPUTS (Model & Market)
# =============================================================================

ax.text(7, 9.8, 'LAYER 1: PROBABILITY ESTIMATION', 
       ha='center', fontsize=11, fontweight='bold', color=GRAY)

# Model box
model_box = FancyBboxPatch((1, 8.5), 4.5, 1,
                          boxstyle="round,pad=0.1",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(model_box)

ax.text(3.25, 9.2, 'ML MODEL', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(3.25, 8.9, 'p = P(Beat | Features)', ha='center', fontsize=9, style='italic')
ax.text(3.25, 8.65, 'RF, XGB, LR ensemble', ha='center', fontsize=7, color=GRAY)

# Market box
market_box = FancyBboxPatch((8.5, 8.5), 4.5, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor=BLUE,
                           facecolor='white',
                           linewidth=2.5)
ax.add_patch(market_box)

ax.text(10.75, 9.2, 'POLYMARKET', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(10.75, 8.9, 'q = Market Price', ha='center', fontsize=9, style='italic')
ax.text(10.75, 8.65, 'Crowd-sourced probability', ha='center', fontsize=7, color=GRAY)

# Arrows down
arrow1 = FancyArrowPatch((3.25, 8.4), (4.5, 7.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=BLUE)
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((10.75, 8.4), (9.5, 7.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=BLUE)
ax.add_patch(arrow2)

# =============================================================================
# LAYER 2: EDGE CALCULATION
# =============================================================================

ax.text(7, 7.8, 'LAYER 2: EDGE IDENTIFICATION', 
       ha='center', fontsize=11, fontweight='bold', color=GRAY)

edge_box = FancyBboxPatch((4.5, 6.7), 5, 0.9,
                         boxstyle="round,pad=0.1",
                         edgecolor=BLUE,
                         facecolor='white',
                         linewidth=2.5)
ax.add_patch(edge_box)

ax.text(7, 7.35, 'EDGE CALCULATION', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(7, 7.05, 'e = p - q', ha='center', fontsize=11, fontfamily='monospace', fontweight='bold')
ax.text(7, 6.8, 'Positive edge → Profitable opportunity', ha='center', fontsize=7, color=GRAY)

# Decision diamond
ax.text(2, 7.1, 'e > 0?', ha='center', fontsize=9, fontweight='bold', color=BLUE,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=BLUE, linewidth=2))

# No trade arrow
arrow_no = FancyArrowPatch((1.5, 7.1), (0.5, 7.1),
                          arrowstyle='->', mutation_scale=15,
                          linewidth=2, color=GRAY, linestyle='--')
ax.add_patch(arrow_no)
ax.text(0.3, 7.4, 'NO', ha='center', fontsize=7, fontweight='bold', color=GRAY)
ax.text(0.3, 6.8, 'Skip', ha='center', fontsize=7, color=GRAY)

# Yes arrow
arrow3 = FancyArrowPatch((7, 6.6), (7, 6.1),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=BLUE)
ax.add_patch(arrow3)

# =============================================================================
# LAYER 3: KELLY CRITERION
# =============================================================================

ax.text(7, 5.95, 'LAYER 3: POSITION SIZING', 
       ha='center', fontsize=11, fontweight='bold', color=GRAY)

kelly_box = FancyBboxPatch((3.5, 4.5), 7, 1.3,
                          boxstyle="round,pad=0.1",
                          edgecolor=BLUE,
                          facecolor='white',
                          linewidth=2.5)
ax.add_patch(kelly_box)

ax.text(7, 5.55, 'KELLY CRITERION', ha='center', fontweight='bold', fontsize=10, color=BLUE)

# Formula box
formula_rect = Rectangle((4.5, 4.85), 5, 0.5, 
                         facecolor=LIGHT_BLUE, alpha=0.2, 
                         edgecolor=BLUE, linewidth=1.5)
ax.add_patch(formula_rect)

ax.text(7, 5.25, 'f* = e / (1 - q)', ha='center', fontsize=12, 
       fontfamily='monospace', fontweight='bold', color=BLUE)
ax.text(7, 5.0, 'Optimal fraction of capital', ha='center', fontsize=7, style='italic', color=GRAY)

ax.text(7, 4.7, 'Fractional Kelly (25% of f*) for risk management', 
       ha='center', fontsize=8, color=GRAY)

# Arrow down
arrow4 = FancyArrowPatch((7, 4.4), (7, 3.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=BLUE)
ax.add_patch(arrow4)

# =============================================================================
# LAYER 4: EXPECTED VALUE
# =============================================================================

ax.text(7, 3.75, 'LAYER 4: EXPECTED VALUE CALCULATION', 
       ha='center', fontsize=11, fontweight='bold', color=GRAY)

# Main EV box
ev_box = FancyBboxPatch((2, 2.1), 10, 1.4,
                       boxstyle="round,pad=0.1",
                       edgecolor=BLUE,
                       facecolor='white',
                       linewidth=2.5)
ax.add_patch(ev_box)

ax.text(7, 3.3, 'EXPECTED VALUE', ha='center', fontweight='bold', fontsize=11, color=BLUE)

# Formula
formula_rect2 = Rectangle((3, 2.55), 8, 0.5, 
                         facecolor=LIGHT_BLUE, alpha=0.2, 
                         edgecolor=BLUE, linewidth=1.5)
ax.add_patch(formula_rect2)

ax.text(7, 2.92, 'EV = p × Profit        - (1-p) × Loss', 
       ha='center', fontsize=10, fontfamily='monospace', fontweight='bold')
ax.text(7, 2.72, '           if Win                          if Lose', 
       ha='center', fontsize=7, style='italic', color=GRAY)

ax.text(7, 2.35, 'Profit = (Shares × $1.00 - Bet) × 0.98  |  Loss = Bet', 
       ha='center', fontsize=7, fontfamily='monospace', color=GRAY)

# =============================================================================
# LAYER 5: OUTCOMES
# =============================================================================

ax.text(7, 1.9, 'LAYER 5: PORTFOLIO OUTCOMES', 
       ha='center', fontsize=11, fontweight='bold', color=GRAY)

# Positive EV
pos_box = FancyBboxPatch((1, 0.5), 5, 1.2,
                        boxstyle="round,pad=0.1",
                        edgecolor=BLUE,
                        facecolor='white',
                        linewidth=2.5)
ax.add_patch(pos_box)

ax.text(3.5, 1.5, 'EV > 0', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(3.5, 1.2, '✓ Place Kelly-weighted bet', ha='center', fontsize=8)
ax.text(3.5, 0.95, '✓ Expected profit over time', ha='center', fontsize=8)
ax.text(3.5, 0.7, '✓ Compound edge across', ha='center', fontsize=8)
ax.text(3.5, 0.55, '   multiple opportunities', ha='center', fontsize=8)

# Strategy example
strat_box = FancyBboxPatch((8, 0.5), 5, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor=BLUE,
                          facecolor=LIGHT_BLUE,
                          alpha=0.3,
                          linewidth=2.5)
ax.add_patch(strat_box)

ax.text(10.5, 1.5, 'EMPIRICAL RESULTS', ha='center', fontweight='bold', fontsize=10, color=BLUE)
ax.text(10.5, 1.2, 'Sweet Spot (10-15% edge):', ha='center', fontsize=8, fontweight='bold')
ax.text(10.5, 0.95, 'ROI: +32.73%', ha='center', fontsize=8)
ax.text(10.5, 0.75, 'Win Rate: 93.75%', ha='center', fontsize=8)
ax.text(10.5, 0.55, 'Avg EV: $8.12 per trade', ha='center', fontsize=8)

# =============================================================================
# SIDE NOTES
# =============================================================================

# Risk box
risk_box = FancyBboxPatch((11.5, 4.5), 2.3, 1.3,
                         boxstyle="round,pad=0.08",
                         edgecolor=GRAY,
                         facecolor='white',
                         linewidth=1.5,
                         linestyle='--')
ax.add_patch(risk_box)

ax.text(12.65, 5.6, 'RISK MGMT', ha='center', fontsize=8, fontweight='bold', color=GRAY)
ax.text(12.65, 5.35, 'Full Kelly:', ha='center', fontsize=7)
ax.text(12.65, 5.15, 'High volatility', ha='center', fontsize=6, color=GRAY)
ax.text(12.65, 4.95, '', ha='center')
ax.text(12.65, 4.8, 'Quarter-Kelly:', ha='center', fontsize=7)
ax.text(12.65, 4.6, 'Reduced risk,', ha='center', fontsize=6, color=GRAY)
ax.text(12.65, 4.45, 'stable growth', ha='center', fontsize=6, color=GRAY)

# Fees box
fee_box = FancyBboxPatch((0.2, 2.1), 1.5, 0.8,
                        boxstyle="round,pad=0.08",
                        edgecolor=GRAY,
                        facecolor='white',
                        linewidth=1.5,
                        linestyle='--')
ax.add_patch(fee_box)

ax.text(0.95, 2.75, 'FEES', ha='center', fontsize=7, fontweight='bold', color=GRAY)
ax.text(0.95, 2.5, 'Polymarket:', ha='center', fontsize=6)
ax.text(0.95, 2.35, '2% on wins', ha='center', fontsize=6, color=GRAY)

plt.tight_layout()
plt.savefig('figures/fig25_ev_framework_diagram.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ fig25_ev_framework_diagram.png created!")
plt.close()

print("\nDiagram shows 5 layers:")
print("  Layer 1: Probability Estimation (Model vs Market)")
print("  Layer 2: Edge Identification (p - q)")
print("  Layer 3: Position Sizing (Kelly Criterion)")
print("  Layer 4: Expected Value Calculation")
print("  Layer 5: Portfolio Outcomes & Results")


