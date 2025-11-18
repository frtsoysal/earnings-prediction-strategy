#!/usr/bin/env python3
"""
Data Architecture Diagram
==========================

Three-layer architecture: Acquisition → Feature Engineering → Model Layer
Clean, professional, single blue color.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

print("Creating data architecture diagram...")

# Blue color
BLUE_HEX = '#2295DC'  # RGB(34, 149, 220)
GRAY = '#4A4A4A'

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'DATA INFRASTRUCTURE ARCHITECTURE',
       ha='center', va='top', fontsize=18, fontweight='bold', color=BLUE_HEX)

ax.text(5, 9.0, 'Three-Layer Pipeline: Acquisition → Engineering → Modeling',
       ha='center', va='top', fontsize=12, style='italic', color=GRAY)

# =============================================================================
# LAYER 1: ACQUISITION
# =============================================================================

# Layer 1 box
layer1_box = FancyBboxPatch((0.5, 6.8), 9, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor=BLUE_HEX,
                           facecolor='white',
                           linewidth=3)
ax.add_patch(layer1_box)

ax.text(5, 8.4, 'LAYER 1: DATA ACQUISITION',
       ha='center', va='center', fontsize=13, fontweight='bold', color=BLUE_HEX)

# API boxes
api_items = [
    ('Alpha Vantage\nAPI', 1.5),
    ('EARNINGS_\nESTIMATES', 2.8),
    ('EARNINGS\nACTUALS', 4.1),
    ('INCOME\nSTATEMENT', 5.4),
    ('CASH\nFLOW', 6.7),
    ('PRICE\nHISTORY', 8.0)
]

for label, x_pos in api_items:
    box = FancyBboxPatch((x_pos-0.45, 7.1), 0.9, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor=BLUE_HEX,
                        facecolor='white',
                        linewidth=2)
    ax.add_patch(box)
    ax.text(x_pos, 7.4, label, ha='center', va='center',
           fontsize=7, fontweight='bold')

ax.text(9.2, 7.4, '498 S&P 500\nCompanies', ha='left', va='center',
       fontsize=8, style='italic', color=GRAY)

# Arrow down
arrow1 = FancyArrowPatch((5, 6.7), (5, 6.1),
                        arrowstyle='->', mutation_scale=35,
                        linewidth=4, color=BLUE_HEX)
ax.add_patch(arrow1)

# =============================================================================
# LAYER 2: FEATURE ENGINEERING
# =============================================================================

# Layer 2 box
layer2_box = FancyBboxPatch((0.5, 4.2), 9, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor=BLUE_HEX,
                           facecolor='white',
                           linewidth=3)
ax.add_patch(layer2_box)

ax.text(5, 5.8, 'LAYER 2: FEATURE ENGINEERING',
       ha='center', va='center', fontsize=13, fontweight='bold', color=BLUE_HEX)

# Feature categories
features = [
    ('Elo Rating\nSystem', 1.5),
    ('Analyst\nConsensus', 3.0),
    ('Estimate\nRevisions', 4.5),
    ('Lagged\nGrowth', 6.0),
    ('Price\nMomentum', 7.5)
]

for label, x_pos in features:
    box = FancyBboxPatch((x_pos-0.5, 4.5), 1.0, 0.7,
                        boxstyle="round,pad=0.05",
                        edgecolor=BLUE_HEX,
                        facecolor='white',
                        linewidth=2)
    ax.add_patch(box)
    ax.text(x_pos, 4.85, label, ha='center', va='center',
           fontsize=8, fontweight='bold')

ax.text(9.2, 5.3, '31 Leak-Safe\nFeatures', ha='left', va='center',
       fontsize=8, style='italic', color=GRAY)
ax.text(9.2, 4.7, '(Temporal\nvalidity)', ha='left', va='center',
       fontsize=7, style='italic', color=GRAY)

# Arrow down
arrow2 = FancyArrowPatch((5, 4.1), (5, 3.5),
                        arrowstyle='->', mutation_scale=35,
                        linewidth=4, color=BLUE_HEX)
ax.add_patch(arrow2)

# =============================================================================
# LAYER 3: MODEL
# =============================================================================

# Layer 3 box
layer3_box = FancyBboxPatch((0.5, 1.6), 9, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor=BLUE_HEX,
                           facecolor='white',
                           linewidth=3)
ax.add_patch(layer3_box)

ax.text(5, 3.2, 'LAYER 3: MACHINE LEARNING MODELS',
       ha='center', va='center', fontsize=13, fontweight='bold', color=BLUE_HEX)

# Model boxes
models = [
    ('Random\nForest', 2.0, '79.86%\nAccuracy'),
    ('XGBoost', 5.0, '79.81%\nAccuracy'),
    ('Logistic\nRegression', 8.0, '82.65%\nAccuracy')
]

for label, x_pos, perf in models:
    box = FancyBboxPatch((x_pos-0.7, 1.9), 1.4, 0.9,
                        boxstyle="round,pad=0.08",
                        edgecolor=BLUE_HEX,
                        facecolor='white',
                        linewidth=2)
    ax.add_patch(box)
    ax.text(x_pos, 2.6, label, ha='center', va='center',
           fontsize=9, fontweight='bold', color=BLUE_HEX)
    ax.text(x_pos, 2.2, perf, ha='center', va='center',
           fontsize=7)

# Preprocessing note
ax.text(5, 1.35, 'Preprocessing: Median Imputation → StandardScaler',
       ha='center', va='center', fontsize=8, style='italic', color=GRAY)

# =============================================================================
# OUTPUT
# =============================================================================

# Arrow down to output
arrow3 = FancyArrowPatch((5, 1.5), (5, 1.0),
                        arrowstyle='->', mutation_scale=35,
                        linewidth=4, color=BLUE_HEX)
ax.add_patch(arrow3)

# Output box
output_box = FancyBboxPatch((3.0, 0.2), 4.0, 0.7,
                           boxstyle="round,pad=0.08",
                           edgecolor=BLUE_HEX,
                           facecolor='white',
                           linewidth=3)
ax.add_patch(output_box)

ax.text(5, 0.7, 'PREDICTION OUTPUT', ha='center', va='center',
       fontsize=11, fontweight='bold', color=BLUE_HEX)
ax.text(5, 0.4, 'Beat Probability (0-1) | Edge Calculation | Kelly Position Size',
       ha='center', va='center', fontsize=8)

# Dataset stats (side annotation)
stats_text = [
    'Dataset:',
    '• 14,239 observations',
    '• 468 companies',
    '• 2011-2025 period',
    '',
    'Performance:',
    '• 82.65% accuracy',
    '• 0.846 ROC-AUC'
]

y_start = 8.2
for i, line in enumerate(stats_text):
    weight = 'bold' if ':' in line else 'normal'
    ax.text(0.2, y_start - i*0.25, line, ha='left', va='top',
           fontsize=8, fontweight=weight, color=GRAY)

plt.tight_layout()
plt.savefig('figures/fig23_data_architecture.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ fig23_data_architecture.png created!")
plt.close()

print("\n" + "=" * 80)
print("✅ DATA ARCHITECTURE DIAGRAM COMPLETED")
print("=" * 80)
print("\nDiagram shows:")
print("   • Layer 1: Data Acquisition (Alpha Vantage API)")
print("   • Layer 2: Feature Engineering (31 features)")
print("   • Layer 3: ML Models (RF, XGB, LR)")
print("   • Output: Predictions & Position Sizing")
print("\nStyle: Single blue color (#2295DC), white background")
print("Location: figures/fig23_data_architecture.png")
print("=" * 80)


