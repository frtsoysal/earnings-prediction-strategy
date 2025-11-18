"""
KELLY CRITERION & EXPECTED VALUE FRAMEWORK
===========================================

Python implementation with worked example for presentation.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("KELLY CRITERION & EXPECTED VALUE - PYTHON IMPLEMENTATION")
print("=" * 80)

# =============================================================================
# EXAMPLE SCENARIO
# =============================================================================

print("\n📊 EXAMPLE SCENARIO:")
print("-" * 80)

# Model and market inputs
p_model = 0.75      # Our model: 75% probability of beat
q_market = 0.60     # Polymarket price: $0.60 per share
base_capital = 100  # Base bet size ($100)

print(f"Model Probability (p):      {p_model:.2%}")
print(f"Market Price (q):            ${q_market:.2f}")
print(f"Base Capital:                ${base_capital:.2f}")

# =============================================================================
# STEP 1: EDGE CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: CALCULATE EDGE")
print("=" * 80)

edge = p_model - q_market

print(f"\nEdge = p_model - p_market")
print(f"Edge = {p_model:.2f} - {q_market:.2f}")
print(f"Edge = {edge:+.2f} ({edge:+.1%})")

if edge > 0:
    print(f"\n✓ POSITIVE EDGE: Model sees {edge:.1%} more upside than market")
elif edge < 0:
    print(f"\n✗ NEGATIVE EDGE: Market more bullish than model")
else:
    print(f"\n~ ZERO EDGE: Fair pricing")

# =============================================================================
# STEP 2: KELLY OPTIMAL FRACTION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: KELLY OPTIMAL BET SIZE")
print("=" * 80)

if edge > 0:
    kelly_optimal = edge / (1 - q_market)
    
    print(f"\nKelly Formula: f* = Edge / (1 - q_market)")
    print(f"f* = {edge:.2f} / (1 - {q_market:.2f})")
    print(f"f* = {edge:.2f} / {1-q_market:.2f}")
    print(f"f* = {kelly_optimal:.4f} ({kelly_optimal:.1%} of capital)")
    
    print(f"\n→ Full Kelly says: Bet {kelly_optimal:.1%} of your capital")
    
    # Quarter-Kelly (our approach)
    kelly_fraction = 0.25
    kelly_bet = kelly_optimal * kelly_fraction * base_capital
    
    print(f"\nQuarter-Kelly (25% of optimal for risk management):")
    print(f"Bet = {kelly_optimal:.4f} × {kelly_fraction:.2f} × ${base_capital}")
    print(f"Bet = ${kelly_bet:.2f}")
    
else:
    print("\nNo bet (negative edge)")
    kelly_bet = 0

# =============================================================================
# STEP 3: EXPECTED VALUE CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: EXPECTED VALUE (EV)")
print("=" * 80)

if kelly_bet > 0:
    # Shares purchased
    shares = kelly_bet / q_market
    
    # If win
    payout_if_win = shares * 1.00  # Each share pays $1.00
    profit_if_win = payout_if_win - kelly_bet
    profit_after_fee = profit_if_win * 0.98  # 2% Polymarket fee
    
    # If lose
    loss_if_lose = kelly_bet
    
    # Expected value
    ev_gross = (p_model * profit_if_win) - ((1 - p_model) * loss_if_lose)
    ev_net = (p_model * profit_after_fee) - ((1 - p_model) * loss_if_lose)
    
    print(f"\nShares purchased: ${kelly_bet:.2f} / ${q_market:.2f} = {shares:.2f} shares")
    
    print(f"\nIF WIN (probability {p_model:.1%}):")
    print(f"   Payout: {shares:.2f} shares × $1.00 = ${payout_if_win:.2f}")
    print(f"   Profit (gross): ${payout_if_win:.2f} - ${kelly_bet:.2f} = ${profit_if_win:.2f}")
    print(f"   Profit (net, -2% fee): ${profit_after_fee:.2f}")
    
    print(f"\nIF LOSE (probability {1-p_model:.1%}):")
    print(f"   Loss: ${loss_if_lose:.2f} (entire stake)")
    
    print(f"\nEXPECTED VALUE:")
    print(f"   EV (gross) = {p_model:.2f} × ${profit_if_win:.2f} - {1-p_model:.2f} × ${loss_if_lose:.2f}")
    print(f"   EV (gross) = ${ev_gross:.2f}")
    print(f"   EV (net)   = ${ev_net:.2f}")
    
    if ev_net > 0:
        print(f"\n✅ POSITIVE EXPECTED VALUE: ${ev_net:.2f} per trade")
    else:
        print(f"\n❌ NEGATIVE EXPECTED VALUE")

# =============================================================================
# STEP 4: COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: KELLY FRACTION COMPARISON")
print("=" * 80)

if edge > 0:
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    comparison = []
    
    for frac in fractions:
        bet = kelly_optimal * frac * base_capital
        bet = min(bet, base_capital)  # Cap at base
        
        shares_buy = bet / q_market
        profit_win = (shares_buy * 1.00 - bet) * 0.98
        loss_lose = bet
        ev = (p_model * profit_win) - ((1 - p_model) * loss_lose)
        
        comparison.append({
            'Kelly %': f"{frac:.0%}",
            'Bet $': f"{bet:.2f}",
            'EV $': f"{ev:.2f}",
            'Risk': 'Low' if frac <= 0.25 else ('Moderate' if frac <= 0.50 else 'High')
        })
    
    df = pd.DataFrame(comparison)
    print("\n")
    print(df.to_string(index=False))
    
    print(f"\n→ Recommended: 25% Kelly (Balance of risk and reward)")

# =============================================================================
# SAVE CODE SNIPPET FOR PRESENTATION
# =============================================================================

code_snippet = '''
# Kelly Criterion Implementation

def calculate_kelly_bet(p_model, q_market, base_capital=100, kelly_fraction=0.25):
    """
    Calculate optimal bet size using Kelly Criterion
    
    Args:
        p_model: Model probability (0-1)
        q_market: Market price (0-1)
        base_capital: Available capital
        kelly_fraction: Fraction of Kelly (0.25 = quarter-Kelly)
    
    Returns:
        (bet_size, expected_value)
    """
    # Calculate edge
    edge = p_model - q_market
    
    if edge <= 0:
        return 0, 0  # No bet on negative edge
    
    # Kelly optimal fraction
    kelly_optimal = edge / (1 - q_market)
    
    # Apply fractional Kelly (risk management)
    bet = kelly_optimal * kelly_fraction * base_capital
    bet = min(bet, base_capital)  # Cap at available capital
    
    # Expected value
    shares = bet / q_market
    profit_if_win = (shares * 1.00 - bet) * 0.98  # 2% fee
    loss_if_lose = bet
    
    ev = p_model * profit_if_win - (1 - p_model) * loss_if_lose
    
    return bet, ev

# Example usage
bet_size, ev = calculate_kelly_bet(p_model=0.75, q_market=0.60)
print(f"Bet: ${bet_size:.2f}, Expected Value: ${ev:.2f}")
'''

with open('kelly_criterion_code.txt', 'w') as f:
    f.write(code_snippet)

print("\n" + "=" * 80)
print("✅ PYTHON CODE SAVED")
print("=" * 80)
print("\nSaved: kelly_criterion_code.txt")
print("Use this code snippet in your presentation slides!")
print("=" * 80)


