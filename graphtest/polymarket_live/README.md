# Polymarket Live Trading Recommendations

Automated system for generating real-time trading recommendations on Polymarket earnings events using our global ML models.

## Overview

This system:
1. Fetches **open** Polymarket earnings beat markets
2. Retrieves historical data for each ticker from Alpha Vantage
3. Applies our trained global ML models (Random Forest, XGBoost, Logistic Regression)
4. Calculates edge (model probability - market price)
5. Recommends positions using Kelly criterion (both quarter-Kelly and full Kelly)

## Usage

```bash
cd /Users/ibrahimfiratsoysal/Documents/ML/graphtest/polymarket_live
python3 live_trading_recommendations.py
```

## Output

### Terminal Output
Shows:
- Number of open markets found
- Data fetch success rate
- Total markets analyzed
- **Strategy Breakdown** - Count and average metrics for each strategy:
  - Sweet Spot (10-15% edge)
  - Positive Edge (>0%)
  - Moderate Edge (5-20%)
  - High Edge (≥8%)
  - Avoid Overconfidence (edge>0, prob<0.80)
  - High Conviction (edge>10%, prob 0.7-0.85)
  - Combination (YES if edge>10%, NO if edge<-10%)
  - Counter-Trade (bet NO on negative edge)
- Top 10 recommendations (sorted by edge) with:
  - Ticker symbol
  - Direction (YES or NO)
  - Edge percentage
  - Position sizes (quarter-Kelly and full Kelly for $100 base)
  - Expected values
  - Model and market probabilities
  - **Applicable strategies**
- **Sweet Spot section** - Highlights best opportunities with historical ROI

### CSV File: `live_recommendations.csv`

Sorted by edge (descending), contains:

| Column | Description |
|--------|-------------|
| `ticker` | Stock ticker symbol |
| `market_id` | Polymarket market ID |
| `question` | Market question text |
| `model_prob_lr` | Logistic Regression probability (primary model) |
| `model_prob_rf` | Random Forest probability |
| `model_prob_xgb` | XGBoost probability |
| `market_price` | Current Polymarket price (0-1) |
| `edge` | Model prob - Market price |
| `kelly_optimal_pct` | Optimal Kelly fraction |
| `position_quarter_kelly_$100` | Position size at 25% Kelly for $100 capital |
| `position_full_kelly_$100` | Position size at 100% Kelly for $100 capital |
| `expected_value_quarter_kelly` | Expected profit/loss for quarter-Kelly |
| `expected_value_full_kelly` | Expected profit/loss for full Kelly |
| `strategies` | Comma-separated list of applicable strategies |
| `polymarket_url` | Direct link to market |

## Trading Strategies

The system analyzes all markets and classifies them into multiple strategies based on backtested performance. Each market can qualify for multiple strategies simultaneously.

### Strategy Definitions

1. **Sweet Spot (10-15% edge)** - Best historical performance (+32.73% ROI, 93.75% win rate)
2. **Positive Edge (>0%)** - All markets where model is more bullish than market (+13.07% ROI)
3. **Moderate Edge (5-20%)** - Mid-range positive edge opportunities (+13.04% ROI)
4. **High Edge (≥8%)** - Strong model conviction (+12.23% ROI)
5. **Avoid Overconfidence (edge>0, prob<0.80)** - Positive edge without overconfidence (+4.79% ROI)
6. **High Conviction (edge>10%, prob 0.7-0.85)** - Sweet spot probability range (varies)
7. **Combination (YES if edge>10%, NO if edge<-10%)** - Trade both extremes (+4.48% ROI)
8. **Counter-Trade (bet NO on negative edge)** - Fade the model when market is more bullish (-2.31% ROI historical)

### Position Sizing

- **Quarter-Kelly (25%)**: Conservative, recommended for most traders. Reduces volatility while capturing most of the edge.
- **Full Kelly (100%)**: Aggressive, mathematically optimal but high volatility. Only for sophisticated traders with high risk tolerance.

### Risk Management

- Base capital: $100 (scales linearly)
- Incorporates 2% Polymarket fee in EV calculations
- Expected value shows profit/loss per trade after fees

## Requirements

- Python 3.7+
- pandas, numpy, joblib, sklearn
- Trained global models in `graphtest/global_model/models/`
- Alpha Vantage API access (key in script)

## Example Output

```
================================================================================
STRATEGY BREAKDOWN
================================================================================

Sweet Spot               :  1 markets | Avg Edge: +14.8% | Avg EV (Q-K): $+2.26
Positive Edge            :  4 markets | Avg Edge: +8.7% | Avg EV (Q-K): $+1.00
Moderate Edge            :  3 markets | Avg Edge: +10.0% | Avg EV (Q-K): $+1.18
High Edge                :  2 markets | Avg Edge: +11.9% | Avg EV (Q-K): $+1.56
Counter-Trade            : 33 markets | Avg Edge: -37.7% | Avg EV (Q-K): $+49.73

================================================================================
TOP 10 RECOMMENDATIONS (BY EDGE)
================================================================================

TGT    | YES | Edge: +14.8% | Quarter-Kelly: $10.12 | Full-Kelly: $40.48
       Model: 78.3% | Market: 63.5% | EV (Q-K): $2.26 | EV (F-K): $9.06
       Strategies: Sweet Spot, Positive Edge, Moderate Edge, High Edge
       Will Target (TGT) beat quarterly earnings?...

================================================================================
🎯 SWEET SPOT OPPORTUNITIES (1 markets)
================================================================================
Historical Performance: +32.73% ROI, 93.75% win rate

TGT    | Edge: +14.8% | Model: 78.3% | Market: 63.5%
       Quarter-Kelly: $10.12 (EV: $2.26)
```

## Notes

- Markets are filtered for "beat" + "earnings" keywords
- Data is fetched in real-time (may take 1-2 minutes for 40+ markets)
- Failed ticker fetches are logged and skipped
- 1-second delay between Alpha Vantage calls (rate limit compliance)

## Historical Performance

Based on backtest (262 events):
- **Sweet Spot (10-15% edge)**: +32.73% ROI, 93.75% win rate
- **All Positive Edge**: +13.07% ROI, 72.22% win rate

See `graphtest/polymarket_backtest/` for full analysis.

