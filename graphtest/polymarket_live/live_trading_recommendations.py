#!/usr/bin/env python3
"""
Polymarket Live Trading Recommendations
========================================

Fetches open Polymarket earnings markets, applies ML models, and recommends
positions with positive edge using Kelly criterion.

Output:
    live_recommendations.csv - Sorted by edge (best opportunities first)
    
Usage:
    python live_trading_recommendations.py
"""

import urllib.request
import urllib.parse
import json
import pandas as pd
import numpy as np
import joblib
import subprocess
import time
import ssl
import re
import os
import sys
from pathlib import Path
from datetime import datetime

# SSL verification bypass
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 80)
print("POLYMARKET LIVE TRADING RECOMMENDATIONS")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
EARNINGS_TAG_ID = "1013"
BASE_CAPITAL = 100  # Base position size for Kelly calculations
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your Alpha Vantage API key

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "graphtest" / "global_model" / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "with_estimates" / "fetch_alpha_vantage.py"

# Safe features (from train_global_model.py)
SAFE_FEATURES = [
    # Price levels BEFORE report
    'price_1m_before',
    'price_3m_before',
    
    # EPS estimates
    'eps_estimate_average',
    'eps_estimate_high',
    'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago',
    'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago',
    'eps_estimate_average_90_days_ago',
    
    # Analyst revisions
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    
    # Revenue estimates
    'revenue_estimate_average',
    'revenue_estimate_high',
    'revenue_estimate_low',
    'revenue_estimate_analyst_count',
    
    # Historical Elo
    'elo_before',
    'elo_decay',
    'elo_momentum',
    'elo_vol_4q',
    
    # LAGGED growth metrics
    'total_revenue_yoy_growth_lag1',
    'total_revenue_qoq_growth_lag1',
    'total_revenue_ttm_yoy_growth_lag1',
    'actual_eps_yoy_growth_lag1',
    'actual_eps_qoq_growth_lag1',
    'ebitda_yoy_growth_lag1',
    'operating_income_yoy_growth_lag1',
    'gross_margin_yoy_change_lag1',
    'operating_margin_yoy_change_lag1'
]

# =============================================================================
# STEP 1: FETCH OPEN POLYMARKET MARKETS
# =============================================================================

print(f"\n📊 1/5 Fetching open earnings markets from Polymarket...")

def fetch_open_polymarket_markets():
    """Fetch open earnings markets from Polymarket Gamma API"""
    all_markets = []
    offset = 0
    limit = 100
    
    while True:
        try:
            url = f"{GAMMA_API_BASE}/markets"
            params = {
                'limit': limit,
                'offset': offset,
                'tag_id': EARNINGS_TAG_ID,
                'closed': 'false',  # Only open markets
                'order': 'id',
                'ascending': 'false'
            }
            
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"
            
            req = urllib.request.Request(full_url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            req.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            # Gamma API returns a list directly
            markets = data if isinstance(data, list) else []
            
            if len(markets) == 0:
                break
            
            # Filter for earnings beat markets
            for market in markets:
                question = str(market.get('question', '') or '')
                
                # Must contain "beat" and "earnings"
                if 'beat' in question.lower() and 'earnings' in question.lower():
                    # Extract ticker from question
                    ticker_match = re.search(r'\b([A-Z]{1,5})\b', question)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        
                        # Get current price (average of bid/ask from market level)
                        current_price = None
                        
                        # Try market-level bestBid/bestAsk first
                        best_bid = market.get('bestBid')
                        best_ask = market.get('bestAsk')
                        
                        if best_bid is not None and best_ask is not None:
                            try:
                                current_price = (float(best_bid) + float(best_ask)) / 2
                            except:
                                pass
                        
                        # Fallback to outcomePrices
                        if current_price is None:
                            outcome_prices = market.get('outcomePrices')
                            if outcome_prices and isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                                try:
                                    # First outcome is typically "Yes"
                                    current_price = float(outcome_prices[0])
                                except:
                                    pass
                        
                        # Fallback to lastTradePrice
                        if current_price is None:
                            last_trade = market.get('lastTradePrice')
                            if last_trade:
                                try:
                                    current_price = float(last_trade)
                                except:
                                    pass
                        
                        all_markets.append({
                            'market_id': market.get('id'),
                            'ticker': ticker,
                            'question': question,
                            'current_price': current_price,
                            'polymarket_url': f"https://polymarket.com/event/{market.get('slug', '')}"
                        })
            
            offset += limit
            
            # Stop if we got less than limit (last page)
            if len(markets) < limit:
                break
                
        except Exception as e:
            print(f"   ⚠️  API error at offset {offset}: {e}")
            break
    
    return all_markets

markets = fetch_open_polymarket_markets()
print(f"   ✓ Found {len(markets)} open earnings beat markets")

if len(markets) == 0:
    print("\n❌ No open markets found. Exiting.")
    sys.exit(0)

# =============================================================================
# STEP 2: LOAD ML MODELS
# =============================================================================

print(f"\n🤖 2/5 Loading global ML models...")

try:
    model_lr = joblib.load(MODEL_DIR / 'global_lr_model.pkl')
    model_rf = joblib.load(MODEL_DIR / 'global_rf_model.pkl')
    model_xgb = joblib.load(MODEL_DIR / 'global_xgb_model.pkl')
    preprocessor = joblib.load(MODEL_DIR / 'global_preprocessor.pkl')
    print(f"   ✓ Loaded all 3 models + preprocessor")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    sys.exit(1)

# =============================================================================
# STEP 3: FETCH ALPHA VANTAGE DATA FOR EACH TICKER
# =============================================================================

print(f"\n📈 3/5 Fetching Alpha Vantage data for tickers...")

results = []
failed_tickers = []

for i, market in enumerate(markets, 1):
    ticker = market['ticker']
    print(f"   [{i}/{len(markets)}] {ticker}...", end=" ")
    
    try:
        # Check if data file already exists
        csv_path = DATA_DIR / f"{ticker}_earnings_with_q4.csv"
        
        # Fetch if not exists
        if not csv_path.exists():
            # Call fetch script
            cmd = [
                'python3',
                str(FETCH_SCRIPT),
                '--symbol', ticker,
                '--api-key', API_KEY
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"FAILED (fetch error)")
                failed_tickers.append(ticker)
                continue
            
            # Rate limit (1 second delay)
            time.sleep(1)
        
        # Read CSV
        if not csv_path.exists():
            print(f"FAILED (no file)")
            failed_tickers.append(ticker)
            continue
            
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print(f"FAILED (empty)")
            failed_tickers.append(ticker)
            continue
        
        # Get most recent quarter (prefer Q4 if available, else latest)
        df_sorted = df.sort_values('date', ascending=False)
        
        # Check for Q4 estimate (no actual_eps yet)
        q4_rows = df_sorted[df_sorted['actual_eps'].isna()]
        if len(q4_rows) > 0:
            row = q4_rows.iloc[0]
        else:
            # Use most recent historical
            row = df_sorted.iloc[0]
        
        # Extract features
        feature_dict = {}
        for feat in SAFE_FEATURES:
            feature_dict[feat] = row.get(feat, np.nan)
        
        # Add to results
        results.append({
            'ticker': ticker,
            'market_id': market['market_id'],
            'question': market['question'],
            'market_price': market['current_price'],
            'polymarket_url': market['polymarket_url'],
            'features': feature_dict
        })
        
        print(f"OK")
        
    except subprocess.TimeoutExpired:
        print(f"FAILED (timeout)")
        failed_tickers.append(ticker)
    except Exception as e:
        print(f"FAILED ({str(e)[:30]})")
        failed_tickers.append(ticker)

print(f"\n   ✓ Successfully fetched: {len(results)} tickers")
print(f"   ✗ Failed: {len(failed_tickers)} tickers")

if len(results) == 0:
    print("\n❌ No data available for predictions. Exiting.")
    sys.exit(0)

# =============================================================================
# STEP 4: GENERATE PREDICTIONS
# =============================================================================

print(f"\n🎯 4/5 Generating ML predictions...")

predictions = []

for item in results:
    try:
        # Create feature dataframe
        features_df = pd.DataFrame([item['features']])[SAFE_FEATURES]
        
        # Fill revision columns with 0 (NaN means no revision)
        revision_cols = [c for c in SAFE_FEATURES if 'revision' in c]
        features_df[revision_cols] = features_df[revision_cols].fillna(0)
        
        # Preprocess
        features_processed = preprocessor.transform(features_df)
        
        # Get probabilities from all models
        prob_lr = model_lr.predict_proba(features_processed)[0][1]
        prob_rf = model_rf.predict_proba(features_processed)[0][1]
        prob_xgb = model_xgb.predict_proba(features_processed)[0][1]
        
        predictions.append({
            'ticker': item['ticker'],
            'market_id': item['market_id'],
            'question': item['question'],
            'model_prob_lr': prob_lr,
            'model_prob_rf': prob_rf,
            'model_prob_xgb': prob_xgb,
            'market_price': item['market_price'],
            'polymarket_url': item['polymarket_url']
        })
        
    except Exception as e:
        print(f"   ⚠️  {item['ticker']}: Prediction failed - {e}")
        continue

print(f"   ✓ Generated predictions for {len(predictions)} markets")

# =============================================================================
# STEP 5: CALCULATE EDGE, KELLY SIZING, AND STRATEGY CLASSIFICATION
# =============================================================================

print(f"\n💰 5/5 Calculating edge, Kelly sizing, and strategy classification...")

recommendations = []

for pred in predictions:
    p_model = pred['model_prob_lr']  # Use LR as primary (best performer)
    p_market = pred['market_price']
    
    if p_market is None or p_market <= 0 or p_market >= 1:
        continue
    
    # Calculate edge
    edge = p_model - p_market
    
    # Kelly optimal fraction (works for both positive and negative edge)
    if edge > 0:
        kelly_optimal = edge / (1 - p_market)
    else:
        # For negative edge (counter-trade), calculate for betting NO
        kelly_optimal = -edge / p_market
    
    # Position sizes
    # Quarter-Kelly (25% of optimal)
    position_quarter_kelly = abs(kelly_optimal) * 0.25 * BASE_CAPITAL
    position_quarter_kelly = min(position_quarter_kelly, BASE_CAPITAL)
    
    # Full Kelly (100% of optimal)
    position_full_kelly = abs(kelly_optimal) * 1.00 * BASE_CAPITAL
    position_full_kelly = min(position_full_kelly, BASE_CAPITAL)
    
    # Expected value calculations (for YES position with positive edge)
    if edge > 0:
        # Quarter-Kelly
        shares_qk = position_quarter_kelly / p_market
        profit_if_win_qk = (shares_qk * 1.00 - position_quarter_kelly) * 0.98  # 2% fee
        loss_if_lose_qk = position_quarter_kelly
        ev_quarter_kelly = p_model * profit_if_win_qk - (1 - p_model) * loss_if_lose_qk
        
        # Full Kelly
        shares_fk = position_full_kelly / p_market
        profit_if_win_fk = (shares_fk * 1.00 - position_full_kelly) * 0.98
        loss_if_lose_fk = position_full_kelly
        ev_full_kelly = p_model * profit_if_win_fk - (1 - p_model) * loss_if_lose_fk
    else:
        # For negative edge, calculate EV for NO position
        no_price = 1 - p_market
        shares_qk = position_quarter_kelly / no_price
        profit_if_win_qk = (shares_qk * 1.00 - position_quarter_kelly) * 0.98
        loss_if_lose_qk = position_quarter_kelly
        ev_quarter_kelly = (1 - p_model) * profit_if_win_qk - p_model * loss_if_lose_qk
        
        shares_fk = position_full_kelly / no_price
        profit_if_win_fk = (shares_fk * 1.00 - position_full_kelly) * 0.98
        loss_if_lose_fk = position_full_kelly
        ev_full_kelly = (1 - p_model) * profit_if_win_fk - p_model * loss_if_lose_fk
    
    # Strategy classification
    strategies = []
    
    # 1. Sweet Spot (10-15% edge)
    if 0.10 <= edge <= 0.15:
        strategies.append("Sweet Spot")
    
    # 2. Positive Edge Only (>0%)
    if edge > 0:
        strategies.append("Positive Edge")
    
    # 3. Moderate Edge (5-20%)
    if 0.05 <= edge <= 0.20:
        strategies.append("Moderate Edge")
    
    # 4. High Edge (>=8%)
    if edge >= 0.08:
        strategies.append("High Edge")
    
    # 5. Avoid Overconfidence (edge>0, prob<0.80)
    if edge > 0 and p_model < 0.80:
        strategies.append("Avoid Overconfidence")
    
    # 6. Combination (YES if edge>10%, NO if edge<-10%)
    if edge > 0.10 or edge < -0.10:
        strategies.append("Combination")
    
    # 7. Counter-Trade (bet NO on negative edge)
    if edge < 0:
        strategies.append("Counter-Trade")
    
    # 8. High Conviction (edge>10%, prob 0.7-0.85)
    if edge > 0.10 and 0.70 <= p_model <= 0.85:
        strategies.append("High Conviction")
    
    recommendations.append({
        'ticker': pred['ticker'],
        'market_id': pred['market_id'],
        'question': pred['question'],
        'model_prob_lr': p_model,
        'model_prob_rf': pred['model_prob_rf'],
        'model_prob_xgb': pred['model_prob_xgb'],
        'market_price': p_market,
        'edge': edge,
        'kelly_optimal_pct': kelly_optimal,
        'position_quarter_kelly_$100': position_quarter_kelly,
        'position_full_kelly_$100': position_full_kelly,
        'expected_value_quarter_kelly': ev_quarter_kelly,
        'expected_value_full_kelly': ev_full_kelly,
        'strategies': ', '.join(strategies) if strategies else 'None',
        'polymarket_url': pred['polymarket_url']
    })

print(f"   ✓ Analyzed {len(recommendations)} markets")

# =============================================================================
# STEP 6: SORT AND OUTPUT
# =============================================================================

if len(recommendations) == 0:
    print("\n❌ No markets analyzed.")
    sys.exit(0)

# Convert to DataFrame and sort by edge (descending)
df_recommendations = pd.DataFrame(recommendations)
df_recommendations = df_recommendations.sort_values('edge', ascending=False)

# Save to CSV
output_path = SCRIPT_DIR / 'live_recommendations.csv'
df_recommendations.to_csv(output_path, index=False)

# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nMarkets analyzed: {len(markets)}")
print(f"Data fetched successfully: {len(results)}")
print(f"Predictions generated: {len(predictions)}")
print(f"Total recommendations: {len(recommendations)}")

# Strategy breakdown
print(f"\n{'='*80}")
print("STRATEGY BREAKDOWN")
print(f"{'='*80}\n")

strategy_names = [
    "Sweet Spot",
    "Positive Edge",
    "Moderate Edge",
    "High Edge",
    "Avoid Overconfidence",
    "High Conviction",
    "Combination",
    "Counter-Trade"
]

for strategy in strategy_names:
    count = df_recommendations['strategies'].str.contains(strategy, na=False).sum()
    if count > 0:
        strategy_df = df_recommendations[df_recommendations['strategies'].str.contains(strategy, na=False)]
        avg_edge = strategy_df['edge'].mean()
        avg_ev_qk = strategy_df['expected_value_quarter_kelly'].mean()
        print(f"{strategy:25s}: {count:2d} markets | Avg Edge: {avg_edge:+.1%} | Avg EV (Q-K): ${avg_ev_qk:+.2f}")

if len(df_recommendations) > 0:
    print(f"\n{'='*80}")
    print("TOP 10 RECOMMENDATIONS (BY EDGE)")
    print(f"{'='*80}\n")
    
    top_10 = df_recommendations.head(10)
    
    for idx, row in top_10.iterrows():
        direction = "YES" if row['edge'] > 0 else "NO"
        print(f"{row['ticker']:6s} | {direction:3s} | Edge: {row['edge']:+.1%} | "
              f"Quarter-Kelly: ${row['position_quarter_kelly_$100']:.2f} | "
              f"Full-Kelly: ${row['position_full_kelly_$100']:.2f}")
        print(f"       Model: {row['model_prob_lr']:.1%} | Market: {row['market_price']:.1%} | "
              f"EV (Q-K): ${row['expected_value_quarter_kelly']:.2f} | "
              f"EV (F-K): ${row['expected_value_full_kelly']:.2f}")
        print(f"       Strategies: {row['strategies']}")
        print(f"       {row['question'][:70]}...")
        print()

# Sweet Spot recommendations
sweet_spot = df_recommendations[df_recommendations['strategies'].str.contains('Sweet Spot', na=False)]
if len(sweet_spot) > 0:
    print(f"\n{'='*80}")
    print(f"🎯 SWEET SPOT OPPORTUNITIES ({len(sweet_spot)} markets)")
    print(f"{'='*80}")
    print("Historical Performance: +32.73% ROI, 93.75% win rate\n")
    
    for idx, row in sweet_spot.iterrows():
        print(f"{row['ticker']:6s} | Edge: {row['edge']:+.1%} | "
              f"Model: {row['model_prob_lr']:.1%} | Market: {row['market_price']:.1%}")
        print(f"       Quarter-Kelly: ${row['position_quarter_kelly_$100']:.2f} (EV: ${row['expected_value_quarter_kelly']:.2f})")
        print()

print(f"\n✅ Full results saved to: {output_path}")
print("\n" + "=" * 80)

