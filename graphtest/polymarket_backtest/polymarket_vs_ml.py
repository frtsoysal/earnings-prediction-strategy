#!/usr/bin/env python3
"""
Polymarket vs ML Predictions - Backtest Analysis
=================================================

Polymarket earnings bet outcomes'larını global ML model predictions ile karşılaştırır.

Workflow:
1. Polymarket CSV'yi oku
2. Her ticker için raw data var mı kontrol et
3. Olmayan tickerları listele (fetch edilmesi için)
4. Event → Quarter matching
5. Features extract et
6. 3 model ile predict (RF, XGB, LR)
7. Polymarket outcome ile karşılaştır
8. Comprehensive analysis

Output:
    backtest_results.csv
    backtest_analysis.txt
    model_comparison.png
    missing_tickers.txt
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from datetime import datetime, timedelta

print("=" * 80)
print("POLYMARKET VS ML PREDICTIONS - BACKTEST ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD POLYMARKET DATA
# =============================================================================

print(f"\n📊 1/6 Loading Polymarket data...")

polymarket_csv = '/Users/ibrahimfiratsoysal/Documents/earnings_history_1year.csv'
poly_df = pd.read_csv(polymarket_csv)

print(f"   • Total Polymarket events: {len(poly_df)}")
print(f"   • Unique tickers: {poly_df['ticker'].nunique()}")
print(f"   • Closed events: {(poly_df['beat'].notna()).sum()}")

# =============================================================================
# 2. CHECK WHICH TICKERS HAVE RAW DATA
# =============================================================================

print(f"\n🔍 2/6 Checking which tickers have raw data...")

# Get all available tickers from raw data
raw_data_dir = '../../data/raw'
available_csvs = glob.glob(f'{raw_data_dir}/*_earnings_with_q4.csv')
available_tickers = set()

for fp in available_csvs:
    filename = os.path.basename(fp)
    ticker = filename.replace('_earnings_with_q4.csv', '')
    # Check if file has data (>1KB)
    if os.path.getsize(fp) > 1000:
        available_tickers.add(ticker)

print(f"   • Available tickers in raw data: {len(available_tickers)}")

# Check Polymarket tickers
poly_tickers = set(poly_df['ticker'].unique())
tickers_with_data = poly_tickers & available_tickers
tickers_missing = poly_tickers - available_tickers

print(f"   • Polymarket tickers with data: {len(tickers_with_data)}")
print(f"   • Polymarket tickers missing: {len(tickers_missing)}")

if tickers_missing:
    print(f"\n   ⚠️  Missing tickers (top 10): {sorted(list(tickers_missing))[:10]}")
    
    # Save missing tickers list
    with open('missing_tickers.txt', 'w') as f:
        f.write("MISSING TICKERS - Need to fetch data\n")
        f.write("=" * 80 + "\n\n")
        for ticker in sorted(tickers_missing):
            count = (poly_df['ticker'] == ticker).sum()
            f.write(f"{ticker:8s} - {count} Polymarket events\n")
    
    print(f"   ✓ Saved: missing_tickers.txt ({len(tickers_missing)} tickers)")

# Filter: Only events where we have raw data
poly_with_data = poly_df[poly_df['ticker'].isin(tickers_with_data)].copy()
print(f"\n   • Polymarket events with available data: {len(poly_with_data)}")

# =============================================================================
# 3. MATCH POLYMARKET EVENTS TO QUARTERS
# =============================================================================

print(f"\n📅 3/6 Matching Polymarket events to quarters...")

def match_to_quarter(closed_date_str):
    """
    Match Polymarket close date to fiscal quarter.
    
    Earnings reports typically happen:
    - Q1 (Mar 31): Reports in April-May
    - Q2 (Jun 30): Reports in July-Aug
    - Q3 (Sep 30): Reports in Oct-Nov
    - Q4 (Dec 31): Reports in Jan-Feb (next year)
    
    Args:
        closed_date_str: "2025-11-13 19:52:42+00"
    
    Returns:
        Quarter end date: "2025-09-30"
    """
    try:
        # Parse date
        closed_dt = pd.to_datetime(closed_date_str)
        year = closed_dt.year
        month = closed_dt.month
        
        # Determine which quarter was being reported
        if month in [1, 2]:  # Jan-Feb → Previous year Q4
            return f"{year-1}-12-31"
        elif month in [3, 4, 5]:  # Mar-May → Q1
            return f"{year}-03-31"
        elif month in [6, 7, 8]:  # Jun-Aug → Q2
            return f"{year}-06-30"
        elif month in [9, 10, 11]:  # Sep-Nov → Q3
            return f"{year}-09-30"
        else:  # December → Q4
            return f"{year}-12-31"
    except:
        return None

poly_with_data['matched_quarter'] = poly_with_data['closedAt'].apply(match_to_quarter)

# Remove events without quarter match
poly_matched = poly_with_data[poly_with_data['matched_quarter'].notna()].copy()

print(f"   • Successfully matched: {len(poly_matched)} events")

# Show quarter distribution
quarter_dist = poly_matched['matched_quarter'].value_counts().head(10)
print(f"\n   Top quarters:")
for quarter, count in quarter_dist.items():
    print(f"      {quarter}: {count} events")

# =============================================================================
# 4. LOAD GLOBAL MODELS
# =============================================================================

print(f"\n🤖 4/6 Loading global models...")

rf_model = joblib.load('../global_model/models/global_rf_model.pkl')
xgb_model = joblib.load('../global_model/models/global_xgb_model.pkl')
lr_model = joblib.load('../global_model/models/global_lr_model.pkl')
preprocessor = joblib.load('../global_model/models/global_preprocessor.pkl')

print(f"   ✓ Loaded Random Forest")
print(f"   ✓ Loaded XGBoost")
print(f"   ✓ Loaded Logistic Regression")
print(f"   ✓ Loaded Preprocessor")

# =============================================================================
# 5. EXTRACT FEATURES & MAKE PREDICTIONS
# =============================================================================

print(f"\n🎯 5/6 Extracting features and making predictions...")

# Feature list (same as global model training)
SAFE_FEATURES = [
    'price_1m_before', 'price_3m_before',
    'eps_estimate_average', 'eps_estimate_high', 'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago', 'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago', 'eps_estimate_average_90_days_ago',
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    'revenue_estimate_average', 'revenue_estimate_high',
    'revenue_estimate_low', 'revenue_estimate_analyst_count',
    'elo_before', 'elo_decay', 'elo_momentum', 'elo_vol_4q',
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

backtest_results = []
feature_extraction_errors = 0

for idx, row in poly_matched.iterrows():
    ticker = row['ticker']
    quarter = row['matched_quarter']
    poly_outcome = row['beat']  # 1=beat, 0=miss
    
    # Load raw data for this ticker
    raw_csv = f"{raw_data_dir}/{ticker}_earnings_with_q4.csv"
    
    try:
        ticker_df = pd.read_csv(raw_csv)
        
        # Find the specific quarter row
        quarter_row = ticker_df[ticker_df['date'] == quarter]
        
        if len(quarter_row) == 0:
            # Quarter not found in data
            feature_extraction_errors += 1
            continue
        
        quarter_row = quarter_row.iloc[0]
        
        # Extract features
        features = {}
        for feat in SAFE_FEATURES:
            if feat in quarter_row.index:
                val = quarter_row[feat]
                # Handle NaN
                if pd.isna(val):
                    if 'revision' in feat:
                        features[feat] = 0  # No revision = 0
                    else:
                        features[feat] = np.nan
                else:
                    features[feat] = val
            else:
                features[feat] = np.nan
        
        # Convert to DataFrame for preprocessing
        X_single = pd.DataFrame([features])
        
        # Preprocess
        X_processed = preprocessor.transform(X_single)
        
        # Predict with all 3 models
        rf_pred = rf_model.predict(X_processed)[0]
        rf_proba = rf_model.predict_proba(X_processed)[0][1]
        
        xgb_pred = xgb_model.predict(X_processed)[0]
        xgb_proba = xgb_model.predict_proba(X_processed)[0][1]
        
        lr_pred = lr_model.predict(X_processed)[0]
        lr_proba = lr_model.predict_proba(X_processed)[0][1]
        
        # Check if predictions match Polymarket outcome
        backtest_results.append({
            'market_id': row['marketId'],
            'ticker': ticker,
            'quarter': quarter,
            'created_at': row['createdAt'],
            'closed_at': row['closedAt'],
            'polymarket_outcome': poly_outcome,
            'rf_prediction': rf_pred,
            'rf_probability': rf_proba,
            'rf_correct': (rf_pred == poly_outcome),
            'xgb_prediction': xgb_pred,
            'xgb_probability': xgb_proba,
            'xgb_correct': (xgb_pred == poly_outcome),
            'lr_prediction': lr_pred,
            'lr_probability': lr_proba,
            'lr_correct': (lr_pred == poly_outcome),
            'ensemble_vote': int((rf_pred + xgb_pred + lr_pred) >= 2),  # Majority vote
            'ensemble_correct': (int((rf_pred + xgb_pred + lr_pred) >= 2) == poly_outcome)
        })
        
    except Exception as e:
        feature_extraction_errors += 1
        continue
    
    if (idx + 1) % 50 == 0:
        print(f"   • Processed {idx + 1}/{len(poly_matched)}...")

print(f"\n   ✓ Successfully processed: {len(backtest_results)} events")
print(f"   • Feature extraction errors: {feature_extraction_errors}")

# =============================================================================
# 6. ANALYZE RESULTS
# =============================================================================

print(f"\n📊 6/6 Analyzing backtest results...")

if len(backtest_results) == 0:
    print("   ❌ No results to analyze!")
    exit(1)

results_df = pd.DataFrame(backtest_results)

# Save detailed results
results_df.to_csv('backtest_results.csv', index=False)
print(f"   ✓ Saved: backtest_results.csv")

# Calculate accuracy for each model
total_events = len(results_df)

rf_accuracy = results_df['rf_correct'].mean()
xgb_accuracy = results_df['xgb_correct'].mean()
lr_accuracy = results_df['lr_correct'].mean()
ensemble_accuracy = results_df['ensemble_correct'].mean()

print(f"\n   Model Accuracies (vs Polymarket outcomes):")
print(f"      Random Forest:        {rf_accuracy:.2%} ({results_df['rf_correct'].sum()}/{total_events})")
print(f"      XGBoost:              {xgb_accuracy:.2%} ({results_df['xgb_correct'].sum()}/{total_events})")
print(f"      Logistic Regression:  {lr_accuracy:.2%} ({results_df['lr_correct'].sum()}/{total_events})")
print(f"      Ensemble (Majority):  {ensemble_accuracy:.2%} ({results_df['ensemble_correct'].sum()}/{total_events})")

# Polymarket beat rate vs ML predictions
poly_beat_rate = results_df['polymarket_outcome'].mean()
rf_beat_rate = results_df['rf_prediction'].mean()
xgb_beat_rate = results_df['xgb_prediction'].mean()
lr_beat_rate = results_df['lr_prediction'].mean()

print(f"\n   Beat Rate Comparison:")
print(f"      Polymarket Actual:    {poly_beat_rate:.2%}")
print(f"      RF Predicted:         {rf_beat_rate:.2%}")
print(f"      XGB Predicted:        {xgb_beat_rate:.2%}")
print(f"      LR Predicted:         {lr_beat_rate:.2%}")

# =============================================================================
# 7. GENERATE COMPREHENSIVE REPORT
# =============================================================================

print(f"\n📄 Generating comprehensive analysis report...")

with open('backtest_analysis.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("POLYMARKET VS ML PREDICTIONS - BACKTEST ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Polymarket CSV: {polymarket_csv}\n")
    f.write(f"Total events: {len(poly_df)}\n")
    f.write(f"Events with raw data: {len(results_df)}\n")
    f.write(f"Unique tickers tested: {results_df['ticker'].nunique()}\n")
    f.write(f"Date range: {results_df['closed_at'].min()} to {results_df['closed_at'].max()}\n\n")
    
    f.write("MODEL ACCURACY (vs Polymarket Outcomes)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Random Forest:        {rf_accuracy:.2%} ({results_df['rf_correct'].sum()}/{total_events} correct)\n")
    f.write(f"XGBoost:              {xgb_accuracy:.2%} ({results_df['xgb_correct'].sum()}/{total_events} correct)\n")
    f.write(f"Logistic Regression:  {lr_accuracy:.2%} ({results_df['lr_correct'].sum()}/{total_events} correct)\n")
    f.write(f"Ensemble (Majority):  {ensemble_accuracy:.2%} ({results_df['ensemble_correct'].sum()}/{total_events} correct)\n\n")
    
    f.write("BEAT RATE COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(f"Polymarket Actual:    {poly_beat_rate:.2%}\n")
    f.write(f"RF Predicted:         {rf_beat_rate:.2%}\n")
    f.write(f"XGB Predicted:        {xgb_beat_rate:.2%}\n")
    f.write(f"LR Predicted:         {lr_beat_rate:.2%}\n\n")
    
    # Confusion analysis for best model
    best_model_name = 'Random Forest' if rf_accuracy >= max(xgb_accuracy, lr_accuracy) else ('XGBoost' if xgb_accuracy >= lr_accuracy else 'Logistic Regression')
    
    if best_model_name == 'Random Forest':
        pred_col = 'rf_prediction'
    elif best_model_name == 'XGBoost':
        pred_col = 'xgb_prediction'
    else:
        pred_col = 'lr_prediction'
    
    # Confusion matrix
    tp = ((results_df['polymarket_outcome'] == 1) & (results_df[pred_col] == 1)).sum()
    tn = ((results_df['polymarket_outcome'] == 0) & (results_df[pred_col] == 0)).sum()
    fp = ((results_df['polymarket_outcome'] == 0) & (results_df[pred_col] == 1)).sum()
    fn = ((results_df['polymarket_outcome'] == 1) & (results_df[pred_col] == 0)).sum()
    
    f.write(f"CONFUSION MATRIX ({best_model_name})\n")
    f.write("-" * 80 + "\n")
    f.write(f"                Predicted Beat    Predicted Miss\n")
    f.write(f"Actual Beat     {tp:8d}          {fn:8d}\n")
    f.write(f"Actual Miss     {fp:8d}          {tn:8d}\n\n")
    
    f.write("QUARTER DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    quarter_summary = results_df.groupby('quarter').agg({
        'market_id': 'count',
        'rf_correct': 'mean',
        'polymarket_outcome': 'mean'
    }).sort_index(ascending=False)
    
    for quarter, row in quarter_summary.head(10).iterrows():
        f.write(f"{quarter}: {int(row['market_id'])} events, RF accuracy: {row['rf_correct']:.1%}, Beat rate: {row['polymarket_outcome']:.1%}\n")
    
    f.write("\n")
    
    f.write("TOP PERFORMING TICKERS (ML Predictions)\n")
    f.write("-" * 80 + "\n")
    ticker_perf = results_df.groupby('ticker').agg({
        'rf_correct': 'mean',
        'market_id': 'count'
    }).sort_values('rf_correct', ascending=False).head(10)
    
    for ticker, row in ticker_perf.iterrows():
        f.write(f"{ticker:6s}: {row['rf_correct']:.1%} accuracy ({int(row['market_id'])} events)\n")
    
    f.write("\n")
    
    f.write("BOTTOM PERFORMING TICKERS\n")
    f.write("-" * 80 + "\n")
    ticker_perf_bottom = results_df.groupby('ticker').agg({
        'rf_correct': 'mean',
        'market_id': 'count'
    }).sort_values('rf_correct').head(10)
    
    for ticker, row in ticker_perf_bottom.iterrows():
        f.write(f"{ticker:6s}: {row['rf_correct']:.1%} accuracy ({int(row['market_id'])} events)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF ANALYSIS\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ Saved: backtest_analysis.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "=" * 80)
print(f"✅ POLYMARKET BACKTEST COMPLETE")
print(f"=" * 80)

print(f"\nBacktest Summary:")
print(f"   • Polymarket events tested: {len(results_df)}")
print(f"   • Unique tickers: {results_df['ticker'].nunique()}")
print(f"   • Date range: {results_df['quarter'].min()} to {results_df['quarter'].max()}")

print(f"\nModel vs Polymarket Accuracy:")
print(f"   🥇 Best Model: {best_model_name} ({max(rf_accuracy, xgb_accuracy, lr_accuracy):.2%})")
print(f"   • Random Forest:        {rf_accuracy:.2%}")
print(f"   • XGBoost:              {xgb_accuracy:.2%}")
print(f"   • Logistic Regression:  {lr_accuracy:.2%}")
print(f"   • Ensemble (Majority):  {ensemble_accuracy:.2%}")

if tickers_missing:
    print(f"\n⚠️  Missing Data:")
    print(f"   • {len(tickers_missing)} tickers need to be fetched")
    print(f"   • See missing_tickers.txt for list")
    print(f"\n   To fetch missing data:")
    print(f"   python3 scripts/with_estimates/fetch_alpha_vantage.py --symbol TICKER")

print(f"\nOutputs:")
print(f"   • backtest_results.csv (detailed predictions)")
print(f"   • backtest_analysis.txt (comprehensive report)")
if tickers_missing:
    print(f"   • missing_tickers.txt ({len(tickers_missing)} tickers to fetch)")

print(f"\n" + "=" * 80)

