#!/usr/bin/env python3
"""
ML Pipeline Runner - Complete Analysis for a Stock Symbol
Runs: Fetch → Prepare → Train → Evaluate

Usage: python3 run_pipeline.py --symbol IBM
"""

import argparse
import subprocess
import sys
import os

def run_command(cmd, desc):
    """Run command and display results"""
    print(f"\n{'='*80}")
    print(f"  {desc}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"\n❌ ERROR in {desc}")
        return False
    return True

def run_full_pipeline(symbol, skip_fetch=False):
    """Execute complete ML pipeline for given symbol"""
    
    symbol = symbol.upper()
    
    print(f"\n{'#'*80}")
    print(f"#  ML PIPELINE FOR {symbol}")
    print(f"{'#'*80}\n")
    
    # Step 1: Fetch data (optional)
    if not skip_fetch:
        print(f"📥 STEP 1/5: Fetching data for {symbol}...")
    if not run_command(
        f"python3 scripts/with_estimates/fetch_alpha_vantage.py --symbol {symbol}",
        f"Fetching earnings data for {symbol}"
    ):
        sys.exit(1)
    else:
        print(f"📥 STEP 1/5: Skipping fetch (using existing data)...")
    
    # Step 2: Prepare data
    if not run_command(
        f"python3 scripts/with_estimates/prepare_data.py --symbol {symbol}",
        f"STEP 2/4: Preparing data for {symbol}"
    ):
        sys.exit(1)
    
    # Step 3: Train models
    if not run_command(
        f"python3 scripts/with_estimates/train_model.py --symbol {symbol}",
        f"STEP 3/4: Training models for {symbol}"
    ):
        sys.exit(1)
    
    # Step 4: Train with Time Series CV
    if not run_command(
        f"python3 scripts/with_estimates/train_model_timeseries.py --symbol {symbol}",
        f"STEP 4/5: Training with Time Series CV for {symbol}"
    ):
        sys.exit(1)
    
    # Step 5: Evaluate
    if not run_command(
        f"python3 scripts/with_estimates/evaluate.py --symbol {symbol}",
        f"STEP 5/5: Evaluating models for {symbol}"
    ):
        sys.exit(1)
    
    # Success summary
    print(f"\n{'='*80}")
    print(f"✅ PIPELINE COMPLETE FOR {symbol}!")
    print(f"{'='*80}")
    print(f"\n📊 Results:")
    print(f"   • Raw data:       data/raw/{symbol}_earnings_with_q4.csv")
    print(f"   • Processed data: data/processed/{symbol}/")
    print(f"   • Models:         models/{symbol}/")
    print(f"   • Reports:        results/{symbol}/")
    print(f"\n📈 View reports:")
    print(f"   cat results/{symbol}/evaluation_report.txt")
    print(f"   cat results/{symbol}/model_comparison.csv")
    print(f"   open results/{symbol}/  # View visualizations\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete ML pipeline for EPS beat prediction'
    )
    parser.add_argument(
        '--symbol', 
        required=True, 
        help='Stock symbol (e.g., IBM, TSLA, AAPL)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching data (use existing CSV)'
    )
    args = parser.parse_args()
    
    run_full_pipeline(args.symbol, skip_fetch=args.skip_fetch)

