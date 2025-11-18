#!/usr/bin/env python3
"""
Fundamentals-Only ML Pipeline Runner
Analyst estimates olmadan, sadece fundamental data ile çalışır.
1996+ historical data kullanır.
"""
import sys
import subprocess
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run fundamentals-only ML pipeline for a stock')
parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., IBM, TSLA)')
parser.add_argument('--api-key', required=True, help='Alpha Vantage API key')
args = parser.parse_args()

symbol = args.symbol.upper()
api_key = args.api_key

print(f"\n{'='*80}")
print(f"🚀 FUNDAMENTALS-ONLY ML PIPELINE: {symbol}")
print(f"{'='*80}\n")

# Step 1: Fetch fundamentals data
print(f"📊 Step 1/4: Fetching fundamentals data...")
result = subprocess.run([
    'python3', 'scripts/only_fundamentals/fetch.py',
    '--symbol', symbol,
    '--api-key', api_key
], capture_output=False)

if result.returncode != 0:
    print(f"\n❌ HATA: fetch.py başarısız oldu!")
    sys.exit(1)

# Step 2: Prepare data
print(f"\n📊 Step 2/4: Preparing data...")
result = subprocess.run([
    'python3', 'scripts/only_fundamentals/prepare_data.py',
    '--symbol', symbol
], capture_output=False)

if result.returncode != 0:
    print(f"\n❌ HATA: prepare_data.py başarısız oldu!")
    sys.exit(1)

# Step 3: Train models
print(f"\n📊 Step 3/4: Training models...")
result = subprocess.run([
    'python3', 'scripts/only_fundamentals/train_model.py',
    '--symbol', symbol
], capture_output=False)

if result.returncode != 0:
    print(f"\n❌ HATA: train_model.py başarısız oldu!")
    sys.exit(1)

# Step 4: Evaluate models
print(f"\n📊 Step 4/4: Evaluating models...")
result = subprocess.run([
    'python3', 'scripts/only_fundamentals/evaluate.py',
    '--symbol', symbol
], capture_output=False)

if result.returncode != 0:
    print(f"\n❌ HATA: evaluate_fundamentals.py başarısız oldu!")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"✅ FUNDAMENTALS-ONLY PIPELINE TAMAMLANDI: {symbol}")
print(f"{'='*80}\n")

