#!/usr/bin/env python3
"""
Fetch Missing Polymarket Tickers
=================================

missing_tickers.txt dosyasından tickerları okur ve fetch_alpha_vantage.py ile çeker.

Premium API: 75 calls/minute
"""

import subprocess
import time
from datetime import datetime
import re

# Config
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your Alpha Vantage API key
CALLS_PER_MINUTE = 75
DELAY = 60.0 / CALLS_PER_MINUTE  # 0.8 saniye

print("=" * 80)
print("FETCHING MISSING POLYMARKET TICKERS")
print("=" * 80)

# Read missing tickers
tickers = []
with open('missing_tickers.txt', 'r') as f:
    for line in f:
        # Parse: "AAPL     - 1 Polymarket events"
        match = re.match(r'([A-Z]+)\s+-', line)
        if match:
            tickers.append(match.group(1))

print(f"\n📊 Found {len(tickers)} missing tickers")
print(f"⏱️  Estimated time: {len(tickers) * DELAY / 60:.1f} minutes")
print(f"\nFirst 10: {tickers[:10]}")

# Fetch each ticker
success = 0
failed = 0
failed_tickers = []
start = datetime.now()

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] {ticker:8s}... ", end='', flush=True)
    
    try:
        result = subprocess.run(
            ['python3', '../../scripts/with_estimates/fetch_alpha_vantage.py',
             '--symbol', ticker, '--api-key', API_KEY],
            capture_output=True,
            timeout=180
        )
        
        if result.returncode == 0:
            print("✓")
            success += 1
        else:
            print("✗")
            failed += 1
            failed_tickers.append(ticker)
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        failed += 1
        failed_tickers.append(ticker)
    
    # Progress every 25
    if (i+1) % 25 == 0:
        elapsed = (datetime.now() - start).total_seconds() / 60
        remaining = (len(tickers) - i - 1) * DELAY / 60
        print(f"\n📊 Progress: {success} ✓, {failed} ✗ | Elapsed: {elapsed:.1f}min, Remaining: {remaining:.1f}min\n")
    
    time.sleep(DELAY)

total_time = (datetime.now() - start).total_seconds() / 60

print(f"\n{'='*80}")
print(f"✅ FETCH COMPLETE!")
print(f"   ⏱️  Total time: {total_time:.1f} minutes")
print(f"   ✅ Success: {success}")
print(f"   ❌ Failed: {failed}")
print(f"   📊 Success rate: {success/len(tickers)*100:.1f}%")

if failed_tickers:
    print(f"\n❌ Failed tickers ({len(failed_tickers)}):")
    for ticker in failed_tickers[:20]:
        print(f"   - {ticker}")
    if len(failed_tickers) > 20:
        print(f"   ... and {len(failed_tickers)-20} more")

print(f"{'='*80}")

