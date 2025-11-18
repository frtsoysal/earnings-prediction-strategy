#!/usr/bin/env python3
"""
S&P 500 tickerları için fetch_alpha_vantage.py'yi toplu çalıştırır
"""
import subprocess
import csv
import time
from datetime import datetime

# Config
API_KEY = 'R2JISTMI7V1RCKWT'  # Premium API key
CALLS_PER_MINUTE = 75
DELAY = 60.0 / CALLS_PER_MINUTE  # 0.8 saniye
START_FROM = 0  # Buradan devam et

# Tickerları oku
tickers = []
with open('sp500_tickers.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tickers.append(row['ticker'])

print(f"📊 {len(tickers)} ticker bulundu")
print(f"⏱️  Tahmini süre: {len(tickers) * DELAY / 60:.1f} dakika\n")

success = 0
failed = 0
start = datetime.now()

for i, ticker in enumerate(tickers[START_FROM:], START_FROM):
    print(f"[{i+1}/{len(tickers)}] {ticker}... ", end='', flush=True)
    
    try:
        result = subprocess.run(
            ['python3', 'scripts/with_estimates/fetch_alpha_vantage.py',
             '--symbol', ticker, '--api-key', API_KEY],
            capture_output=True,
            timeout=180
        )
        
        if result.returncode == 0:
            print("✓")
            success += 1
        else:
            print(f"✗ (code {result.returncode})")
            if i < 5:  # İlk 5 hatayı detaylı göster
                print(f"    Error: {result.stderr.decode()[:200]}")
            failed += 1
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        failed += 1
    
    # Progress her 25 tickerda
    if (i+1) % 25 == 0:
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"\n📊 Progress: {success} ✓, {failed} ✗ | Elapsed: {elapsed:.1f}min\n")
    
    time.sleep(DELAY)

print(f"\n✅ Done! {success} success, {failed} failed")
print(f"⏱️  Total: {(datetime.now() - start).total_seconds() / 60:.1f} minutes")

