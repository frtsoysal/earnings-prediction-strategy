#!/usr/bin/env python3
"""
Yeni S&P 500 tickerları için veri çek
"""
import subprocess
import time
from datetime import datetime

# Config
API_KEY = 'R2JISTMI7V1RCKWT'
CALLS_PER_MINUTE = 75
DELAY = 60.0 / CALLS_PER_MINUTE  # 0.8 saniye

# Yeni tickerları oku
with open('new_tickers_to_fetch.txt', 'r') as f:
    tickers = [line.strip() for line in f if line.strip()]

print(f"📊 {len(tickers)} yeni ticker için veri çekiliyor...")
print(f"⏱️  Tahmini süre: {len(tickers) * DELAY / 60:.1f} dakika\n")

success = 0
failed = 0
failed_tickers = []
start = datetime.now()

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] {ticker:6s}... ", end='', flush=True)
    
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
            print("✗")
            failed += 1
            failed_tickers.append(ticker)
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        failed += 1
        failed_tickers.append(ticker)
    
    # Progress her 10 tickerda
    if (i+1) % 10 == 0:
        elapsed = (datetime.now() - start).total_seconds() / 60
        remaining = (len(tickers) - i - 1) * DELAY / 60
        print(f"\n📊 Progress: {success} ✓, {failed} ✗ | Elapsed: {elapsed:.1f}min, Remaining: {remaining:.1f}min\n")
    
    time.sleep(DELAY)

total_time = (datetime.now() - start).total_seconds() / 60

print(f"\n{'='*80}")
print(f"✅ TAMAMLANDI!")
print(f"   ⏱️  Toplam süre: {total_time:.1f} dakika")
print(f"   ✅ Başarılı: {success}")
print(f"   ❌ Hata: {failed}")
print(f"   📊 Başarı oranı: {(success/len(tickers))*100:.1f}%")

if failed_tickers:
    print(f"\n❌ Başarısız olan tickerlar:")
    for ticker in failed_tickers:
        print(f"   - {ticker}")
print(f"{'='*80}")

