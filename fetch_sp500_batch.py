#!/usr/bin/env python3
"""
S&P 500 Batch Data Fetcher
Tüm S&P 500 tickerları için Alpha Vantage'den veri çeker.
Premium API: 75 calls/minute limit
"""

import subprocess
import time
import csv
from datetime import datetime
import os

# Config
API_KEY = 'NI9UUKNAPMG66IY5'
TICKERS_CSV = 'sp500_tickers.csv'
CALLS_PER_MINUTE = 75  # Premium tier limit
DELAY_BETWEEN_CALLS = 60.0 / CALLS_PER_MINUTE  # 0.8 seconds
START_FROM_INDEX = 0  # Kaldığı yerden devam etmek için değiştir

# Log dosyası
LOG_FILE = f'fetch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def read_tickers():
    """CSV'den ticker listesini oku"""
    tickers = []
    with open(TICKERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append({
                'ticker': row['ticker'],
                'company': row['company_name']
            })
    return tickers

def fetch_ticker(ticker, api_key):
    """Bir ticker için veri çek"""
    try:
        result = subprocess.run(
            ['python3', 'scripts/with_estimates/fetch_alpha_vantage.py', 
             '--symbol', ticker, '--api-key', api_key],
            capture_output=True,
            text=True,
            timeout=180  # 3 dakika timeout
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 90)
    print("📊 S&P 500 TOPLU VERİ ÇEKME")
    print("=" * 90)
    
    # Ticker listesini oku
    print(f"\n1️⃣ Ticker listesi okunuyor: {TICKERS_CSV}")
    tickers = read_tickers()
    total_tickers = len(tickers)
    print(f"   ✓ {total_tickers} ticker bulundu")
    
    # Kaldığı yerden devam
    if START_FROM_INDEX > 0:
        print(f"   ⚠️  Index {START_FROM_INDEX}'dan başlanıyor")
        tickers = tickers[START_FROM_INDEX:]
    
    # Süre tahmini
    total_time_minutes = (len(tickers) * DELAY_BETWEEN_CALLS) / 60
    print(f"\n2️⃣ İşlem başlıyor...")
    print(f"   ⏱️  Tahmini süre: {total_time_minutes:.1f} dakika (~{total_time_minutes/60:.1f} saat)")
    print(f"   📝 Log dosyası: {LOG_FILE}")
    print(f"   🚀 Rate limit: {CALLS_PER_MINUTE} call/dakika ({DELAY_BETWEEN_CALLS:.2f}s delay)\n")
    
    # İşlemi başlat
    success_count = 0
    error_count = 0
    start_time = datetime.now()
    
    with open(LOG_FILE, 'w') as log:
        log.write(f"S&P 500 Batch Fetch Started: {start_time}\n")
        log.write(f"Total tickers: {total_tickers}\n")
        log.write(f"Start index: {START_FROM_INDEX}\n")
        log.write(f"Rate limit: {CALLS_PER_MINUTE} calls/minute\n")
        log.write("=" * 90 + "\n\n")
        
        for i, ticker_info in enumerate(tickers, start=START_FROM_INDEX):
            ticker = ticker_info['ticker']
            company = ticker_info['company']
            current_num = i + 1
            
            # Progress göster
            print(f"[{current_num}/{total_tickers}] {ticker:6s} - {company[:40]:40s} ", end='', flush=True)
            log.write(f"[{current_num}/{total_tickers}] {ticker} - {company}\n")
            log.write(f"  Time: {datetime.now()}\n")
            
            # Veriyi çek
            fetch_start = time.time()
            success, error_msg = fetch_ticker(ticker, API_KEY)
            fetch_duration = time.time() - fetch_start
            
            if success:
                print(f"✓ ({fetch_duration:.1f}s)")
                log.write(f"  Status: SUCCESS ({fetch_duration:.1f}s)\n")
                success_count += 1
            else:
                print(f"✗ ({error_msg[:30]})")
                log.write(f"  Status: FAILED\n")
                log.write(f"  Error: {error_msg}\n")
                error_count += 1
            
            log.write("\n")
            log.flush()
            
            # Rate limiting (son ticker hariç)
            if i < total_tickers - 1:
                time.sleep(DELAY_BETWEEN_CALLS)
            
            # Her 25 tickerda progress raporu
            if current_num % 25 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                remaining_tickers = total_tickers - current_num
                est_remaining = (remaining_tickers * DELAY_BETWEEN_CALLS) / 60
                
                print(f"\n{'='*90}")
                print(f"📊 İlerleme Raporu (Index {current_num})")
                print(f"   ✅ Başarılı: {success_count}")
                print(f"   ❌ Hata: {error_count}")
                print(f"   ⏱️  Geçen süre: {elapsed:.1f} dakika")
                print(f"   ⏳ Kalan tahmini: {est_remaining:.1f} dakika")
                print(f"{'='*90}\n")
                
                log.write(f"--- PROGRESS REPORT (Index {current_num}) ---\n")
                log.write(f"Success: {success_count}, Errors: {error_count}\n")
                log.write(f"Elapsed: {elapsed:.1f}min, Remaining: {est_remaining:.1f}min\n\n")
        
        # Final rapor
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        log.write("\n" + "=" * 90 + "\n")
        log.write(f"COMPLETED: {end_time}\n")
        log.write(f"Duration: {total_duration:.1f} minutes\n")
        log.write(f"Success: {success_count}\n")
        log.write(f"Errors: {error_count}\n")
        log.write(f"Success rate: {(success_count/total_tickers)*100:.1f}%\n")
    
    print("\n" + "=" * 90)
    print("✅ TAMAMLANDI!")
    print(f"   ⏱️  Toplam süre: {total_duration:.1f} dakika ({total_duration/60:.1f} saat)")
    print(f"   ✅ Başarılı: {success_count}")
    print(f"   ❌ Hata: {error_count}")
    print(f"   📊 Başarı oranı: {(success_count/total_tickers)*100:.1f}%")
    print(f"   📝 Log: {LOG_FILE}")
    print("=" * 90)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu!")
        print("   Kaldığı yerden devam etmek için START_FROM_INDEX değerini güncelle.")

