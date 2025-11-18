#!/usr/bin/env python3
"""
Quarterly Beat/Miss Analysis
=============================

Tüm S&P 500 şirketlerinin çeyrek bazlı beat/miss istatistiklerini çıkarır.

Output:
    excel_data/quarterly_beat_miss.csv
    
Columns:
    - date (quarter end date)
    - total_companies (o çeyrekte rapor veren şirket sayısı)
    - beat_count (beat eden şirket sayısı)
    - miss_count (miss eden şirket sayısı)
    - beat_rate (beat / total)
    - avg_eps_delta (ortalama eps_delta)
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

# Config
DATA_GLOB = "../data/raw/*_earnings_with_q4.csv"
OUTPUT_FILE = "excel_data/quarterly_beat_miss.csv"

print("=" * 80)
print("QUARTERLY BEAT/MISS ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD ALL DATA
# =============================================================================

print(f"\n📊 Veri yükleniyor...")

csv_files = glob.glob(DATA_GLOB)
print(f"   • {len(csv_files)} CSV dosyası bulundu")

dfs = []
for fp in csv_files:
    filename = os.path.basename(fp)
    symbol = filename.replace("_earnings_with_q4.csv", "")
    
    try:
        df = pd.read_csv(fp, low_memory=False)
        df['symbol'] = symbol
        dfs.append(df)
    except Exception as e:
        continue

full = pd.concat(dfs, ignore_index=True)
print(f"   • {len(full):,} toplam satır")
print(f"   • {full['symbol'].nunique()} şirket")

# =============================================================================
# 2. FILTER QUARTERLY DATA
# =============================================================================

print(f"\n🔍 Quarterly verileri filtreleniyor...")

# Sadece quarterly data (fiscal year değil)
full['is_quarterly'] = full['horizon'].str.contains('quarter|Q[1-4]', case=False, na=False)
quarterly = full[full['is_quarterly']].copy()

print(f"   • {len(quarterly):,} quarterly satır")

# Date parse
quarterly['date'] = pd.to_datetime(quarterly['date'], errors='coerce')
quarterly = quarterly.dropna(subset=['date'])

# eps_beat kontrolü
quarterly = quarterly[quarterly['eps_beat'].notna()].copy()
quarterly['eps_beat'] = quarterly['eps_beat'].astype(int)

print(f"   • {len(quarterly):,} satır (beat data ile)")

# =============================================================================
# 3. GROUP BY QUARTER
# =============================================================================

print(f"\n📈 Çeyrek bazlı gruplama...")

# Her çeyrek için istatistikler
grouped = quarterly.groupby('date').agg({
    'symbol': 'count',                    # toplam şirket
    'eps_beat': ['sum', 'mean'],         # beat count, beat rate
    'eps_delta': 'mean'                   # ortalama eps delta
}).reset_index()

# Column isimleri düzelt
grouped.columns = ['date', 'total_companies', 'beat_count', 'beat_rate', 'avg_eps_delta']

# Miss count hesapla
grouped['miss_count'] = grouped['total_companies'] - grouped['beat_count']

# Sırala (eskiden yeniye)
grouped = grouped.sort_values('date')

# Beat rate'i yüzdeye çevir
grouped['beat_rate_pct'] = (grouped['beat_rate'] * 100).round(2)

print(f"   • {len(grouped)} çeyrek bulundu")
print(f"   • Tarih aralığı: {grouped['date'].min()} - {grouped['date'].max()}")

# =============================================================================
# 4. ADD HISTORICAL vs FUTURE FLAG
# =============================================================================

print(f"\n📅 Geçmiş/gelecek flag'i ekleniyor...")

today = pd.Timestamp.now()
grouped['status'] = grouped['date'].apply(lambda x: 'Historical' if x < today else 'Future')

historical_count = (grouped['status'] == 'Historical').sum()
future_count = (grouped['status'] == 'Future').sum()

print(f"   • {historical_count} geçmiş çeyrek")
print(f"   • {future_count} gelecek çeyrek")

# =============================================================================
# 5. EXPORT CSV
# =============================================================================

print(f"\n💾 CSV kaydediliyor...")

# Final columns order
output_cols = [
    'date',
    'status',
    'total_companies',
    'beat_count',
    'miss_count',
    'beat_rate_pct',
    'avg_eps_delta'
]

output_df = grouped[output_cols].copy()

# Date formatı
output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')

# Save
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Kaydedildi: {OUTPUT_FILE}")

# =============================================================================
# 6. SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Özet İstatistikler:")
print(f"{'='*60}")

# Geçmiş veriler için
historical = output_df[output_df['status'] == 'Historical'].copy()

if len(historical) > 0:
    print(f"\n🕐 GEÇMIŞ ÇEYREKLER ({len(historical)} çeyrek):")
    print(f"   • Ortalama beat rate: {historical['beat_rate_pct'].mean():.2f}%")
    print(f"   • En yüksek beat rate: {historical['beat_rate_pct'].max():.2f}% ({historical.loc[historical['beat_rate_pct'].idxmax(), 'date']})")
    print(f"   • En düşük beat rate: {historical['beat_rate_pct'].min():.2f}% ({historical.loc[historical['beat_rate_pct'].idxmin(), 'date']})")
    print(f"   • Toplam beat: {historical['beat_count'].sum():,.0f}")
    print(f"   • Toplam miss: {historical['miss_count'].sum():,.0f}")
    print(f"   • Ortalama EPS delta: ${historical['avg_eps_delta'].mean():.4f}")

# Son 5 çeyrek
print(f"\n📅 SON 5 ÇEYREK:")
print("="*60)
last_5 = output_df.tail(5)[['date', 'status', 'total_companies', 'beat_count', 'miss_count', 'beat_rate_pct']]
print(last_5.to_string(index=False))

# İlk 5 çeyrek
print(f"\n📅 İLK 5 ÇEYREK:")
print("="*60)
first_5 = output_df.head(5)[['date', 'status', 'total_companies', 'beat_count', 'miss_count', 'beat_rate_pct']]
print(first_5.to_string(index=False))

print(f"\n{'='*80}")
print(f"✅ TAMAMLANDI!")
print(f"   Output: {OUTPUT_FILE}")
print(f"{'='*80}")

