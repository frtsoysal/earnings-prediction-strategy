#!/usr/bin/env python3
"""
Polymarket Earnings Data Fetcher
=================================

Polymarket API'den earnings bet market'lerini çeker ve CSV'ye kaydeder.
TypeScript earningsBacktest.ts kodundan uyarlanmıştır.

API Endpoints:
- Gamma API: https://gamma-api.polymarket.com/markets

Output:
    polymarket_earnings_raw.csv
    
Columns:
    - market_id
    - ticker (extracted from title)
    - title
    - created_at
    - closed_at
    - is_closed
    - current_yes_price (0-1 scale)
    - final_outcome (Yes/No)
    - status (won/lost/open/no_data)
"""

import urllib.request
import urllib.parse
import json
import pandas as pd
import re
from datetime import datetime, timedelta
import time
import ssl

# SSL verification bypass
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 80)
print("POLYMARKET EARNINGS DATA FETCHER")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
EARNINGS_TAG_ID = "1013"  # Earnings tag
DAYS_BACK = 365  # 1 yıl geriye
INCLUDE_OPEN = True  # Açık market'leri de dahil et
LIMIT = 100

# =============================================================================
# 1. FETCH EARNINGS MARKETS FROM GAMMA API
# =============================================================================

print(f"\n📊 1/3 Fetching earnings markets (last {DAYS_BACK} days)...")

# Date cutoff
cutoff_date = datetime.now() - timedelta(days=DAYS_BACK + 2)
print(f"   • Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"   • Include open markets: {INCLUDE_OPEN}")

all_markets = []
offset = 0
has_more = True

while has_more:
    try:
        # Gamma API request
        url = f"{GAMMA_API_BASE}/markets"
        params = {
            'limit': LIMIT,
            'offset': offset,
            'tag_id': EARNINGS_TAG_ID,
            'order': 'id',
            'ascending': 'false'
        }
        
        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"
        
        # Add headers (User-Agent required by API)
        req = urllib.request.Request(full_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        
        markets = data if isinstance(data, list) else data.get('data', [])
        
        if len(markets) == 0:
            has_more = False
            break
        
        # Filter earnings beat markets
        for market in markets:
            # Text filter: "beat" and "earnings"
            question = str(market.get('question', '') or '')
            title = str(market.get('title', '') or '')
            text = (question + ' ' + title).lower()
            
            has_earnings = 'earnings' in text or 'earning' in text
            has_beat = 'beat' in text
            
            # Debug: Print first few that pass/fail
            if offset == 0 and len(all_markets) < 3:
                pass_fail = "✓" if (has_earnings and has_beat) else "✗"
                print(f"      {pass_fail} {title[:60]}")
            
            if not (has_earnings and has_beat):
                continue
            
            # Date filter
            created_at = market.get('createdAt', '')
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if created_dt < cutoff_date:
                        continue
                except:
                    continue
            
            # Closed/Open filter
            is_closed = market.get('closed', False)
            if not INCLUDE_OPEN and not is_closed:
                continue
            
            all_markets.append(market)
        
        # Log progress
        if offset == 0:
            closed_count = sum(1 for m in all_markets if m.get('closed'))
            open_count = len(all_markets) - closed_count
            print(f"   • First batch: {len(markets)} API results")
            print(f"   • After filters: {len(all_markets)} earnings beat markets")
            print(f"   • Closed: {closed_count}, Open: {open_count}")
        
        # Check oldest market
        if len(markets) > 0:
            oldest = markets[-1]
            oldest_created = oldest.get('createdAt', '')
            if oldest_created:
                try:
                    oldest_dt = datetime.fromisoformat(oldest_created.replace('Z', '+00:00'))
                    if oldest_dt < cutoff_date:
                        print(f"   • Stopping: Reached cutoff date")
                        has_more = False
                        break
                except:
                    pass
        
        offset += LIMIT
        
        # Safety limit
        if offset >= 5000:
            print(f"   ⚠️  Safety limit reached ({offset} checked)")
            has_more = False
        
        # Rate limiting
        time.sleep(0.3)
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        has_more = False

closed_count = sum(1 for m in all_markets if m.get('closed'))
open_count = len(all_markets) - closed_count

print(f"\n   ✅ Found {len(all_markets)} earnings beat markets")
print(f"      • Closed: {closed_count}")
print(f"      • Open: {open_count}")

# =============================================================================
# 2. EXTRACT TICKER FROM TITLE
# =============================================================================

print(f"\n🔍 2/3 Extracting tickers from market titles...")

def extract_ticker(title):
    """
    Extract stock ticker from market title
    Examples:
    - "Will Apple (AAPL) beat Q3 2024 earnings?" -> AAPL
    - "Tesla (TSLA) earnings beat" -> TSLA
    - "MSFT Q4 earnings beat expectations?" -> MSFT
    """
    # Pattern 1: (TICKER) format
    match = re.search(r'\(([A-Z]{1,5})\)', title)
    if match:
        return match.group(1)
    
    # Pattern 2: $TICKER format
    match = re.search(r'\$([A-Z]{1,5})', title)
    if match:
        return match.group(1)
    
    # Pattern 3: Standalone ticker
    words = title.upper().split()
    for word in words:
        word_clean = re.sub(r'[^A-Z]', '', word)
        if 2 <= len(word_clean) <= 5:
            return word_clean
    
    return 'UNKNOWN'

ticker_counts = {}
for market in all_markets:
    title = market.get('title') or market.get('question', '')
    ticker = extract_ticker(title)
    market['ticker'] = ticker
    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

print(f"   • Extracted {len(ticker_counts)} unique tickers")
print(f"   • Top 10 tickers:")
for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"      {ticker:8s}: {count} markets")

# =============================================================================
# 3. PROCESS MARKET OUTCOMES
# =============================================================================

print(f"\n💰 3/3 Processing market outcomes...")

results = []

for market in all_markets:
    market_id = market.get('id', 'unknown')
    title = market.get('title') or market.get('question', 'Unknown')
    ticker = market.get('ticker', 'UNKNOWN')
    created_at = market.get('createdAt', '')
    closed_at = market.get('closedTime', '')
    is_closed = market.get('closed', False)
    
    # Get current Yes price
    current_yes_price = None
    outcome_prices = market.get('outcomePrices')
    
    if outcome_prices:
        if isinstance(outcome_prices, str):
            try:
                prices_array = json.loads(outcome_prices)
            except:
                prices_array = []
        elif isinstance(outcome_prices, list):
            prices_array = outcome_prices
        else:
            prices_array = []
        
        if len(prices_array) > 0:
            current_yes_price = float(prices_array[0])
    
    # Determine outcome
    final_outcome = None
    status = 'open'
    
    if current_yes_price is not None:
        if is_closed:
            final_outcome = 'Yes' if current_yes_price > 0.50 else 'No'
            status = 'won' if final_outcome == 'Yes' else 'lost'
        else:
            status = 'open'
    else:
        status = 'no_data'
    
    results.append({
        'market_id': market_id,
        'ticker': ticker,
        'title': title,
        'created_at': created_at,
        'closed_at': closed_at if closed_at else '',
        'is_closed': is_closed,
        'current_yes_price': current_yes_price if current_yes_price is not None else '',
        'final_outcome': final_outcome if final_outcome else '',
        'status': status
    })

print(f"   ✅ Processed {len(results)} markets")

# =============================================================================
# 4. SAVE TO CSV
# =============================================================================

print(f"\n💾 Saving to CSV...")

df = pd.DataFrame(results)

# Sort by created_at (newest first) if data exists
if len(df) > 0 and 'created_at' in df.columns:
    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.sort_values('created_dt', ascending=False)
    df = df.drop('created_dt', axis=1)

csv_file = 'polymarket_earnings_raw.csv'
df.to_csv(csv_file, index=False)

print(f"   ✓ Saved: {csv_file}")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Summary Statistics:")
print("=" * 80)

print(f"\nTotal Markets: {len(df)}")

if len(df) > 0:
    print(f"\nMarket Status:")
    closed_count = (df['is_closed'] == True).sum()
    open_count = (df['is_closed'] == False).sum()
    print(f"   • Closed (resolved): {closed_count}")
    print(f"   • Open (pending):    {open_count}")

    print(f"\nOutcomes (for closed markets only):")
    closed_df = df[df['is_closed'] == True]
    if len(closed_df) > 0:
        yes_count = (closed_df['final_outcome'] == 'Yes').sum()
        no_count = (closed_df['final_outcome'] == 'No').sum()
        print(f"   • Yes (company beat): {yes_count}")
        print(f"   • No (company miss):  {no_count}")
        yes_rate = yes_count / len(closed_df) * 100 if len(closed_df) > 0 else 0
        print(f"   • Polymarket beat rate: {yes_rate:.1f}%")

    print(f"\nTop 10 Tickers:")
    ticker_dist = df[df['ticker'] != 'UNKNOWN']['ticker'].value_counts().head(10)
    for ticker, count in ticker_dist.items():
        print(f"   • {ticker:6s}: {count} markets")

    print(f"\nDate Range:")
    if 'created_at' in df.columns and len(df) > 0:
        first_date = df['created_at'].iloc[0]
        last_date = df['created_at'].iloc[-1]
        if first_date:
            print(f"   • Newest: {first_date[:10]}")
        if last_date:
            print(f"   • Oldest: {last_date[:10]}")
else:
    print("\n⚠️  No markets found! Check filters or API.")

print("\n" + "=" * 80)
print("✅ POLYMARKET DATA FETCH COMPLETE")
print("=" * 80)
print(f"\nOutput: {csv_file}")
print(f"Rows: {len(df)}")
print("=" * 80)

