import urllib.request
import json
import csv
import ssl
from collections import defaultdict
import numpy as np
import argparse
import sys
import os

# Add parent directory to path for imports (go up 2 levels: with_estimates -> scripts -> ML)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_paths, ensure_directories

# SSL sertifika doğrulamasını devre dışı bırak (gerekirse)
ssl._create_default_https_context = ssl._create_unverified_context

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fetch earnings data from Alpha Vantage')
parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., IBM, TSLA)')
parser.add_argument('--api-key', default='R2JISTMI7V1RCKWT', help='Alpha Vantage API key')
args = parser.parse_args()

# Configuration
api_key = args.api_key
symbol = args.symbol.upper()
paths = ensure_directories(symbol)

# Fetch ESTIMATES data
print(f"📊 Veri çekiliyor: {symbol}...")
print("   1/3 Tahminler (Estimates)...")
url_estimates = f'https://www.alphavantage.co/query?function=EARNINGS_ESTIMATES&symbol={symbol}&apikey={api_key}'
with urllib.request.urlopen(url_estimates) as response:
    data_estimates = json.loads(response.read().decode())

estimates = data_estimates['estimates']

# Fetch ACTUAL EARNINGS data (+ Q4 estimates!)
print("   2/3 Gerçekleşen veriler & Q4 tahminleri (Actuals & Q4 Estimates)...")
url_earnings = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}'
with urllib.request.urlopen(url_earnings) as response:
    data_earnings = json.loads(response.read().decode())

# Create separate lookup dictionaries for annual and quarterly actual EPS
annual_eps_lookup = {}
quarterly_eps_lookup = {}
q4_estimate_lookup = {}  # Q4 estimates from EARNINGS API
quarterly_reported_dates = {}  # Reported dates for quarters

# Add annual earnings
for annual in data_earnings.get('annualEarnings', []):
    date = annual['fiscalDateEnding']
    annual_eps_lookup[date] = annual['reportedEPS']

# Add quarterly earnings (both actual and estimate + reported date)
for quarterly in data_earnings.get('quarterlyEarnings', []):
    date = quarterly['fiscalDateEnding']
    quarterly_eps_lookup[date] = quarterly['reportedEPS']
    
    # Store reported date (announcement date)
    quarterly_reported_dates[date] = quarterly.get('reportedDate', '')
    
    # Store Q4 estimates (December quarters) separately
    if date.endswith('-12-31') and quarterly.get('estimatedEPS'):
        q4_estimate_lookup[date] = quarterly['estimatedEPS']

# Fetch INCOME STATEMENT data (for margins, revenue, EBITDA)
print("   3/5 Gelir tablosu verileri (Income Statement)...")
url_income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
with urllib.request.urlopen(url_income) as response:
    data_income = json.loads(response.read().decode())

# Create financial metrics lookups (by fiscalDateEnding)
financial_lookup = {}  # Will store all financial metrics

# Process quarterly reports
for report in data_income.get('quarterlyReports', []):
    date = report.get('fiscalDateEnding', '')
    if date:
        try:
            total_revenue = float(report.get('totalRevenue', 0))
            gross_profit = float(report.get('grossProfit', 0))
            operating_income = float(report.get('operatingIncome', 0))
            net_income = float(report.get('netIncome', 0))
            ebitda = float(report.get('ebitda', 0))
            
            financial_lookup[date] = {
                'total_revenue': total_revenue,
                'operating_income': operating_income,
                'ebitda': ebitda,
                'gross_margin': round((gross_profit / total_revenue) * 100, 2) if total_revenue > 0 else 0,
                'operating_margin': round((operating_income / total_revenue) * 100, 2) if total_revenue > 0 else 0,
                'net_margin': round((net_income / total_revenue) * 100, 2) if total_revenue > 0 else 0
            }
        except (ValueError, TypeError, ZeroDivisionError):
            pass

# Process annual reports
for report in data_income.get('annualReports', []):
    date = report.get('fiscalDateEnding', '')
    if date:
        try:
            total_revenue = float(report.get('totalRevenue', 0))
            gross_profit = float(report.get('grossProfit', 0))
            operating_income = float(report.get('operatingIncome', 0))
            net_income = float(report.get('netIncome', 0))
            ebitda = float(report.get('ebitda', 0))
            
            financial_lookup[date] = {
                'total_revenue': total_revenue,
                'operating_income': operating_income,
                'ebitda': ebitda,
                'gross_margin': round((gross_profit / total_revenue) * 100, 2) if total_revenue > 0 else 0,
                'operating_margin': round((operating_income / total_revenue) * 100, 2) if total_revenue > 0 else 0,
                'net_margin': round((net_income / total_revenue) * 100, 2) if total_revenue > 0 else 0
            }
        except (ValueError, TypeError, ZeroDivisionError):
            pass

print(f"   ✓ {len(financial_lookup)} dönem için finansal veriler hesaplandı")

# Fetch CASH FLOW data (for free cash flow)
print("   4/5 Nakit akışı verileri (Cash Flow)...")
url_cashflow = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'
with urllib.request.urlopen(url_cashflow) as response:
    data_cashflow = json.loads(response.read().decode())

# Create free cash flow lookup
fcf_lookup = {}

# Process quarterly cash flows
for report in data_cashflow.get('quarterlyReports', []):
    date = report.get('fiscalDateEnding', '')
    if date:
        try:
            operating_cashflow = float(report.get('operatingCashflow', 0))
            capital_expenditures = float(report.get('capitalExpenditures', 0))
            free_cash_flow = operating_cashflow - abs(capital_expenditures)
            fcf_lookup[date] = free_cash_flow
        except (ValueError, TypeError):
            pass

# Process annual cash flows
for report in data_cashflow.get('annualReports', []):
    date = report.get('fiscalDateEnding', '')
    if date:
        try:
            operating_cashflow = float(report.get('operatingCashflow', 0))
            capital_expenditures = float(report.get('capitalExpenditures', 0))
            free_cash_flow = operating_cashflow - abs(capital_expenditures)
            fcf_lookup[date] = free_cash_flow
        except (ValueError, TypeError):
            pass

print(f"   ✓ {len(fcf_lookup)} dönem için free cash flow hesaplandı")

# Add actual_eps, reported_date, financial metrics to estimates
for row in estimates:
    date = row['date']
    horizon = row['horizon']
    
    # Choose the right lookup based on horizon type
    if 'fiscal year' in horizon:
        row['actual_eps'] = annual_eps_lookup.get(date, '')
        row['reported_date'] = ''  # No reported date for annual
    elif 'quarter' in horizon:
        row['actual_eps'] = quarterly_eps_lookup.get(date, '')
        row['reported_date'] = quarterly_reported_dates.get(date, '')
    else:
        row['actual_eps'] = ''
        row['reported_date'] = ''
    
    # Add financial metrics (revenue, margins, EBITDA, FCF)
    financials = financial_lookup.get(date, {})
    row['total_revenue'] = financials.get('total_revenue', '')
    row['operating_income'] = financials.get('operating_income', '')
    row['ebitda'] = financials.get('ebitda', '')
    row['gross_margin'] = financials.get('gross_margin', '')
    row['operating_margin'] = financials.get('operating_margin', '')
    row['net_margin'] = financials.get('net_margin', '')
    row['free_cash_flow'] = fcf_lookup.get(date, '')

# Fetch PRICE data (using free TIME_SERIES_DAILY endpoint)
print("   5/6 Fiyat verileri (Price History)...")
url_prices = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}'
with urllib.request.urlopen(url_prices) as response:
    data_prices = json.loads(response.read().decode())

prices = data_prices.get('Time Series (Daily)', {})
if prices:
    print(f"   ✓ {len(prices)} günlük fiyat verisi çekildi")
else:
    print(f"   ⚠️  Fiyat verisi alınamadı (API limiti veya premium endpoint)")
    if 'Note' in data_prices or 'Information' in data_prices:
        print(f"   API mesajı: {data_prices.get('Note') or data_prices.get('Information')}")

# Skip saving original CSV - we only need the _with_q4.csv version

# ============================================================================
# FİYAT DEĞİŞİMİ HESAPLAMA (Price Changes)
# ============================================================================

from datetime import datetime, timedelta

def calculate_price_change(reported_date, prices, months_back):
    """
    Calculate price change from X months before the reported date.
    
    Args:
        reported_date: Earnings announcement date
        prices: Price dictionary from Alpha Vantage
        months_back: How many months to look back
    
    Returns:
        (price_before, price_at_report, change_pct)
    """
    if not reported_date or not prices:
        return None, None, None
    
    try:
        report_dt = datetime.strptime(reported_date, '%Y-%m-%d')
        before_dt = report_dt - timedelta(days=months_back*30)
        
        # Find price at report date (or closest available)
        price_at_report = None
        for i in range(10):  # Check up to 10 days around
            check_date = (report_dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if check_date in prices:
                price_at_report = float(prices[check_date]['4. close'])
                break
        
        if not price_at_report:
            return None, None, None
        
        # Find price around before_dt
        price_before = None
        for i in range(10):  # Check up to 10 days back
            check_date = (before_dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if check_date in prices:
                price_before = float(prices[check_date]['4. close'])
                break
        
        if price_before:
            change_pct = ((price_at_report - price_before) / price_before) * 100
            return price_before, price_at_report, change_pct
    except Exception as e:
        pass
    
    return None, None, None

print("   6/6 Fiyat değişimleri hesaplanıyor...")

# Add price change columns to estimates
for row in estimates:
    reported_date = row.get('reported_date', '')
    horizon = row.get('horizon', '')
    
    # Only calculate for quarterly data (not fiscal year)
    if reported_date and 'quarter' in horizon.lower() and 'fiscal year' not in horizon.lower():
        # 1 month price change
        p1m_before, p1m_at, p1m_change = calculate_price_change(reported_date, prices, 1)
        row['price_1m_before'] = round(p1m_before, 2) if p1m_before else ''
        row['price_at_report'] = round(p1m_at, 2) if p1m_at else ''
        row['price_change_1m_pct'] = round(p1m_change, 2) if p1m_change else ''
        
        # 3 month price change
        p3m_before, _, p3m_change = calculate_price_change(reported_date, prices, 3)
        row['price_3m_before'] = round(p3m_before, 2) if p3m_before else ''
        row['price_change_3m_pct'] = round(p3m_change, 2) if p3m_change else ''
    else:
        # No price data for fiscal year or missing reported_date
        row['price_1m_before'] = ''
        row['price_at_report'] = ''
        row['price_change_1m_pct'] = ''
        row['price_3m_before'] = ''
        row['price_change_3m_pct'] = ''

# FIXED: Update original_fields with all financial metrics in fixed order
original_fields = [
    'date',
    'horizon',
    'actual_eps',
    'reported_date',
    'price_1m_before',
    'price_at_report',
    'price_change_1m_pct',
    'price_3m_before',
    'price_change_3m_pct',
    'total_revenue',
    'operating_income',
    'ebitda',
    'free_cash_flow',
    'gross_margin',
    'operating_margin',
    'net_margin',
    'eps_estimate_average',
    'eps_estimate_high',
    'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago',
    'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago',
    'eps_estimate_average_90_days_ago',
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    'revenue_estimate_average',
    'revenue_estimate_high',
    'revenue_estimate_low',
    'revenue_estimate_analyst_count'
]

# ============================================================================
# Q4 EPS HESAPLAMA (Fiscal Year - Q1 - Q2 - Q3)
# ============================================================================

# Group by year
years_data = defaultdict(lambda: {'fiscal_year': None, 'quarters': []})

for row in estimates:
    date = row['date']
    year = date.split('-')[0]
    horizon = row['horizon']
    
    if 'fiscal year' in horizon:
        years_data[year]['fiscal_year'] = row
    elif 'quarter' in horizon:
        years_data[year]['quarters'].append(row)

# Calculate Q4 for each year
results = []

for year in sorted(years_data.keys(), reverse=True):
    year_info = years_data[year]
    
    if year_info['fiscal_year'] and len(year_info['quarters']) >= 2:
        fiscal_year_eps = float(year_info['fiscal_year']['eps_estimate_average'])
        fiscal_year_revenue = float(year_info['fiscal_year']['revenue_estimate_average'])
        
        # Sum all quarters
        quarters_eps_sum = sum(float(q['eps_estimate_average']) for q in year_info['quarters'])
        quarters_revenue_sum = sum(float(q['revenue_estimate_average']) for q in year_info['quarters'])
        
        # Calculate Q4 (missing quarter)
        q4_eps = fiscal_year_eps - quarters_eps_sum
        q4_revenue = fiscal_year_revenue - quarters_revenue_sum
        
        results.append({
            'year': year,
            'fiscal_year_eps': fiscal_year_eps,
            'quarters_count': len(year_info['quarters']),
            'quarters_eps_sum': quarters_eps_sum,
            'q4_eps_calculated': round(q4_eps, 4),
            'q4_revenue_calculated': round(q4_revenue, 2)
        })

# Create enhanced CSV with Q4 rows
output_rows = []

for row in estimates:
    output_rows.append(row)
    
    # If this is a fiscal year row, add calculated Q4 after it
    if 'fiscal year' in row['horizon']:
        year = row['date'].split('-')[0]
        
        # Find if we calculated Q4 for this year
        q4_data = next((r for r in results if r['year'] == year), None)
        
        if q4_data and q4_data['quarters_count'] == 3:  # Only if we have exactly 3 quarters
            q4_row = row.copy()
            q4_date = f"{year}-12-31"
            q4_row['date'] = q4_date
            
            # Use API Q4 estimate if available, otherwise use calculation
            if q4_date in q4_estimate_lookup:
                q4_row['eps_estimate_average'] = q4_estimate_lookup[q4_date]
                q4_row['horizon'] = 'Q4 from API (EARNINGS estimatedEPS)'
            else:
                q4_row['eps_estimate_average'] = str(q4_data['q4_eps_calculated'])
                q4_row['horizon'] = 'Q4 calculated (fiscal year - Q1-Q2-Q3)'
            
            q4_row['revenue_estimate_average'] = str(q4_data['q4_revenue_calculated'])
            
            # Calculate actual Q4 EPS if data available
            fiscal_year_actual = row.get('actual_eps', '')
            if fiscal_year_actual:
                try:
                    fiscal_actual_float = float(fiscal_year_actual)
                    # Get year info for this year
                    year_info_for_q4 = years_data[year]
                    # Sum actual EPS from quarters
                    quarters_actual_sum = 0
                    quarters_with_actual = 0
                    for q in year_info_for_q4['quarters']:
                        q_actual = q.get('actual_eps', '')
                        if q_actual:
                            quarters_actual_sum += float(q_actual)
                            quarters_with_actual += 1
                    
                    # Only calculate if we have all 3 quarters' actuals
                    if quarters_with_actual == 3:
                        q4_actual = fiscal_actual_float - quarters_actual_sum
                        q4_row['actual_eps'] = str(round(q4_actual, 2))
                    else:
                        q4_row['actual_eps'] = ''
                except (ValueError, TypeError):
                    q4_row['actual_eps'] = ''
            else:
                q4_row['actual_eps'] = ''
            
            # Q4 rows: add reported_date if available (for API Q4s)
            q4_reported = quarterly_reported_dates.get(q4_date, '')
            q4_row['reported_date'] = q4_reported
            
            # Calculate price changes for Q4 if reported_date exists
            if q4_reported:
                p1m_before, p1m_at, p1m_change = calculate_price_change(q4_reported, prices, 1)
                q4_row['price_1m_before'] = round(p1m_before, 2) if p1m_before else ''
                q4_row['price_at_report'] = round(p1m_at, 2) if p1m_at else ''
                q4_row['price_change_1m_pct'] = round(p1m_change, 2) if p1m_change else ''
                
                p3m_before, _, p3m_change = calculate_price_change(q4_reported, prices, 3)
                q4_row['price_3m_before'] = round(p3m_before, 2) if p3m_before else ''
                q4_row['price_change_3m_pct'] = round(p3m_change, 2) if p3m_change else ''
            else:
                # No reported_date for calculated Q4s
                q4_row['price_1m_before'] = ''
                q4_row['price_at_report'] = ''
                q4_row['price_change_1m_pct'] = ''
                q4_row['price_3m_before'] = ''
                q4_row['price_change_3m_pct'] = ''
            
            # Add Q4 financial metrics if available
            q4_financials = financial_lookup.get(q4_date, {})
            q4_row['total_revenue'] = q4_financials.get('total_revenue', '')
            q4_row['operating_income'] = q4_financials.get('operating_income', '')
            q4_row['ebitda'] = q4_financials.get('ebitda', '')
            q4_row['gross_margin'] = q4_financials.get('gross_margin', '')
            q4_row['operating_margin'] = q4_financials.get('operating_margin', '')
            q4_row['net_margin'] = q4_financials.get('net_margin', '')
            q4_row['free_cash_flow'] = fcf_lookup.get(q4_date, '')
            
            output_rows.append(q4_row)

# Write enhanced CSV (with actual_eps column)
csv_with_q4 = paths['raw_csv']
with open(csv_with_q4, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=original_fields)
    writer.writeheader()
    writer.writerows(output_rows)

# Count Q4s from API vs calculated
q4_from_api = sum(1 for row in output_rows if 'Q4 from API' in row.get('horizon', ''))
q4_calculated = sum(1 for row in output_rows if 'Q4 calculated' in row.get('horizon', ''))

print(f"✓ Q4 hesaplamalı veri kaydedildi: {csv_with_q4}")
print(f"✓ Toplam {len(output_rows)} satır ({q4_from_api} Q4 API'dan, {q4_calculated} Q4 hesaplanan)")

# Print Q4 calculations summary
if results:
    print("\n" + "=" * 90)
    print(f"{'Year':<8} {'Fiscal EPS':<12} {'Q1+Q2+Q3':<12} {'Q4 (Calc)':<12} {'Q4 Revenue':<20}")
    print("=" * 90)
    for r in results:
        print(f"{r['year']:<8} ${r['fiscal_year_eps']:<11.4f} ${r['quarters_eps_sum']:<11.4f} "
              f"${r['q4_eps_calculated']:<11.4f} ${r['q4_revenue_calculated']:>18,.0f}")
    print("=" * 90)

# ============================================================================
# ELO RATING SYSTEM
# ============================================================================

# Elo configuration
INITIAL_ELO = 1500
BASE_K = 32

def calculate_adaptive_K(n_analysts, base_k=BASE_K):
    """
    Calculate adaptive K-factor based on analyst coverage.
    Low coverage → higher K (more volatile)
    High coverage → lower K (more stable)
    
    Args:
        n_analysts: Number of analysts covering the stock
        base_k: Base K-factor (default: 32)
    
    Returns:
        Adjusted K-factor
    """
    if n_analysts >= 20:
        return base_k * 0.7  # High coverage → more stable
    elif n_analysts >= 15:
        return base_k * 0.85
    elif n_analysts >= 10:
        return base_k * 1.0  # Normal
    elif n_analysts >= 5:
        return base_k * 1.2
    else:
        return base_k * 1.5  # Low coverage → more volatile

def update_elo(current_elo, actual_eps, estimate_eps, K):
    """
    Update Elo rating based on earnings performance.
    
    Formula: new_elo = old_elo + K * (actual - expected)
    
    Args:
        current_elo: Current Elo rating
        actual_eps: Actual reported EPS
        estimate_eps: Estimated EPS
        K: K-factor (sensitivity to changes)
    
    Returns:
        New Elo rating
    """
    surprise = actual_eps - estimate_eps
    # Scale surprise to Elo points (multiply by 100 to make changes more visible)
    elo_change = K * surprise * 100
    new_elo = current_elo + elo_change
    return new_elo

def calculate_elo_decay(elo_history, weights=[0.4, 0.3, 0.2, 0.1]):
    """
    Calculate weighted average of recent Elo ratings.
    Recent quarters get more weight.
    
    Args:
        elo_history: List of recent Elo ratings (most recent first)
        weights: Weights for [most recent, ..., oldest]
    
    Returns:
        Weighted Elo average
    """
    if len(elo_history) == 0:
        return np.nan
    
    # Take last N values (up to length of weights)
    recent = elo_history[-len(weights):]
    n = len(recent)
    
    if n == 0:
        return np.nan
    
    # Use only relevant weights
    w = weights[:n]
    # Normalize weights
    w = np.array(w) / sum(w)
    
    return np.dot(recent, w)

def calculate_elo_volatility(elo_changes, window=4):
    """
    Calculate rolling standard deviation of Elo changes.
    
    Args:
        elo_changes: List of recent Elo changes
        window: Rolling window size (default: 4 quarters)
    
    Returns:
        Standard deviation of recent Elo changes
    """
    if len(elo_changes) < 2:
        return 0.0
    
    recent = elo_changes[-window:]
    return np.std(recent)

def apply_elo_ratings(rows):
    """
    Apply Elo rating system to earnings data.
    
    Args:
        rows: List of dicts with earnings data (must be sorted by date)
    
    Returns:
        List of dicts with added Elo columns
    """
    print("\n" + "=" * 90)
    print("🎯 ELO RATING SYSTEM UYGULANIY...")
    print("=" * 90)
    
    # Sort by date (oldest first)
    rows_sorted = sorted(rows, key=lambda x: x['date'])
    
    # Track Elo history
    current_elo = INITIAL_ELO
    elo_history = []
    elo_changes = []
    
    # Process each row
    for i, row in enumerate(rows_sorted):
        # Get data
        actual_eps = row.get('actual_eps', '')
        estimate_eps = row.get('eps_estimate_average', '')
        n_analysts = float(row.get('eps_estimate_analyst_count', 10))
        horizon = row.get('horizon', '')
        
        # Elo before this event
        row['elo_before'] = round(current_elo, 2)
        
        # ONLY UPDATE ELO FOR QUARTERLY EVENTS (skip annual to avoid double-counting)
        is_quarterly = 'quarter' in horizon.lower() or 'q4' in horizon.lower()
        
        # If we have actual EPS AND it's a quarterly event, update Elo
        if actual_eps and estimate_eps and is_quarterly:
            try:
                actual = float(actual_eps)
                estimate = float(estimate_eps)
                
                # Calculate adaptive K
                K = calculate_adaptive_K(n_analysts)
                row['K_adaptive'] = round(K, 2)
                
                # Update Elo
                new_elo = update_elo(current_elo, actual, estimate, K)
                elo_change = new_elo - current_elo
                
                row['elo_after'] = round(new_elo, 2)
                row['elo_change'] = round(elo_change, 2)
                
                # Update tracking
                elo_history.append(current_elo)
                elo_changes.append(elo_change)
                current_elo = new_elo
                
            except (ValueError, TypeError):
                # If conversion fails, keep current Elo
                row['elo_after'] = round(current_elo, 2)
                row['elo_change'] = 0.0
                row['K_adaptive'] = round(BASE_K, 2)
        else:
            # Annual events or future estimates: no Elo update, just keep current
            if 'fiscal year' in horizon.lower() and actual_eps:
                # Annual row with actual: show current Elo but no change
                row['elo_after'] = round(current_elo, 2)
                row['elo_change'] = 0.0
                row['K_adaptive'] = np.nan
            else:
                # Future estimates: no Elo update
                row['elo_after'] = np.nan
                row['elo_change'] = np.nan
                row['K_adaptive'] = round(calculate_adaptive_K(n_analysts), 2)
        
        # Calculate Elo decay (weighted average)
        if len(elo_history) > 0:
            row['elo_decay'] = round(calculate_elo_decay(elo_history), 2)
        else:
            row['elo_decay'] = round(current_elo, 2)
        
        # Calculate Elo volatility
        if len(elo_changes) > 0:
            row['elo_vol_4q'] = round(calculate_elo_volatility(elo_changes), 2)
        else:
            row['elo_vol_4q'] = 0.0
        
        # Calculate Elo momentum (average of recent changes)
        if len(elo_changes) > 0:
            recent_changes = elo_changes[-3:]  # Last 3 quarters
            row['elo_momentum'] = round(np.mean(recent_changes), 2)
        else:
            row['elo_momentum'] = 0.0
    
    # Print summary
    quarterly_events = [r for r in rows_sorted if 'quarter' in r.get('horizon', '').lower() or 'q4' in r.get('horizon', '').lower()]
    quarterly_with_actual = [r for r in quarterly_events if r.get('actual_eps')]
    
    print(f"\n✓ Elo ratings hesaplandı (sadece quarterly events):")
    print(f"  • Başlangıç Elo: {INITIAL_ELO}")
    print(f"  • Mevcut Elo: {round(current_elo, 2)}")
    print(f"  • Toplam değişim: {round(current_elo - INITIAL_ELO, 2)}")
    print(f"  • Quarterly events: {len(quarterly_with_actual)}")
    print(f"  • Elo range: {round(min(r['elo_before'] for r in rows_sorted), 2)} - {round(max(r['elo_before'] for r in rows_sorted), 2)}")
    
    # Print first and last events
    print(f"\n📊 İlk Quarterly Event:")
    if quarterly_with_actual:
        first = quarterly_with_actual[0]
        print(f"  {first['date']} {first['horizon']}")
        print(f"  Elo: {first['elo_before']} → {first['elo_after']} (Δ {first['elo_change']:+.2f})")
    
        print(f"\n📊 Son Quarterly Event:")
        last = quarterly_with_actual[-1]
        print(f"  {last['date']} {last['horizon']}")
        print(f"  Elo: {last['elo_before']} → {last['elo_after']} (Δ {last['elo_change']:+.2f})")
    
    print("\n⚠️  NOT: Annual (fiscal year) satırları Elo hesaplamasına dahil DEĞİL")
    print("    (Annual = Q1+Q2+Q3+Q4 toplamı, double-counting önlendi)")
    print("=" * 90)
    
    return rows_sorted

# Apply Elo to the data
output_rows = apply_elo_ratings(output_rows)

# ============================================================================
# GROWTH METRICS CALCULATION
# ============================================================================

from datetime import datetime

def parse_quarter(date_str):
    """Parse date string and return (year, quarter)"""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        quarter = (dt.month - 1) // 3 + 1
        return dt.year, quarter
    except:
        return None, None

def calculate_growth(current, previous):
    """Calculate percentage growth, handling edge cases"""
    try:
        current = float(current) if current else None
        previous = float(previous) if previous else None
        
        if current is None or previous is None:
            return np.nan
        
        # Handle zero/negative denominators
        if previous == 0:
            return np.nan
        
        # For negative values, use absolute value in denominator
        if previous < 0:
            # If both negative, standard calculation
            # If switching signs, mark as special case
            growth = ((current - previous) / abs(previous)) * 100
            return round(growth, 2)
        
        return round(((current - previous) / previous) * 100, 2)
    except (ValueError, TypeError, ZeroDivisionError):
        return np.nan

def apply_growth_metrics(rows):
    """
    Calculate growth metrics for quarterly data:
    - YoY Same-Quarter Growth (Q3 2025 vs Q3 2024)
    - QoQ Sequential Growth (Q3 2025 vs Q2 2025)
    - TTM YoY Growth (Trailing 12 months)
    - Smoothed QoQ (2Q and 4Q averages)
    - Margin Changes (YoY)
    """
    print("\n" + "=" * 90)
    print("📈 GROWTH METRICS HESAPLANIYOR...")
    print("=" * 90)
    
    # Sort by date and filter quarterly events only
    all_rows = sorted(rows, key=lambda x: x['date'])
    quarterly_rows = [r for r in all_rows if 'quarter' in r.get('horizon', '').lower() or 'q4' in r.get('horizon', '').lower()]
    
    # Create lookup dictionary: (year, quarter) -> row
    quarter_lookup = {}
    for row in quarterly_rows:
        year, quarter = parse_quarter(row['date'])
        if year and quarter:
            quarter_lookup[(year, quarter)] = row
    
    # Metrics to calculate growth for
    growth_metrics = [
        'total_revenue',
        'actual_eps',
        'ebitda',
        'operating_income',
        'free_cash_flow'
    ]
    
    # Store QoQ values for smoothing
    qoq_history = {metric: [] for metric in growth_metrics}
    
    print(f"\n✓ {len(quarterly_rows)} quarterly events bulundu")
    print(f"✓ Hesaplanacak metrikler: {', '.join(growth_metrics)}")
    
    # Calculate growth for each quarterly row
    for i, row in enumerate(quarterly_rows):
        year, quarter = parse_quarter(row['date'])
        if not year or not quarter:
            continue
        
        # ============================================
        # 1. YoY Same-Quarter Growth
        # ============================================
        prev_year_key = (year - 1, quarter)
        if prev_year_key in quarter_lookup:
            prev_year_row = quarter_lookup[prev_year_key]
            for metric in growth_metrics:
                current_val = row.get(metric)
                prev_val = prev_year_row.get(metric)
                growth = calculate_growth(current_val, prev_val)
                row[f'{metric}_yoy_growth'] = growth
        else:
            for metric in growth_metrics:
                row[f'{metric}_yoy_growth'] = np.nan
        
        # ============================================
        # 2. QoQ Sequential Growth
        # ============================================
        if i > 0:
            prev_quarter_row = quarterly_rows[i - 1]
            for metric in growth_metrics:
                current_val = row.get(metric)
                prev_val = prev_quarter_row.get(metric)
                growth = calculate_growth(current_val, prev_val)
                row[f'{metric}_qoq_growth'] = growth
                
                # Store for smoothing
                if not np.isnan(growth):
                    qoq_history[metric].append(growth)
        else:
            for metric in growth_metrics:
                row[f'{metric}_qoq_growth'] = np.nan
        
        # ============================================
        # 3. TTM YoY Growth
        # ============================================
        # Need at least 8 quarters (4 current + 4 previous year)
        if i >= 7:
            # Current TTM: sum of last 4 quarters (including current)
            ttm_current = {}
            ttm_previous = {}
            
            for metric in growth_metrics:
                # Sum last 4 quarters
                current_sum = sum(
                    float(val) if val and val != 'None' else 0
                    for j in range(i - 3, i + 1)
                    if (val := quarterly_rows[j].get(metric))
                )
                
                # Sum quarters from 1 year ago (i-7 to i-4)
                previous_sum = sum(
                    float(val) if val and val != 'None' else 0
                    for j in range(i - 7, i - 3)
                    if (val := quarterly_rows[j].get(metric))
                )
                
                if current_sum > 0 or previous_sum > 0:
                    ttm_growth = calculate_growth(current_sum, previous_sum)
                    row[f'{metric}_ttm_yoy_growth'] = ttm_growth
                else:
                    row[f'{metric}_ttm_yoy_growth'] = np.nan
        else:
            for metric in growth_metrics:
                row[f'{metric}_ttm_yoy_growth'] = np.nan
        
        # ============================================
        # 4. Smoothed QoQ (2Q and 4Q averages)
        # ============================================
        for metric in growth_metrics:
            qoq_current = row.get(f'{metric}_qoq_growth')
            
            # 2Q average
            if i >= 1 and not np.isnan(qoq_current):
                prev_qoq = quarterly_rows[i - 1].get(f'{metric}_qoq_growth')
                if not np.isnan(prev_qoq):
                    row[f'{metric}_qoq_2q_avg'] = round((qoq_current + prev_qoq) / 2, 2)
                else:
                    row[f'{metric}_qoq_2q_avg'] = np.nan
            else:
                row[f'{metric}_qoq_2q_avg'] = np.nan
            
            # 4Q average
            if i >= 3:
                recent_qoqs = [
                    quarterly_rows[j].get(f'{metric}_qoq_growth')
                    for j in range(i - 3, i + 1)
                ]
                valid_qoqs = [q for q in recent_qoqs if not np.isnan(q)]
                if len(valid_qoqs) >= 2:
                    row[f'{metric}_qoq_4q_avg'] = round(np.mean(valid_qoqs), 2)
                else:
                    row[f'{metric}_qoq_4q_avg'] = np.nan
            else:
                row[f'{metric}_qoq_4q_avg'] = np.nan
        
        # ============================================
        # 5. Margin Changes (YoY in percentage points)
        # ============================================
        margin_metrics = ['gross_margin', 'operating_margin', 'net_margin']
        if prev_year_key in quarter_lookup:
            prev_year_row = quarter_lookup[prev_year_key]
            for metric in margin_metrics:
                current_val = row.get(metric)
                prev_val = prev_year_row.get(metric)
                try:
                    current_val = float(current_val) if current_val else None
                    prev_val = float(prev_val) if prev_val else None
                    if current_val is not None and prev_val is not None:
                        # Percentage point change (not % growth)
                        row[f'{metric}_yoy_change'] = round(current_val - prev_val, 2)
                    else:
                        row[f'{metric}_yoy_change'] = np.nan
                except (ValueError, TypeError):
                    row[f'{metric}_yoy_change'] = np.nan
        else:
            for metric in margin_metrics:
                row[f'{metric}_yoy_change'] = np.nan
    
    # Initialize growth columns for non-quarterly rows (annual, future)
    non_quarterly = [r for r in all_rows if r not in quarterly_rows]
    for row in non_quarterly:
        for metric in growth_metrics:
            row[f'{metric}_yoy_growth'] = np.nan
            row[f'{metric}_qoq_growth'] = np.nan
            row[f'{metric}_ttm_yoy_growth'] = np.nan
            row[f'{metric}_qoq_2q_avg'] = np.nan
            row[f'{metric}_qoq_4q_avg'] = np.nan
        for metric in ['gross_margin', 'operating_margin', 'net_margin']:
            row[f'{metric}_yoy_change'] = np.nan
    
    # Print summary statistics
    print("\n✓ Growth metrics hesaplandı:")
    print(f"  • YoY calculations: {sum(1 for r in quarterly_rows if not np.isnan(r.get('total_revenue_yoy_growth', np.nan)))}")
    print(f"  • QoQ calculations: {sum(1 for r in quarterly_rows if not np.isnan(r.get('total_revenue_qoq_growth', np.nan)))}")
    print(f"  • TTM calculations: {sum(1 for r in quarterly_rows if not np.isnan(r.get('total_revenue_ttm_yoy_growth', np.nan)))}")
    
    # Show example of recent growth rates
    if len(quarterly_rows) >= 3:
        print(f"\n📊 Son 3 çeyrek Revenue growth rates:")
        for row in quarterly_rows[-3:]:
            date = row['date']
            yoy = row.get('total_revenue_yoy_growth', np.nan)
            qoq = row.get('total_revenue_qoq_growth', np.nan)
            print(f"  {date}: YoY={yoy:>6.1f}%, QoQ={qoq:>6.1f}%")
    
    # ============================================
    # 6. ADD LAG-1 FEATURES (Previous quarter's growth)
    # ============================================
    print(f"\n✓ Lag-1 features ekleniyor...")
    lag_features = [f'{m}_yoy_growth' for m in growth_metrics] + \
                   [f'{m}_qoq_growth' for m in growth_metrics] + \
                   [f'{m}_ttm_yoy_growth' for m in growth_metrics] + \
                   ['gross_margin_yoy_change', 'operating_margin_yoy_change', 'net_margin_yoy_change']
    
    for i, row in enumerate(quarterly_rows):
        if i > 0:
            prev = quarterly_rows[i - 1]
            for feat in lag_features:
                row[f'{feat}_lag1'] = prev.get(feat, np.nan)
        else:
            for feat in lag_features:
                row[f'{feat}_lag1'] = np.nan
    
    # Initialize for non-quarterly
    for row in non_quarterly:
        for feat in lag_features:
            row[f'{feat}_lag1'] = np.nan
    
    print(f"  • {len(lag_features)} growth metric için lag-1 eklendi")
    print("=" * 90)
    
    return all_rows

# Apply growth metrics (includes lag features)
output_rows = apply_growth_metrics(output_rows)

# ============================================================================
# ADD EPS BEAT & DELTA COLUMNS
# ============================================================================
print("\n📊 EPS Beat & Delta hesaplanıyor...")

for row in output_rows:
    actual_eps = row.get('actual_eps', '')
    estimate_eps = row.get('eps_estimate_average', '')
    
    if actual_eps and estimate_eps:
        try:
            actual = float(actual_eps)
            estimate = float(estimate_eps)
            
            # eps_beat: 1 if beat, 0 if miss or equal
            row['eps_beat'] = 1 if actual > estimate else 0
            
            # eps_delta: actual - estimate
            row['eps_delta'] = round(actual - estimate, 4)
            
        except (ValueError, TypeError):
            row['eps_beat'] = np.nan
            row['eps_delta'] = np.nan
    else:
        row['eps_beat'] = np.nan
        row['eps_delta'] = np.nan

# FIXED: Final column order - ALWAYS THE SAME for all symbols
all_fields = [
    'date',
    'horizon',
    'eps_beat',
    'eps_delta',
    'actual_eps',
    'reported_date',
    'price_1m_before',
    'price_at_report',
    'price_change_1m_pct',
    'price_3m_before',
    'price_change_3m_pct',
    'total_revenue',
    'operating_income',
    'ebitda',
    'free_cash_flow',
    'gross_margin',
    'operating_margin',
    'net_margin',
    'eps_estimate_average',
    'eps_estimate_high',
    'eps_estimate_low',
    'eps_estimate_analyst_count',
    'eps_estimate_average_7_days_ago',
    'eps_estimate_average_30_days_ago',
    'eps_estimate_average_60_days_ago',
    'eps_estimate_average_90_days_ago',
    'eps_estimate_revision_up_trailing_7_days',
    'eps_estimate_revision_down_trailing_7_days',
    'eps_estimate_revision_up_trailing_30_days',
    'eps_estimate_revision_down_trailing_30_days',
    'revenue_estimate_average',
    'revenue_estimate_high',
    'revenue_estimate_low',
    'revenue_estimate_analyst_count',
    'elo_before',
    'elo_after',
    'elo_change',
    'elo_decay',
    'elo_vol_4q',
    'elo_momentum',
    'K_adaptive',
    # Growth metrics - Revenue
    'total_revenue_yoy_growth',
    'total_revenue_qoq_growth',
    'total_revenue_ttm_yoy_growth',
    'total_revenue_qoq_2q_avg',
    'total_revenue_qoq_4q_avg',
    # Growth metrics - EPS
    'actual_eps_yoy_growth',
    'actual_eps_qoq_growth',
    'actual_eps_ttm_yoy_growth',
    'actual_eps_qoq_2q_avg',
    'actual_eps_qoq_4q_avg',
    # Growth metrics - EBITDA
    'ebitda_yoy_growth',
    'ebitda_qoq_growth',
    'ebitda_ttm_yoy_growth',
    'ebitda_qoq_2q_avg',
    'ebitda_qoq_4q_avg',
    # Growth metrics - Operating Income
    'operating_income_yoy_growth',
    'operating_income_qoq_growth',
    'operating_income_ttm_yoy_growth',
    'operating_income_qoq_2q_avg',
    'operating_income_qoq_4q_avg',
    # Growth metrics - Free Cash Flow
    'free_cash_flow_yoy_growth',
    'free_cash_flow_qoq_growth',
    'free_cash_flow_ttm_yoy_growth',
    'free_cash_flow_qoq_2q_avg',
    'free_cash_flow_qoq_4q_avg',
    # Margin changes (YoY percentage point changes)
    'gross_margin_yoy_change',
    'operating_margin_yoy_change',
    'net_margin_yoy_change',
    # Lagged growth features (t-1) - SAFE for ML prediction
    'total_revenue_yoy_growth_lag1',
    'total_revenue_qoq_growth_lag1',
    'total_revenue_ttm_yoy_growth_lag1',
    'total_revenue_qoq_2q_avg_lag1',
    'total_revenue_qoq_4q_avg_lag1',
    'actual_eps_yoy_growth_lag1',
    'actual_eps_qoq_growth_lag1',
    'actual_eps_ttm_yoy_growth_lag1',
    'actual_eps_qoq_2q_avg_lag1',
    'actual_eps_qoq_4q_avg_lag1',
    'ebitda_yoy_growth_lag1',
    'ebitda_qoq_growth_lag1',
    'ebitda_ttm_yoy_growth_lag1',
    'operating_income_yoy_growth_lag1',
    'operating_income_qoq_growth_lag1',
    'operating_income_ttm_yoy_growth_lag1',
    'free_cash_flow_yoy_growth_lag1',
    'free_cash_flow_qoq_growth_lag1',
    'free_cash_flow_ttm_yoy_growth_lag1',
    'gross_margin_yoy_change_lag1',
    'operating_margin_yoy_change_lag1',
    'net_margin_yoy_change_lag1'
]

# Re-save the CSV with all features
with open(csv_with_q4, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"✓ EPS beat/delta kaydedildi")
print(f"✓ Toplam kolonlar: {len(all_fields)}")
print(f"\n✓ Final veri kaydedildi: {csv_with_q4}")

