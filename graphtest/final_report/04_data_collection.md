# 3. DATA COLLECTION & PREPARATION

## 3.1 S&P 500 Universe Construction

Our analysis begins with the S&P 500 index, representing the large-cap core of U.S. equities and comprising approximately 500 companies at any given time. We sourced the current constituent list as of November 2025, yielding 503 tickers including recent additions such as Palantir (PLTR), Uber (UBER), Coinbase (COIN), and other technology growth names that entered the index over the past two years. For each ticker, we attempted to fetch comprehensive earnings history via Alpha Vantage's premium API.

Of the 503 companies, we successfully retrieved data for 498 (99.0% success rate). The five failures—BF.B, BRK.B, FISV, Q, and SOLS—represent either Class B shares where Alpha Vantage tracks only Class A, SPACs with limited history, or tickers where API coverage is incomplete. These omissions are immaterial to our analysis given the 99% coverage achieved. More critically, of the 498 companies with data, 468 (93.6%) possess historical earnings with beat/miss classifications—meaning they have both analyst estimates and actual reported results for at least several quarters. The remaining 30 companies either are recent IPOs without sufficient history or have data quality issues preventing beat/miss determination.

This 468-company analytical universe represents $35+ trillion in market capitalization and spans all eleven GICS sectors, providing robust cross-sectional diversity for our machine learning models to learn generalizable patterns rather than idiosyncratic company effects.

## 3.2 Alpha Vantage Data Pipeline

For each of the 498 companies, we query four distinct Alpha Vantage API endpoints to construct a comprehensive feature set. The EARNINGS_ESTIMATES endpoint provides forward-looking analyst consensus: EPS estimates (average, high, low), analyst count, and crucially, historical snapshots of these estimates at 7, 30, 60, and 90 days before the current date. These historical snapshots allow us to construct revision momentum features without lookahead bias. The EARNINGS endpoint returns actual reported EPS figures and announcement dates, enabling us to calculate beats, misses, and Elo rating updates.

The INCOME_STATEMENT endpoint delivers quarterly revenues, operating income, EBITDA, and margins—fundamental metrics that inform our lagged growth features. The CASH_FLOW endpoint provides operating cash flow and capital expenditures, from which we derive free cash flow. Finally, the TIME_SERIES_DAILY endpoint gives us daily stock prices, allowing us to compute price levels at one month and three months before each earnings announcement (we specifically avoid using post-announcement price changes to prevent temporal leakage).

Data collection spanned approximately five hours using Alpha Vantage's premium tier (75 API calls per minute), fetching and processing 498 companies with automated error handling for missing data or malformed responses. The resulting dataset comprises 20,202 total rows across all companies, representing quarterly observations spanning from 2011 to 2025 depending on each company's data availability. After filtering to retain only quarterly earnings (excluding annual fiscal year summaries) and removing future quarters without actual results, we arrive at 14,239 observations suitable for model training and testing.

## 3.3 Data Quality Assessment

Data quality varies across tickers and time periods. Large-cap technology companies like Apple, Microsoft, and Google have complete histories with minimal missing values—typically 32-40 quarterly observations per company covering 8-10 years. Smaller constituents or recent S&P 500 additions may have only 15-20 quarters of data. We impose no minimum history requirement for inclusion, instead relying on our global (cross-company) modeling approach to pool information and handle sparse individual ticker histories.

Missing value patterns are systematic and informative. Analyst revision fields (upgrades and downgrades) are frequently null, particularly for smaller companies or older historical periods when Alpha Vantage's estimate tracking was less comprehensive. We treat these nulls as zeros under the interpretation that "no recorded revision" means no revision occurred. For numeric fields like revenue growth or margin changes, we apply median imputation within our preprocessing pipeline, ensuring models can handle incomplete feature vectors without dropping observations.

We document average data completeness at 85-90% across the 31 features we ultimately employ, with Elo metrics (which we calculate ourselves from reported results) achieving 100% completeness and analyst revision fields showing the lowest completeness at 50-60%. This heterogeneity is acceptable because our Random Forest and XGBoost models handle mixed completeness gracefully, and our imputation strategy is conservative (median rather than mean, to avoid outlier influence).

## 3.4 Temporal Coverage Analysis

The temporal distribution of our data reveals important trends. Figure 3.1 presents the yearly breakdown of earnings observations from 2017 through 2025, showing a stable flow of approximately 1,700-1,850 reports per year. Notably, beat rates have increased from 68.3% in 2017 to 77.0% in 2025—a nine-percentage-point secular trend that has significant implications for our models and strategies.

This rising beat rate could reflect several mechanisms. First, analysts may have grown systematically more conservative in response to criticism or regulatory pressure, lowering estimates to create easier hurdles for management teams. Second, companies may have improved their guidance processes, providing more realistic targets that they can then exceed. Third, the COVID-19 pandemic (2020-2021) and subsequent recovery may have created unusual volatility followed by strong rebounds, inflating recent beat rates temporarily.

Regardless of cause, the trend means our models cannot assume a stationary base rate. A model trained on 2011-2015 data (when beat rates were ~65%) will systematically underpredict beats in 2024-2025 (when beat rates approach 77%) unless it adapts. Our temporal train/test split (35% training on 2011-2020 data, 65% testing on 2020-2025 data) partially addresses this by training through the transition period, but we recommend quarterly recalibration as discussed in Section 12 to track evolving base rates.

The quarterly distribution within each year is balanced, with approximately equal numbers of March, June, September, and December fiscal quarter-end observations. This balance ensures our models are not biased toward any particular seasonal pattern, though we note that December quarters (often including holiday retail results) show slightly higher volatility in beat/miss outcomes.

---

**Figure 3.1: S&P 500 Beat Rate Trends (2017-2025)**  
*See: final_report/figures/fig1_yearly_beat_rate_trend.png*

**Table 3.1: Annual Data Summary**  
*See: final_report/tables/table2_yearly_beat_rates.txt*

The upward trend in beat rates from 67-69% (2017-2019) to 74-77% (2021-2025) represents a structural shift in analyst forecasting behavior or corporate earnings quality. Our models must adapt to this evolving baseline through regular retraining.

---

**Pages:** 13-17  
**Section:** Data Collection & Preparation  
**Classification:** Confidential

