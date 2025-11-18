# REFERENCES AND SOURCES

## Academic and Theoretical Foundations

### 1. Elo Rating System
**Original Work:**
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
- Concept: Adaptive rating system for predicting outcomes based on historical performance

**Application to Finance:**
- Our adaptation applies chess Elo methodology to corporate earnings predictions
- Novel contribution: elo_momentum, elo_decay, elo_vol_4q metrics for earnings context

### 2. Kelly Criterion
**Original Work:**
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), 917-926.
- DOI: 10.1002/j.1538-7305.1956.tb03809.x

**Concept:** 
- Optimal bet sizing formula: f* = (bp - q) / b
- Where: p = win probability, q = 1-p, b = odds
- Our implementation uses quarter-Kelly (25%) for risk management

### 3. Machine Learning Methods

**Random Forest:**
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
- DOI: 10.1023/A:1010933404324

**XGBoost:**
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, 785-794.
- DOI: 10.1145/2939672.2939785

**Logistic Regression:**
- Standard statistical method (no single citation required)
- Implementation via scikit-learn 1.3.0

### 4. Model Evaluation Metrics

**Brier Score:**
- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 78(1), 1-3.
- Measures probabilistic forecast accuracy: BS = (1/N) Σ(p_i - o_i)²

**ROC-AUC:**
- Hanley, J. A., & McNeil, B. J. (1982). "The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve." *Radiology*, 143(1), 29-36.

**Confusion Matrix / Classification Metrics:**
- Standard machine learning evaluation (no single source required)

---

## Data Sources

### 5. Alpha Vantage API
**Provider:** Alpha Vantage Inc.
- Website: https://www.alphavantage.co/
- Subscription: Premium tier (~$50/month)
- Data Used:
  - EARNINGS endpoint (actual reported EPS)
  - EARNINGS_ESTIMATES endpoint (analyst consensus, high, low, count)
  - INCOME_STATEMENT endpoint (revenue, EBITDA, operating income, margins)
  - CASH_FLOW endpoint (operating cash flow)
  - TIME_SERIES_DAILY endpoint (stock prices)
- License: Commercial use permitted under premium subscription terms
- Attribution: "Data provided by Alpha Vantage"

### 6. Polymarket Data
**Platform:** Polymarket (Polygon blockchain-based prediction market)
- Gamma API: https://gamma-api.polymarket.com/
- CLOB API: https://clob.polymarket.com/
- Data Type: Market prices, outcomes, trading history for earnings events
- License: Public API (no authentication required for reads)
- Rate Limits: ~100 requests/minute (observed)
- Time Period: Q3 2025 backtest (262 markets)

### 7. S&P 500 Constituent List
**Source:** Wikipedia
- URL: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
- License: Creative Commons Attribution-ShareAlike License
- Updated: Quarterly (manual refresh to track index changes)
- Alternative: S&P Dow Jones Indices (subscription service)

---

## Software and Libraries

### 8. Python Packages (Open Source)
All code uses open-source libraries under permissive licenses:

- **pandas** (2.0.3) - BSD License - Data manipulation
- **numpy** (1.24.3) - BSD License - Numerical computing
- **scikit-learn** (1.3.0) - BSD License - Machine learning
- **xgboost** (2.0.0) - Apache 2.0 License - Gradient boosting
- **matplotlib** (3.7.2) - PSF License - Visualization
- **seaborn** (0.12.2) - BSD License - Statistical visualization
- **joblib** (1.3.1) - BSD License - Model serialization
- **requests** (2.31.0) - Apache 2.0 License - API calls

Package citations available at respective documentation sites.

---

## Methodological Inspirations

### 9. Financial Machine Learning Literature

**General ML for Finance:**
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 7: Cross-validation in finance (temporal splits)
  - Chapter 10: Bet sizing and Kelly criterion applications

**Analyst Forecast Literature:**
- Brown, L. D., & Rozeff, M. S. (1978). "The Superiority of Analyst Forecasts as Measures of Expectations." *Journal of Finance*, 33(1), 1-16.
- Discusses analyst consensus as proxy for market expectations

**Earnings Announcement Effects:**
- Ball, R., & Brown, P. (1968). "An Empirical Evaluation of Accounting Income Numbers." *Journal of Accounting Research*, 6(2), 159-178.
- Seminal work on stock price reactions to earnings surprises

### 10. Prediction Markets Research

**Market Efficiency:**
- Wolfers, J., & Zitzewitz, E. (2004). "Prediction Markets." *Journal of Economic Perspectives*, 18(2), 107-126.
- DOI: 10.1257/0895330041371321
- General overview of prediction market mechanisms and efficiency

**Information Aggregation:**
- Surowiecki, J. (2004). *The Wisdom of Crowds*. Anchor Books.
- Popular treatment of how markets aggregate diverse information

---

## Style and Presentation Reference

### 11. Professional Report Formatting
**Inspiration:** User-provided document: `article_measuringthemoat.pdf`
- Used as style guide for academic prose formatting
- Influenced structure: section organization, paragraph flow, figure captions
- **No content was directly copied or paraphrased from this document**
- Our report presents entirely original analysis on different subject matter (earnings prediction vs. moat measurement)

---

## Original Contributions (No Citation Required)

The following elements represent **original work** developed for this project:

1. **Elo Rating Adaptation to Earnings:**
   - elo_momentum, elo_decay, elo_vol_4q metrics
   - K-factor calibration for earnings context (K=32)
   - Decay parameter tuning (λ=0.95)

2. **Global Cross-Company Model:**
   - Training single model across all S&P 500 companies
   - Demonstrated 82.65% accuracy vs. 66.43% for ticker-specific models

3. **Edge-Based Strategy Classification:**
   - "Sweet Spot" strategy (10-15% edge, 32.73% ROI)
   - Nine-strategy comparison framework
   - Kelly fraction optimization analysis

4. **Leak Prevention Protocol:**
   - 29 feature leak audit with explicit LEAK_COLS list
   - Temporal train/test split (35%/65%)
   - Feature timestamp validation procedures

5. **Polymarket Backtest Methodology:**
   - Integration of ML predictions with real market prices
   - Fee-adjusted P&L simulation (2% on winnings)
   - Price bucket analysis for calibration assessment

6. **All Code, Figures, and Tables:**
   - 5,000+ lines of original Python code
   - 21 custom visualizations
   - 7 formatted analysis tables
   - Complete data pipeline and feature engineering

---

## Attribution Statement

This report synthesizes established methodologies (Elo ratings, Kelly criterion, Random Forest/XGBoost models) with original feature engineering, data collection, and empirical analysis. All statistical results, backtests, and conclusions represent our independent research using publicly available data sources and open-source software.

Where we adapt existing frameworks (e.g., Elo for earnings, Kelly for position sizing), we acknowledge the original theoretical foundations while emphasizing that our specific implementations, parameter choices, and applications are novel.

---

## Recommended Citation for This Report

If referencing this work:

> [Your Name/Organization]. (2025). *Quantitative Earnings Prediction Strategy: Machine Learning Meets Prediction Markets*. Unpublished investment strategy report. [Document ID: QRD-2025-EARN-v1.0]

---

## Academic Integrity Statement

✅ **All analysis is original** - We collected, cleaned, and analyzed our own dataset  
✅ **All code is original** - Written from scratch using standard libraries  
✅ **All figures are original** - Generated from our data and analysis  
✅ **Theoretical foundations are cited** - Elo, Kelly, Brier, etc. properly attributed  
✅ **Data sources are documented** - Alpha Vantage, Polymarket, S&P 500 list  
✅ **Style inspiration acknowledged** - article_measuringthemoat.pdf for formatting only

❌ **No content plagiarism** - No text copied from academic papers, reports, or websites  
❌ **No data plagiarism** - All data obtained through legitimate API access with attribution  
❌ **No code plagiarism** - No copying from GitHub, Stack Overflow, or other sources without attribution

---

**Last Updated:** November 18, 2025  
**Version:** 1.0  
**Status:** Ready for academic/professional submission

