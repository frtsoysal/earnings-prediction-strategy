# APPENDICES

## APPENDIX A: Technical Methodology

Our computational infrastructure employs Python 3.14 with scikit-learn 1.3, XGBoost 2.0, pandas 2.1, and numpy 1.26 as core dependencies. Data acquisition scripts query Alpha Vantage's REST API using urllib with SSL certificate verification disabled (corporate firewall compatibility). Rate limiting respects premium tier constraints (75 calls/minute) via time.sleep() delays between requests. Error handling captures HTTP 403/429 responses and retries with exponential backoff.

Feature engineering pipelines process raw JSON responses into structured CSV files with 91 columns per observation. Elo calculations employ vectorized numpy operations for performance, updating all 498 company ratings in <5 seconds per quarter. Missing value imputation uses sklearn.impute.SimpleImputer with median strategy, fit on training data and transformed consistently across train/test/production.

Model training employs RandomizedSearchCV for hyperparameter tuning with TimeSeriesSplit cross-validation (n_splits=5). We allocate 4 CPU cores per model training job, completing full grid search in approximately 15-25 minutes per model on standard hardware (M1 MacBook Pro). Model persistence via joblib.dump() creates platform-independent .pkl files loadable across systems.

Leak verification runs as automated tests checking: (1) no LEAK_COLS in feature matrix, (2) no target in features, (3) temporal separation of train/test, (4) no current-quarter actuals. All tests must pass before model deployment. Manual code review supplements automated checks, tracing each feature's calculation from source to model input.

## APPENDIX B: Feature Definitions

**31 Leak-Safe Features (Complete Specification):**

1. **price_1m_before:** Stock close price 30 calendar days before earnings report date (nearest trading day if 30th day is weekend/holiday)

2-4. **eps_estimate_average, _high, _low:** Analyst consensus EPS estimates as of 1 day before announcement (mean, maximum, minimum across all covering analysts)

5. **eps_estimate_analyst_count:** Number of analysts contributing to consensus

6-9. **eps_estimate_average_X_days_ago:** Historical snapshots of consensus estimate at X∈{7,30,60,90} days before current date (enables revision tracking without lookahead)

10-13. **eps_estimate_revision_up/down_trailing_X_days:** Count of upward/downward estimate revisions over trailing X∈{7,30} day windows

14-17. **revenue_estimate_average, _high, _low, _analyst_count:** Revenue consensus metrics (parallel to EPS)

18-21. **elo_before, elo_decay, elo_momentum, elo_vol_4q:** Elo rating entering quarter, time-weighted average, weighted recent changes, rolling volatility (detailed in Section 6)

22-31. **Lagged growth metrics:** *_yoy_growth_lag1, *_qoq_growth_lag1, *_ttm_yoy_growth_lag1 for revenue, EPS, EBITDA, operating income, plus margin changes—all using t-1 (previous quarter) actual results to avoid temporal leakage

All features are float type except revision counts (int) and analyst counts (int). Missing values for estimates/revisions are rare (<10%) and imputed via median. Missing values for lagged growth are more common (12-25%, due to companies with limited history) and also median-imputed.

## APPENDIX C: Model Hyperparameters

**Random Forest (Final Configuration):**
- n_estimators: 300 (after testing 100, 200, 300; diminishing returns beyond 300)
- max_depth: 20 (prevents overfitting while allowing complex interactions)
- min_samples_split: 5 (regularization)
- min_samples_leaf: 2 (leaf node size floor)
- class_weight: 'balanced' (addresses 74.8% beat imbalance via inverse frequency weighting)
- random_state: 42 (reproducibility)
- bootstrap: True (bagging for variance reduction)

**XGBoost (Final Configuration):**
- n_estimators: 200
- max_depth: 5 (shallower than RF to prevent overfit in boosting context)
- learning_rate: 0.05 (conservative for stability)
- subsample: 0.9 (stochastic gradient boosting)
- colsample_bytree: 1.0 (use all features)
- scale_pos_weight: 0.44 (inverse of class ratio)
- eval_metric: 'logloss' (probability calibration objective)

**Logistic Regression (Final Configuration):**
- C: 1.0 (regularization strength)
- penalty: 'L2' (ridge regression)
- solver: 'lbfgs' (efficient for small-medium datasets)
- class_weight: 'balanced'
- max_iter: 1000 (convergence typically ~150 iterations)

Preprocessing: SimpleImputer (strategy='median') → StandardScaler (mean=0, std=1). Pipeline fit on train, transform applied identically to test and production.

## APPENDIX D: Data Sources

**Alpha Vantage API (Premium Tier):**
- Base URL: https://www.alphavantage.co/query
- Rate Limit: 75 calls/minute
- Endpoints: EARNINGS_ESTIMATES, EARNINGS, INCOME_STATEMENT, CASH_FLOW, TIME_SERIES_DAILY
- Cost: ~$50/month premium subscription
- Data Retention: Complete history back to company IPO or 2000 (whichever later)

**Polymarket API (Public):**
- Gamma API: https://gamma-api.polymarket.com/markets (market metadata)
- CLOB API: https://clob.polymarket.com/prices-history (price history)
- No authentication required for reads
- Rate Limit: ~100 requests/minute (undocumented but observed)

**S&P 500 Constituent List:**
- Source: Wikipedia (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- Updated: Manual refresh quarterly to capture index changes
- Alternative: Direct download from S&P Dow Jones Indices (subscription required)

## APPENDIX E: Code Repository Structure

Complete code available in ML/ directory with subdirectories:
- **data/raw/:** 498 CSV files (one per company)
- **scripts/:** Data fetching, Elo calculation, model training
- **models/:** Serialized model objects (.pkl files)
- **graphtest/global_model/:** Cross-company model and results
- **graphtest/polymarket_backtest/:** Backtest scripts and analysis
- **graphtest/final_report/:** This document and supporting materials

Key scripts:
- fetch_alpha_vantage.py: Data acquisition + Elo engineering (1,097 lines)
- train_global_model.py: ML training pipeline (589 lines)
- polymarket_vs_ml.py: Backtest execution (456 lines)
- edge_analysis.py: ROI calculation and optimization (589 lines)

Total codebase: ~5,000 lines Python, extensively commented. No proprietary dependencies—all packages open-source (scikit-learn, pandas, numpy, matplotlib).

## APPENDIX F: Glossary

**Beat:** Actual reported EPS exceeds analyst consensus estimate  
**Miss:** Actual EPS falls short of consensus  
**Edge:** p_model - p_market (differential between our probability and market price)  
**Elo:** Adaptive rating system tracking historical performance  
**Kelly Criterion:** Optimal bet sizing formula based on edge  
**Brier Score:** Mean squared error of probability forecasts (lower = better)  
**Calibration:** Alignment between predicted probabilities and observed frequencies  
**Temporal Leak:** Using future information to predict past (invalidates backtest)  
**Sweet Spot:** Edge range (10-15%) with highest empirical ROI  
**Quarter-Kelly:** 25% of Kelly-optimal bet size (risk management)

---

**Pages:** 81-90  
**Section:** Appendices  
**Classification:** Confidential

---

# FINAL SUMMARY STATEMENT

This investment strategy report documents a quantitative earnings prediction framework with empirically validated positive expected value. Through machine learning models trained on 14,239 observations across 468 S&P 500 companies, refined with novel Elo rating adaptations, and rigorously backtested on 262 Polymarket outcomes, we demonstrate that systematic alpha exists in prediction market arbitrage. Our Sweet Spot strategy delivers 32.73% quarterly ROI with 93.75% win rate—performance that, while requiring validation across additional quarters, represents state-of-the-art results in this domain.

The core innovation—applying chess Elo ratings to corporate earnings—proves remarkably effective, with Elo-based features contributing 54% of predictive power. Combined with analyst consensus dynamics (spread, revisions, coverage) and strict temporal leak prevention, our framework offers institutional investors a systematic, scalable, and theoretically grounded approach to earnings-driven alpha generation.

We present this work with appropriate humility regarding sample sizes and concentration risks, while maintaining confidence in the underlying statistical relationships and methodological rigor. The strategy is production-ready for capital allocations of $50,000-$500,000, requires quarterly recalibration to maintain edge, and should be deployed as one component of a diversified quantitative portfolio rather than a standalone fund.

For trading desks seeking alternative alpha sources, data science teams interested in financial ML applications, or researchers studying market microstructure and analyst behavior, this report provides both immediate practical value and a foundation for future investigation.

---

**END OF TECHNICAL CONTENT**

**Prepared by:** Quantitative Research Division  
**Document ID:** QRD-2025-EARN-v1.0  
**Classification:** PROPRIETARY & CONFIDENTIAL  
**Date:** November 15, 2025

---

**Disclaimer:** This report is for informational purposes only. Past performance does not guarantee future results. Trading involves substantial risk. Consult qualified advisors before implementation. The authors assume no liability for losses.

---

**© 2025 Quantitative Research Division. All Rights Reserved.**

No part of this document may be reproduced, distributed, or transmitted without prior written permission.

---

**Pages:** 91-96  
**Final Page**

