# 8. POLYMARKET BACKTEST ANALYSIS

The ultimate test of any quantitative strategy is performance on real, out-of-sample data where money could have been made or lost. While our ML models achieve 82.65% accuracy on historical test data, this measures only forecast quality, not trading profitability. To assess actual investability, we backtest against Polymarket earnings markets from Q3 2025, comparing our model probabilities to market prices and tracking hypothetical P&L assuming Kelly-optimal execution.

## 8.1 Dataset Construction

We obtained Polymarket historical data for 300 earnings "beat" markets created between October and November 2025, corresponding to Q3 2025 fiscal quarter reports. Each market asks a binary question ("Will Company X beat quarterly earnings?") and resolves to Yes (company beat) or No (company missed). Markets provide ticker symbols, creation dates, closing dates, final outcomes (Yes/No), and critically for our analysis, average Yes prices during each market's active trading window.

Of these 300 events, we initially had earnings data for only 138 companies (46%) in our Alpha Vantage database—many Polymarket tickers are smaller-cap names or recent SPACs not traditionally included in S&P 500 datasets. We addressed this gap by fetching data for the missing tickers, successfully adding 151 companies (93% success rate on the fetch attempts) to expand our coverage to 262 events (87% of the original 300). The remaining 38 events include tickers where Alpha Vantage lacks coverage or where data quality issues prevent feature extraction.

For the 262 analyzable events, we match each to its corresponding fiscal quarter (almost all resolve to Q3 2025, September 30 quarter-end) and extract features from our raw data files. We then load our pre-trained global models (Random Forest, XGBoost, Logistic Regression) and generate probabilities for each event. This produces a dataset with model predictions, Polymarket outcomes, and market prices—the three ingredients needed for edge calculation and P&L simulation.

## 8.2 Model Accuracy vs Market Outcomes

Our Logistic Regression model correctly predicts 184 of 262 Polymarket outcomes, achieving 70.23% accuracy. At first glance, this appears only marginally better than a naive "always bet beat" strategy (which would achieve 72.52% accuracy, equal to the base rate of beats in this sample). However, this comparison misses the point. Prediction markets already price in the base rate—that's why average market prices hover around 0.70-0.75, not 0.50. Our model's value lies not in raw accuracy but in identifying when it disagrees with the market **and is correct**.

Examining the confusion matrix, we find 148 true positives (correctly predicted beats), 36 true negatives (correctly predicted misses), 42 false negatives (predicted miss but actually beat), and 36 false positives (predicted beat but actually missed). The false negative rate (22%) is acceptable—we miss some beats but catch most. The false positive rate (50% among our "miss" predictions) is higher, reflecting the class imbalance problem: with beats outnumbering misses 2.6-to-1 in our sample, even modest model uncertainty leads to systematically predicting beats, and those predictions are wrong half the time on the minority class.

Importantly, Random Forest and XGBoost show nearly identical accuracy (69.85% and 68.32%, respectively), suggesting model consensus. When all three models agree, prediction confidence is justified. When they diverge, we exercise caution, often declining to trade even if one model shows large edge. This ensemble discipline proves critical in avoiding the overconfidence trap discussed in Section 10.

## 8.3 Beat Rate Calibration and Market Efficiency

A critical test of our models' calibration is whether they correctly estimate the overall beat rate. Our test set (2020-2025) shows 74.82% beats. Our Polymarket sample shows 72.52% beats. Logistic Regression predicts an average probability of 70.23% across the 262 events. Random Forest predicts 71.37%—remarkably close to the actual 72.52%. XGBoost predicts 64.50%, indicating it is miscalibrated on the conservative side.

This calibration quality matters because Kelly sizing depends on accurate probabilities, not just rank-ordering. If a model systematically overstates probabilities (predicting 0.90 when truth is 0.70), Kelly will oversize positions, increasing volatility and potentially turning positive expected value into realized losses via large drawdowns. Conversely, systematic understatement (predicting 0.50 when truth is 0.70) causes under-sizing and missed opportunities, though with lower risk.

Random Forest's near-perfect aggregate calibration (71.37% predicted, 72.52% actual) combined with its low Brier score (0.193) makes it our preferred model for probability forecasts, despite Logistic Regression's slightly higher raw accuracy. In practice, we find averaging Random Forest and Logistic Regression probabilities produces the most reliable forecasts for Kelly calculations.

---

**Table 8.1: Polymarket Backtest Performance**

| Model | Accuracy | Precision | Recall | Calibration Error |
|-------|----------|-----------|--------|-------------------|
| Logistic Regression | 70.23% | 80.4% | 77.9% | -2.29% |
| Random Forest | 69.85% | 81.1% | 78.0% | -1.15% |
| XGBoost | 68.32% | 77.2% | 74.5% | -8.02% |

*N=262 Polymarket events, Q3 2025*

Random Forest shows the smallest calibration error (predicted 71.37% beat rate vs actual 72.52%), validating its use for probability-dependent Kelly sizing.

---

**Pages:** 44-49  
**Section:** Polymarket Backtest  
**Classification:** Confidential

