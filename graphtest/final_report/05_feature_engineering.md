# 4. FEATURE ENGINEERING FRAMEWORK

## 4.1 Elo Rating System for Corporate Earnings

The most innovative component of our feature engineering is the adaptation of Arpad Elo's rating system from competitive chess to corporate earnings prediction. Developed in the 1960s, the Elo system assigns each player a numeric rating that evolves based on game outcomes relative to expectations. Elo's elegance lies in its simplicity: beat a higher-rated opponent and your rating rises substantially; beat a lower-rated opponent and it rises modestly; lose and it declines proportionally. Over time, ratings converge to reflect true skill levels, and the system's expected outcome probabilities prove remarkably accurate.

We adapt this framework by treating each company as a "player" and each earnings announcement as a "game" against analyst consensus. A beat represents a win; a miss represents a loss. The magnitude of the surprise—actual EPS minus estimated EPS—determines the rating change intensity. Companies that consistently beat estimates accumulate high Elo ratings, signaling robust earnings quality and conservative guidance. Companies that frequently miss see their ratings decline, flagging execution challenges or overly optimistic analyst coverage.

The mathematical update formula is: Elo_new = Elo_old + K × Surprise × 100, where Surprise = (Actual_EPS - Estimate_EPS) and K is an adaptive sensitivity parameter. We calibrate K based on analyst coverage: companies with 20+ analysts receive K×0.7 (more stable, well-understood), while those with fewer than 10 analysts receive K×1.5 (higher volatility, less information). This adaptive mechanism accounts for information quality differences across the market cap spectrum.

From the base Elo rating, we derive four critical metrics that enter our models. **elo_before** is simply the rating entering the current quarter—a level indicator. **elo_momentum** is a weighted average of the last four rating changes, with recent quarters weighted more heavily (0.4, 0.3, 0.2, 0.1)—a trend indicator. **elo_decay** applies the same weights to rating levels rather than changes, creating a time-discounted performance measure. **elo_vol_4q** computes the standard deviation of recent rating changes, quantifying consistency versus erratic performance. Together, these four features account for 53.98% of our Random Forest model's predictive power, validating that historical performance patterns dominate all other signals.

## 4.2 Analyst Consensus Metrics

Beyond Elo, our second major feature category captures analyst consensus dynamics. The **estimate spread**—defined as (High - Low) / |Average| × 100—measures disagreement among analysts. We find this metric inversely predicts beats: tight spreads correlate with 78% beat rates while wide spreads correlate with 63% beat rates, a statistically robust effect (p<0.0001) detailed in Section 7. The economic interpretation is straightforward: when analysts disagree, the company faces genuine uncertainty (business model transitions, competitive threats, macroeconomic sensitivity), making positive surprises less likely.

The **revision momentum** feature tracks the net change in estimate upgrades versus downgrades over the trailing 30 days. Analysts update their models continuously as new information arrives—competitor results, management commentary, industry data. We construct Revision_Momentum = Upgrades_30d - Downgrades_30d, finding that positive momentum predicts beats with a 12 percentage point spread (79% beat rate for strong positive momentum versus 67% for strong negative). This effect is incremental to the absolute estimate level, suggesting the rate of change in analyst opinion contains information not yet fully reflected in the consensus number itself.

We also include the raw estimate levels (average, high, low) and analyst count, though these prove less predictive than the dynamic metrics. Interestingly, analyst coverage shows a non-linear effect: companies with 10-20 analysts beat 75% of the time (optimal), while both under-covered (<10) and over-covered (30+) companies underperform. This U-shaped relationship, documented in Table 7.2, suggests a "Goldilocks" zone where information is sufficient but not so abundant that markets become hyper-efficient.

## 4.3 Momentum and Growth Indicators

Our third feature category comprises lagged financial metrics—growth rates and margin changes from the previous quarter. The "lagged" qualifier is critical: we use only t-1 (prior quarter) values, never t (current quarter, which would require already knowing the actual results we're trying to predict). For example, actual_eps_yoy_growth_lag1 represents the year-over-year EPS growth rate from the previous quarter's actual reported results, which is public information when we're forecasting the current quarter.

These lagged features—total_revenue_yoy_growth_lag1, ebitda_yoy_growth_lag1, operating_margin_yoy_change_lag1, and others—capture business momentum. A company that grew revenue 15% last quarter is more likely to beat estimates this quarter than one that declined 5%, all else equal. We include year-over-year, quarter-over-quarter, and trailing-twelve-month variants to capture different temporal patterns. Collectively, these growth features contribute approximately 15-20% of model importance, a meaningful but secondary role compared to Elo metrics.

We explicitly exclude current-quarter growth metrics from our feature set. While tempting to include (they would boost in-sample accuracy dramatically), doing so would constitute temporal leakage—using information from the future to predict the past. Our automated leak detection scripts verify that no such features appear in our final model inputs.

## 4.4 Temporal Leak Prevention Protocol

Financial machine learning is fraught with subtle temporal leakage risks that can produce misleadingly high backtested performance that collapses in live trading. We implement a multi-layered defense against such errors. First, we maintain a comprehensive exclusion list of features known to contain post-announcement information: actual_eps, eps_delta, elo_after, elo_change, price_at_report, price_change_1m_pct, and all current-quarter growth metrics without lag suffixes.

Second, we conduct automated tests before model training. Our verification script asserts that no excluded features appear in the training matrix, that the target variable (eps_beat) is not included as a feature, and that all test set dates strictly exceed all training set dates with no overlap. These tests run as part of our continuous integration pipeline and block deployment if any fail.

Third, we implement manual code review focusing on the data pipeline. Each feature's calculation is traced from raw API response to final model input, documenting the information timestamp. For example, eps_estimate_average_30_days_ago explicitly uses estimates from 30 days before the current date, ensuring it represents information a trader could have accessed in real-time. Our leak verification report (Appendix A) documents these checks for all 31 features.

The importance of this discipline cannot be overstated. Early in our research, we inadvertently included price_change_1m_pct in our feature set, believing it represented price movement in the month before the earnings announcement. Closer inspection revealed it actually calculated (price_at_report - price_1m_before) / price_1m_before—using the report day price, which is unknown at prediction time. After removing this and similar leakage sources, our model accuracy dropped from an illusory 95% to a realistic 83%, but the latter figure represents true out-of-sample performance we can expect in deployment.

---

**Figure 4.1: Elo Feature Importance**  
*See: final_report/figures/fig3_feature_importance.png*

Elo-based features (momentum, before, decay, volatility) comprise 54% of total predictive power in our Random Forest model—an unprecedented concentration that speaks to the dominance of historical performance patterns in predicting future earnings outcomes.

---

**Pages:** 18-24  
**Section:** Feature Engineering  
**Classification:** Confidential

