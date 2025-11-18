# 9. EDGE & EXPECTED VALUE ANALYSIS

The transition from model accuracy to trading profitability requires careful analysis of the **edge**—the differential between our model's assessed probability and the market's implied probability (reflected in Polymarket prices). A model can be highly accurate yet unprofitable if it agrees with the market on every outcome. Conversely, a model with modest accuracy can generate significant returns if its errors are uncorrelated with market prices and its correct divergences are systematically directional.

## 9.1 Edge Definition and Measurement

For each of our 248 Polymarket events with available pricing data, we calculate Edge_LR = p_LR - q_PM, where p_LR is our Logistic Regression model's predicted probability of a beat and q_PM is Polymarket's average Yes price during the market's lifetime. Positive edge indicates our model is more bullish than the market; negative edge indicates the market is more bullish than our model.

The distribution of edges is revealing. Mean edge is -9.72%, suggesting Polymarket participants are on average slightly more optimistic than our model. This systematic bias likely reflects the platform's user base—retail traders who may overweight recent positive earnings trends or exhibit "hope" bias favoring beats over misses. Median edge is -8.15%, confirming the distribution is roughly symmetric around a negative center. Standard deviation is substantial at 26.35%, indicating wide variation from event to event—some with +60% edge (model extremely bullish), others with -85% edge (model extremely bearish).

Critically, 90 of 248 events (36.3%) show positive edge, meaning our model sees more upside than the market prices in. These 90 events are our candidate trading opportunities. The remaining 158 events have negative edge, and our strategy excludes them (or, as we test in Section 10, attempts to bet against them via "No" contracts, though this proves unprofitable in practice).

## 9.2 Edge Buckets and Performance Patterns

To understand which edges are genuine versus noise, we partition events into buckets: <-20%, -20 to -15%, ..., +15 to +20%, >+20%. The pattern that emerges is non-linear and striking. At negative extremes (<-20% edge), actual beat rate is 65.28%—the market's optimism is partially justified, though our model is still wrong to be so pessimistic. At moderate negative edges (-15 to -10%), beat rate jumps to 80.77%, contradicting our model and suggesting calibration issues in this range.

Near zero edge (-5% to +5%), beat rates cluster around 70-86%, with our model accuracy in the 67-86% range—respectable but not exceptional. The key finding appears at +10 to +15% edge: here, actual beat rate is 93.75% (15 of 16 events), far exceeding both our model's predicted ~78% and the market's implied ~65%. This is the "Sweet Spot"—a region where systematic market underpricing creates exploitable opportunities.

Paradoxically, edges exceeding +20% perform poorly: only 57.69% beat rate (15 of 26 events). This suggests model overconfidence. When our Logistic Regression assigns probabilities above 0.85-0.90, and markets trade at 0.50-0.60, the divergence often reflects model error rather than market mispricing. We speculate this occurs because extreme probabilities saturate our models' output ranges, or because the events triggering such extremes involve data quality issues or atypical circumstances the model hasn't encountered in training.

## 9.3 Calibration Quality and Brier Scores

Brier score—the mean squared error between predicted probabilities and binary outcomes—provides a formal measure of calibration. A Brier score of 0.00 represents perfect prophecy; 0.25 represents random guessing (for a 50/50 base rate); and higher scores indicate worse-than-random forecasts. Our Random Forest achieves Brier 0.1928, Logistic Regression 0.2153, and XGBoost 0.2271—all substantially better than the 0.25 baseline and competitive with published benchmarks in probability forecasting literature.

Random Forest's superior Brier score reflects better calibration: its probabilities are not just rank-ordered correctly but numerically accurate. When RF says 0.75, actual outcomes are beats approximately 75% of the time across that probability bucket. This property is essential for Kelly sizing, which requires knowing not just which event is more likely to beat but by how much. Our calibration curve analysis (Figure 9.1) visualizes this: RF probabilities cluster tightly around the diagonal (perfect calibration line), while XGB shows some deviation at the extremes.

---

**Figure 9.1: Probability Calibration Curves**

The diagonal represents perfect calibration. Random Forest (blue) adheres most closely, validating its use for Kelly bet sizing. XGBoost (purple) and Logistic Regression (green) show moderate deviations but remain well-calibrated overall.

---

**Pages:** 50-56  
**Section:** Edge & Expected Value  
**Classification:** Confidential

