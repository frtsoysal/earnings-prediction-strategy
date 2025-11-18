# 7. RESEARCH FINDINGS: ANALYST CONSENSUS DYNAMICS

Beyond strategy development, our dataset enables empirical research on analyst behavior and consensus formation. We investigate three questions with implications for both academic understanding of analyst forecasting and practical signal construction for quantitative strategies.

## 7.1 Consensus Uncertainty and Beat Probability

Our first research question concerns the relationship between analyst disagreement and earnings outcomes. When analysts' estimates span a wide range—say, $1.50 to $2.50 for a stock with $2.00 consensus—it signals genuine uncertainty about business fundamentals, competitive dynamics, or macroeconomic sensitivity. Under efficient markets, such uncertainty might be already priced into the stock, leaving beat probability at the unconditional mean. Alternatively, under behavioral theories, high uncertainty might create conservative positioning by management (lowering guidance to avoid misses) or conservative analyst estimates (fear of being an outlier), actually increasing beat probability.

Our empirical findings decisively reject both neutrality and the behavioral upside hypothesis, instead supporting a downside relationship. We partition our 13,856 observations into quintiles based on estimate spread percentage: (High - Low) / |Average| × 100. Q1 (lowest spread, averaging 4.78%) shows 78.42% beat rate. Q5 (highest spread, averaging 124.83%) shows only 63.30% beat rate. The monotonic decline across quintiles (78.42%, 75.61%, 74.63%, 72.57%, 63.30%) is statistically highly significant (χ²=186.80 with 4 degrees of freedom, p<0.0001) and economically substantial—a 15.1 percentage point range.

The mechanism appears to be genuine fundamental uncertainty rather than behavioral bias. Companies with high estimate spreads disproportionately include those undergoing business model transitions, facing regulatory uncertainties, or operating in cyclical industries with volatile earnings. These genuine headwinds make beating estimates objectively harder, explaining the lower success rates. For our strategy, this finding is immediately actionable: we avoid high-spread earnings or, when we cannot avoid them (some Sweet Spot edges occur despite high spread), we size positions cautiously.

## 7.2 Optimal Analyst Coverage

Our second research question examines whether more analyst coverage improves forecast accuracy and thus beat predictability. One might hypothesize a monotonic relationship: more analysts → better information aggregation → tighter estimates → higher beat rates (if companies exploit tight estimates to guide conservatively). Alternatively, more coverage might mean more scrutiny and tougher hurdles, reducing beat rates.

Neither monotonic story holds. Instead, we document a U-shaped relationship. Companies with <10 analysts beat 67.57% of the time. Coverage of 10-20 analysts shows peak performance at 74.98% beat rate—our "optimal" range. Coverage of 20-30 analysts drops slightly to 73.31%, while 30+ analyst coverage falls to 69.40%. The pattern suggests a Goldilocks zone: sufficient information for reasonable forecasts (10-20 analysts) without the hyper-efficiency that arises when dozens of research teams scrutinize every financial detail.

The statistical significance is marginal (Spearman ρ=0.045, p<0.0001)—coverage matters, but weakly and non-linearly. From a practical standpoint, we note that most Sweet Spot opportunities (93% win rate in backtest) involve companies in the 15-25 analyst range, possibly because these mid-coverage companies offer the best balance of predictability and market inefficiency.

## 7.3 Estimate Revision Momentum

Perhaps our strongest research finding concerns analyst estimate revisions. We construct Revision_Momentum = Revisions_Up_30d - Revisions_Down_30d, counting how many analysts raised versus lowered their estimates in the month before earnings. Strong positive momentum (+5.8 average) corresponds to 78.69% beat rate; strong negative momentum (-4.6 average) corresponds to 66.67% beat rate—a 12.0 percentage point spread that is both statistically significant (Spearman ρ=0.088, p<0.0001) and larger than most other effects we measure.

The interpretation is that revisions contain incremental information not yet fully reflected in the consensus estimate level. When multiple analysts raise forecasts, it signals improving fundamentals or positive earnings preannouncements that haven't fully diffused. Yet the consensus estimate itself—the average of all forecasts—adjusts slowly, particularly if only a subset of analysts have updated. This creates a window where revision momentum predicts beats even after controlling for the estimate level. As more research teams update, momentum declines and the signal fades, but the initial divergence creates tradeable edge.

For our models, revision momentum enters both directly (as a feature) and indirectly (correlated with Elo momentum, since beats following positive revisions boost Elo). The combined effect makes recent estimate dynamics one of our most reliable signals, and we prioritize trades where both revision momentum and Elo momentum align positively.

---

**Figure 7.1: Consensus Spread Effect**  
*See: final_report/figures/fig4_consensus_spread.png*

**Table 7.1: Consensus Spread Quintile Analysis**  
*See: final_report/tables/table3_consensus_spread.txt*

---

**Pages:** 38-43  
**Section:** Research Findings  
**Classification:** Confidential

