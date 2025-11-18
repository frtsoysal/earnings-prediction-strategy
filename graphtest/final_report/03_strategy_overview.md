# 2. STRATEGY OVERVIEW

## 2.1 Investment Thesis

Our investment thesis rests on a fundamental market inefficiency: prediction markets for corporate earnings, while theoretically efficient aggregators of dispersed information, systematically misprice certain events in predictable ways. Specifically, we observe that Polymarket participants appear to underweight the informational content of (a) historical earnings performance trajectories, (b) recent analyst estimate revision patterns, and (c) consensus uncertainty metrics. When our machine learning models identify significant divergences between our calculated probability and the market-implied probability, we have an edge—and where there is edge, properly sized with Kelly criterion, there is expected value.

The theoretical foundation draws from three bodies of literature. First, the behavioral finance literature on analyst forecasting biases suggests that consensus estimates systematically lag fundamental changes, creating predictable patterns in beats and misses. Second, the market microstructure literature on information aggregation suggests that thinly traded or nascent markets (like Polymarket for individual earnings events) may not fully incorporate all available signals. Third, the quantitative finance literature on sports betting and Kelly criterion provides the mathematical framework for optimal position sizing when we have a probabilistic edge.

Our empirical contribution is demonstrating that these theoretical inefficiencies exist, persist, and are exploitable at scale across a large cross-section of companies and time periods.

## 2.2 Core Hypotheses

We advance four testable hypotheses that form the backbone of our strategy.

**Hypothesis 1 (Elo Momentum):** Companies exhibiting positive momentum in their Elo ratings—meaning they have recently outperformed analyst expectations by increasing margins—are significantly more likely to beat estimates in the current quarter than companies with declining or stable Elo trajectories. This reflects both genuine business momentum (improving fundamentals persist across quarters) and behavioral factors (management teams that beat once often guide conservatively to beat again).

**Hypothesis 2 (Consensus Uncertainty):** Higher dispersion among analyst estimates—measured as the percentage spread between the highest and lowest forecasts—inversely predicts beat probability. Wide spreads indicate information asymmetry or uncertainty about business fundamentals, both of which correlate with increased miss risk. Conversely, tight consensus suggests clarity and confidence, often preceding beats.

**Hypothesis 3 (Revision Momentum):** The net change in analyst estimate revisions over the prior 30 days—upgrades minus downgrades—predicts beats better than the absolute level of estimates. Positive revision momentum indicates analysts are chasing improving fundamentals, typically lagging the actual improvement and thus setting up beats. Negative momentum indicates deteriorating confidence and foreshadows misses.

**Hypothesis 4 (Edge Sweet Spot):** When our model probability diverges from the Polymarket price by 10-15%, we have identified a systematic mispricing. Divergences below 10% represent noise or fair pricing. Divergences above 20% represent model overconfidence (the model is likely wrong when it disagrees too strongly with the crowd). The 10-15% band is the "Sweet Spot" where our models' incremental information genuinely improves on market consensus.

Each of these hypotheses receives empirical validation in subsequent sections, with statistical tests rejecting nulls at p<0.001 significance levels.

## 2.3 Data Architecture

Our data infrastructure comprises three primary layers. The **acquisition layer** pulls from Alpha Vantage's premium API, fetching earnings estimates, actuals, financial statements, and price histories for 498 S&P 500 constituents. This process runs weekly, updating our database with the latest analyst revisions and ensuring we have fresh features before each earnings season. The **feature engineering layer** transforms raw API responses into the 31 leak-safe features our models consume, including Elo rating calculations, lagged growth metrics, and consensus statistics. The **model layer** maintains trained Random Forest, XGBoost, and Logistic Regression classifiers, along with preprocessing pipelines (imputation and standardization), persisted as serialized objects for fast inference.

Critically, our architecture enforces strict temporal ordering. Features for quarter Q are computed using only information available before Q's earnings announcement. This prevents lookahead bias—a common pitfall in financial machine learning where researchers inadvertently use future information (like post-announcement stock moves) to predict the past. Our automated leak detection tests verify this property before any model deployment.

## 2.4 Expected Value Framework

The mathematical foundation of our trading strategy is the Kelly criterion for optimal bet sizing in scenarios with a probabilistic edge. Let p represent our model's estimated probability that a company beats estimates, and q represent the Polymarket price (equivalently, the market-implied probability). Our **edge** is simply e = p - q. If p > q, we have positive edge—our model sees more upside than the market prices in. If p < q, we have negative edge—the market is more bullish than we are, and we should not trade (or, theoretically, bet against, though our backtest shows counter-strategies perform poorly).

Given positive edge, the Kelly criterion prescribes an optimal bet size as a fraction of our capital. For a binary outcome contract like Polymarket where winners pay $1.00 per share, the Kelly optimal fraction is f* = e / (1 - q). For example, if our model assigns p=0.75 to a beat, and Polymarket trades at q=0.60, our edge is e=0.15 and Kelly says invest f*=0.15/(1-0.60)=0.375, or 37.5% of capital, in that single event.

Full Kelly sizing, while mathematically optimal for log-wealth growth, is notoriously volatile and can lead to large drawdowns. Standard practice in quantitative finance is to use fractional Kelly—typically 10% to 50% of the full Kelly allocation. We employ quarter-Kelly (25%) throughout this analysis, a conservative choice that significantly reduces risk while capturing most of the edge. Our empirical tests show that ROI is invariant to Kelly fraction (the edge determines returns), but capital requirement and volatility scale with aggression, making 25% an attractive risk-return balance.

Expected value for a given trade is calculated as EV = p × Profit_if_Win - (1-p) × Loss_if_Lose, where profit equals the payout minus stake and loss equals the stake. Kelly sizing ensures that even when individual trades lose (as 7-30% will, depending on our hit rate), the aggregate expected value across a portfolio of properly sized bets remains positive and compounds over time.

---

**Figure 2.1: Kelly Criterion Framework**  
*See: final_report/figures/fig7_kelly_optimization.png*

This figure illustrates how ROI remains constant across Kelly fractions while capital requirements scale linearly—a key insight supporting our 25% Kelly recommendation.

---

**Pages:** 9-12  
**Section:** Strategy Overview  
**Classification:** Confidential

