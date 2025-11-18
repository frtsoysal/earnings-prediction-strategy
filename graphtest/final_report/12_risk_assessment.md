# 11. RISK ASSESSMENT & LIMITATIONS

No investment strategy is without risk, and responsible disclosure requires honest assessment of limitations, failure modes, and scenarios under which our approach might underperform or lose capital. This section catalogs material risks across statistical, market, operational, and regulatory dimensions.

## 11.1 Statistical and Sample Size Considerations

Our Polymarket backtest comprises 262 events, of which our Sweet Spot strategy trades just 16. While statistically significant (binomial p=0.002 against a 50% null), a 16-trade sample leaves considerable uncertainty around true expected returns. The 95% confidence interval on our 93.75% observed win rate spans approximately [68%, 99%]—wide enough that unlucky variance could produce materially lower realized performance over the next 16 trades. We address this limitation by recommending no institutional scale-up beyond $50,000 allocated capital until we accumulate 40-60 Sweet Spot trades across multiple earnings seasons (Q4 2025, Q1 2026, Q2 2026), which will narrow confidence intervals and validate consistency.

Additionally, all 262 backtest events correspond to a single fiscal quarter (Q3 2025, September 30 quarter-end). While our ML model trained on data spanning all four fiscal quarters (March, June, September, December), our live validation has seen only September quarters. Seasonal effects—tax considerations, holiday retail patterns, fiscal year-end dynamics—could theoretically affect beat rates or edge distributions in ways our backtest hasn't captured. We mitigate this by planning multi-quarter validation before drawing definitive conclusions about strategy robustness.

Finally, our training period (2011-2020) and test period (2020-2025) overlap with the COVID-19 pandemic, Federal Reserve policy shifts, and significant macroeconomic regime changes. If the 2020-2025 test period represents an unusual regime (unusually high beat rates, unusually profitable Polymarket mispricings), forward performance may disappoint. Quarterly recalibration and continuous performance monitoring can detect such regime changes early, allowing strategy suspension or modification before significant losses accumulate.

## 11.2 Concentration and Diversification Risk

Our strategy concentrates in three dimensions that prudent risk management would ordinarily diversify. First, geographic concentration: all companies are S&P 500 constituents, meaning U.S.-domiciled large-caps. Second, temporal concentration: earnings seasons are quarterly events, leaving capital idle ~75% of the year between major waves. Third, platform concentration: Polymarket is our sole trading venue, exposing us to platform-specific risks (outages, regulatory restrictions, rule changes).

Standard portfolio theory suggests diversifying across geographies (international earnings), asset classes (combining earnings bets with other strategies), and platforms (other prediction markets or related derivatives). However, over-diversification dilutes edge. If our specific edge derives from Polymarket's user base exhibiting particular biases on S&P 500 earnings, expanding to European equities or different platforms might sacrifice profitability for diversification benefits we don't need (given low correlation to traditional equity returns and event-driven nature limiting systemic risk).

We recommend moderate diversification: deploying the strategy as one component of a broader quantitative portfolio rather than a standalone fund, maintaining position size limits to prevent single-event catastrophic loss, and exploring comparable platforms (Kalshi, PredictIt if available) once Polymarket track record is established. But we stop short of advocating geographic or asset class diversification that would dilute focus on our demonstrated edge.

## 11.3 Model Limitations and Failure Modes

Our models exhibit several documented failure modes. First, overconfidence at extreme probabilities: when LR assigns >0.85 probability, actual win rates drop to 60-70%, not the 85%+ the probability implies. This calibration breakdown at the tails likely reflects sparse training data in extreme regions and model saturation. Second, poor miss detection: only 50% specificity (true negative rate) suggests our models struggle to identify misses, possibly due to training on a 74.82% beat base rate that leaves misses as the minority class. Third, company-type bias: our models may underperform on recent IPOs, SPACs, or micro-caps where historical Elo patterns are absent or unreliable—though our global model's ability to generalize across companies mitigates this somewhat.

Perhaps most concerning is the potential for strategy decay. As machine learning proliferates in finance and more participants adopt similar approaches, edges compress. Polymarket, while nascent now, could attract institutional capital or sophisticated algorithms that eliminate the mispricings we exploit. We estimate a strategy half-life of 12-18 months, after which returns may decline to zero or turn negative as markets efficiently incorporate the signals we've identified. Continuous performance monitoring and quarterly recalibration are essential to detect decay early and adjust or retire the strategy before sustained losses occur.

---

**Pages:** 65-70  
**Section:** Risk Assessment  
**Classification:** Confidential

