# 13. CONCLUSIONS & STRATEGIC RECOMMENDATIONS

## 13.1 Summary of Findings

This research demonstrates that systematic, positive expected value exists at the intersection of machine learning, corporate earnings prediction, and decentralized prediction markets. Our findings span empirical, methodological, and practical dimensions.

Empirically, we document that historical earnings performance—quantified via adapted Elo ratings—predicts future beats with greater power than traditional financial metrics, analyst estimates, or price momentum. Elo-based features account for 54% of our models' predictive capacity, with elo_momentum alone contributing 36%. This dominance validates our hypothesis that earnings performance exhibits path-dependent, mean-reverting, and momentum-driven dynamics analogous to competitive games. Companies on improving trajectories tend to continue improving (at least for several quarters), while those on deteriorating paths continue deteriorating, creating exploitable patterns.

We further document robust relationships between analyst consensus characteristics and beat probability. Consensus uncertainty (estimate spread) inversely predicts beats, with a 15.1 percentage point spread between tight and wide consensus (χ²=186.80, p<0.0001). Estimate revision momentum positively predicts beats, with a 12.0 percentage point spread between strong positive and strong negative momentum (Spearman ρ=0.088, p<0.0001). These effects are statistically significant, economically meaningful, and incremental to Elo—collectively they enhance our models beyond what historical performance alone achieves.

Methodologically, we contribute a rigorous framework for temporal leak prevention in financial ML, an often-neglected aspect of research that can fatally compromise real-world performance. Our multi-layered approach—exclusion lists, automated tests, manual audits—ensures all features represent point-in-time available information. We also pioneer (to our knowledge) the first application of Elo ratings to corporate earnings, opening avenues for future research on other corporate events.

Practically, our Polymarket backtest validates profitability. Our Sweet Spot strategy (10-15% edge, quarter-Kelly sizing) achieves 32.73% quarterly ROI with 93.75% win rate across 16 trades. While sample size is limited, statistical tests reject randomness (p=0.002), and the consistency of results across edge buckets (positive edges generally profitable, negative and extreme edges unprofitable) provides pattern validation. Annualizing to ~130% (four quarterly cycles) places this strategy among the highest Sharpe ratio opportunities in quantitative finance, albeit with scaling constraints due to Polymarket's liquidity.

## 13.2 Optimal Trading Protocol

For institutional implementation, we recommend the following protocol:

**Entry Rule:** Trade when (a) edge (p_model - p_market) is between 0.10 and 0.15, (b) model probability is between 0.25 and 0.85, and (c) Polymarket price is between 0.40 and 0.70. These filters select high-quality signals while avoiding overconfidence traps (>20% edge) and extremely priced markets (where liquidity or positioning may distort).

**Position Sizing:** Calculate Kelly optimal as Edge/(1-Price), apply 25% fraction, and cap at lesser of (Kelly bet, $5,000 per event, 10% of total allocated capital). For example, with edge 0.12, price 0.60, and $10,000 seasonal allocation: Kelly_optimal = 0.12/0.40 = 0.30, Quarter_Kelly = 0.075, Bet = 0.075 × $10,000 = $750, capped at $750 (within $5,000 limit).

**Execution:** Place limit orders at current midpoint or better, accepting up to 1% worse fill if immediate execution desired. Hold positions until market resolution (typically 1-7 days post-announcement). Record outcomes, update Elo ratings, and track realized vs expected P&L for calibration drift detection.

**Monitoring:** Daily review of open positions and upcoming earnings. Weekly P&L reconciliation and edge realization analysis. Quarterly model retraining with latest three months of data, backtesting new model on most recent quarter before deployment.

## 13.3 Capital Allocation and Scalability

For a $100,000 institutional allocation, we propose $10,000-$20,000 per earnings season (quarterly deployment). Expecting 15-20 Sweet Spot or high-quality All Events trades per quarter with average position size $500-1,500, this allocation supports the strategy comfortably without hitting position caps. At 30% quarterly ROI, returns approximate $3,000-$6,000 per quarter or $12,000-$24,000 annually—a meaningful absolute contribution for a sub-strategy within a broader quantitative book.

Scaling beyond $500,000 total allocation becomes challenging due to Polymarket's market size limitations. Individual earnings markets rarely exceed $100,000-200,000 total volume, constraining single-position sizes to $5,000-$10,000 without materially impacting prices. A $1 million strategy might deploy $100,000 per quarter across 30-40 trades, requiring relaxation of Sweet Spot filters to include broader 5-20% edges—which our analysis shows reduces ROI from 32% to 13% but maintains positive expectation.

Ultimately, this strategy is best suited for $50,000-$500,000 in capital seeking asymmetric, event-driven returns with low correlation to traditional beta. Larger allocations require either venue diversification (when other prediction markets mature) or acceptance of lower per-dollar returns as position sizes strain market capacity.

## 13.4 Future Research Directions

Several extensions could enhance strategy performance or scientific understanding. First, incorporating earnings call transcript sentiment via NLP could add predictive signal: management tone, word choice, and Q&A dynamics often contain forward guidance not captured in numerical estimates. Second, sector-specific models might improve on our global approach for industries with unique dynamics (e.g., commodity-linked earnings, seasonal retail, regulated utilities). Third, predicting beat magnitude rather than binary outcomes could enable graduated position sizing: larger bets on expected large beats, smaller on marginal beats.

From a platform perspective, expanding to other prediction markets (Kalshi for regulated U.S. events, Polymarket competitors in DeFi) would validate whether our edge generalizes or is Polymarket-specific. Exploring international markets (FTSE 100, DAX, Nikkei) would test geographic universality of Elo and analyst consensus effects. Finally, real-time estimate tracking—monitoring intraday analyst updates rather than end-of-day snapshots—could provide earlier signals for position entry or exit.

Methodologically, investigating model ensembles via stacking (training a meta-model on RF, XGB, LR probabilities) might improve calibration. Bayesian approaches could quantify uncertainty around our probability estimates, enabling confidence-weighted Kelly sizing. And causal inference techniques (difference-in-differences, regression discontinuity around estimate revision events) could sharpen our understanding of which features represent true causal drivers versus mere correlations, improving model interpretability and stability.

---

**Pages:** 76-80  
**Section:** Conclusions & Recommendations  
**Classification:** Confidential

