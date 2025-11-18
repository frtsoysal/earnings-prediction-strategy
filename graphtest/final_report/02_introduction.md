# 1. INTRODUCTION

## 1.1 Background & Market Opportunity

Corporate earnings announcements represent recurring, high-information events that have long attracted the attention of quantitative traders and fundamental analysts alike. Every quarter, thousands of publicly traded companies report financial results and provide guidance, creating discrete moments when new information enters the market and prices adjust accordingly. For S&P 500 constituents—the large-cap core of the U.S. equity market—these announcements are particularly significant, often moving individual stock prices by 4-6% in a single trading session and occasionally triggering much larger moves when results deviate substantially from expectations.

The central question in earnings prediction has traditionally been framed as: Will the company beat, meet, or miss analyst consensus estimates? This binary or trinary classification problem has proven remarkably persistent and profitable. Companies that consistently beat estimates tend to enjoy positive momentum, higher valuations, and favorable analyst coverage, while chronic missers face the opposite dynamic. The ability to predict these outcomes even marginally better than market consensus creates opportunities for alpha generation across multiple strategies—from directional equity trades to options strategies to, more recently, prediction market contracts.

## 1.2 Earnings Announcements as Information Events

To understand why earnings predictions remain profitable despite decades of academic and practitioner attention, we must recognize the multi-layered information structure embedded in these announcements. First, there is the actual reported number—earnings per share, revenue, margins, and guidance. Second, there is the analyst consensus—a forward-looking estimate aggregated from multiple sell-side research teams. Third, there is the market's expectation, imperfectly reflected in option implied volatility, stock price momentum leading into the announcement, and prediction market prices where available.

Importantly, these three layers often diverge. Analysts may be systematically biased (research shows a historical tendency toward conservatism, with average beat rates exceeding 70% over the past decade, as documented in Figure 1 of this report). Markets may over- or under-react to recent price trends, creating technical patterns uncorrelated with fundamental reality. And prediction markets, while theoretically efficient aggregators of distributed information, may suffer from limited liquidity, participant biases, or slow incorporation of new data like estimate revisions.

Our strategy exploits these divergences by constructing a probabilistic forecast superior to both analyst consensus (which is deterministic) and market pricing (which may be informationally incomplete). The machine learning models we employ can synthesize patterns across companies, time periods, and feature combinations that human analysts and market participants may overlook or underweight. In particular, we find that historical performance trajectories—captured via our Elo rating system—contain far more predictive signal than traditional financial metrics like revenue growth or margin trends, yet appear underutilized by market participants.

## 1.3 Polymarket: A New Trading Venue

Polymarket is a decentralized prediction market platform where users trade binary outcome contracts on real-world events. For earnings predictions, Polymarket offers contracts phrased as "Will Company X beat earnings estimates this quarter?" with Yes and No shares trading between $0.00 and $1.00 per share. Upon resolution, winning shares pay $1.00, creating a clear payoff structure.

Several features make Polymarket attractive as a testing ground for earnings strategies. First, outcomes are objectively resolvable—earnings either beat consensus or they don't, determined by comparing the reported EPS to the pre-announcement analyst average. Second, markets are liquid enough for meaningful position sizes (typically $10,000-$100,000+ in total trading volume per event), while not so liquid that institutional capital would significantly move prices. Third, the binary structure simplifies modeling compared to magnitude-based predictions (how much will they beat by?). Fourth, settlement is immediate and transparent—no counterparty risk or ambiguity about payoffs.

The platform charges a 2% fee on gross winnings, which we incorporate into all return calculations. Markets typically open 1-14 days before the scheduled earnings announcement and close shortly after the actual report. During this window, prices fluctuate based on new information, participant flows, and occasionally late estimate revisions. For our backtest analysis, we use the average price during each market's lifetime as a proxy for the executable price—a conservative assumption that likely understates actual profitability since skilled traders can often achieve better fills by timing entries.

## 1.4 Report Objectives and Scope

This report serves multiple audiences and purposes. For quantitative portfolio managers and trading desk heads, it provides a turnkey investment strategy with empirically validated returns, complete implementation specifications, and risk assessment. For data scientists and machine learning practitioners, it demonstrates the application of supervised learning to financial prediction with careful attention to temporal validity and feature engineering. For academic researchers interested in market microstructure or behavioral finance, it offers empirical evidence on analyst dynamics, consensus formation, and the efficiency (or lack thereof) of nascent prediction markets.

Our scope encompasses three primary components. First, we document the complete data pipeline and feature engineering process, including our novel application of Elo ratings to corporate earnings—to our knowledge, the first such adaptation in published literature. Second, we present comprehensive empirical analysis of 14,239 historical earnings events, testing hypotheses about analyst consensus dynamics, revision momentum, and coverage effects. Third, we backtest our complete strategy on 262 real Polymarket outcomes from Q3 2025, calculating edge, expected value, and realized P&L under realistic assumptions including fees and Kelly-optimal position sizing.

What this report does not cover: We do not address market microstructure details of Polymarket order execution, we do not explore event-driven equity strategies based on our predictions (though such extensions are natural), and we do not provide legal or compliance guidance on prediction market participation (readers should consult appropriate advisors). Our focus remains squarely on the quantitative strategy: data, models, backtest, and results.

The remainder of this report is structured as follows. Section 2 outlines our investment thesis and strategic framework. Section 3 details data collection across 498 S&P 500 companies. Section 4 explains our feature engineering, particularly the Elo system. Section 5 describes our machine learning methodology and model performance. Section 6 provides deep exposition on Elo ratings. Section 7 presents research findings on analyst consensus. Section 8 covers our Polymarket backtest. Section 9 analyzes edge and calibration. Section 10 optimizes strategy parameters. Section 11 assesses risks. Section 12 offers implementation guidance. Section 13 concludes with recommendations and future directions.

---

**Figure 1.1: S&P 500 Historical Beat Rates (2017-2025)**  
*See: final_report/figures/fig1_yearly_beat_rate_trend.png*

The trend toward higher beat rates in recent years (77.0% in 2025 versus 67.2% in 2019) likely reflects increasing analyst conservatism, improved corporate guidance quality, or both. This secular shift has implications for baseline comparisons and suggests our models must adapt to evolving base rates—addressed through quarterly recalibration protocols described in Section 12.

---

**Pages:** 5-8  
**Section:** Introduction  
**Classification:** Confidential

