# 6. ELO RANKING SYSTEM - DETAILED EXPOSITION

The Elo rating system's journey from competitive chess to corporate finance represents an elegant example of cross-domain mathematical adaptation. When Arpad Elo introduced his rating formula in the 1960s to replace earlier chess ranking systems, he revolutionized competitive gaming by creating a probabilistic, self-correcting mechanism that converged to true skill levels over repeated contests. We recognized that quarterly earnings reports share key structural similarities with chess matches: binary outcomes (beat/miss versus win/loss), repeated events (companies report every quarter), and performance relative to expectations (analyst consensus versus opponent rating). This section explicates our adaptation in detail, both for transparency and to enable replication.

## 6.1 Conceptual Foundation

In chess, each player carries a rating (typically 800-2800 for human players, with 1500 as average). Before a match, Elo predicts the outcome probability: a 1700-rated player facing a 1500-rated opponent has approximately 75% win probability. After the match, both ratings update based on the actual outcome. If the 1700 player wins as expected, their rating rises modestly (perhaps +8). If they lose unexpectedly, it falls substantially (perhaps -32). The opponent's rating moves inversely. Over time, ratings converge such that a player's rating accurately predicts their expected performance against opponents of various strengths.

For earnings, we construct the analogy as follows. Each company starts at Elo 1500 (neutral). Each quarter, they "face" analyst consensus—the estimate represents the "opponent strength." Beating estimates is a "win"; missing is a "loss." The magnitude of surprise—how much they beat or missed by—determines the rating change intensity, analogous to beating a much stronger or weaker opponent. After many quarters, companies accumulate Elo ratings that reflect their tendency to beat (high Elo) or miss (low Elo) estimates.

The key insight is that Elo captures not just a binary track record (X% historical beat rate) but a weighted, recency-biased trajectory. A company that beat estimates three years ago but has missed the last four quarters will have declining Elo despite a >50% cumulative beat rate. This dynamic weighting makes Elo more predictive than naive historical averages.

## 6.2 Mathematical Formulation and Adaptive Mechanisms

The core update equation is Elo_new = Elo_old + K × Surprise × 100, where Surprise = (Actual_EPS - Estimate_EPS). The K parameter controls update sensitivity—higher K means ratings change more dramatically per outcome. In chess, K is fixed (often 32 for adults, 40 for juniors). For earnings, we innovate by making K adaptive to analyst coverage.

The rationale is information quality. A company with 25 analysts covering it has extensive research, frequent investor calls, and transparent financials—making earnings relatively predictable. When such a company beats or misses, it's a smaller "surprise" because the fundamentals were well-known. Conversely, a company with only 5 analysts is informationally opaque; beats and misses are noisier signals, potentially driven by one-time items or reporting timing rather than sustainable performance changes. We therefore scale K downward for high-coverage companies (K×0.7 for 20+ analysts) and upward for low-coverage (K×1.5 for <5 analysts), with intermediate scalings for 5-19 analysts.

This adaptive K dramatically improves Elo's predictive power in our cross-sectional setting. Without it, large-cap tech companies with 40 analysts and small-cap industrials with 6 analysts would have equally volatile Elo trajectories despite vastly different information environments. With adaptive K, Elo volatility correctly reflects fundamental uncertainty, and Elo momentum signals become more reliable.

From base Elo, we construct elo_momentum as 0.4×Δt + 0.3×Δt-1 + 0.2×Δt-2 + 0.1×Δt-3, where Δt is the most recent quarter's Elo change. This exponential weighting emphasizes recent performance while still incorporating medium-term trends. A company with momentum +200 (recent large positive changes) is on a hot streak; momentum -150 signals deterioration. Empirically, elo_momentum emerges as our single strongest feature (36% importance), validating this weighted approach over simpler alternatives like raw Elo level or binary "beat last quarter" indicators.

## 6.3 Empirical Validation

Testing Elo's predictive power, we compute Spearman rank correlation between elo_before and subsequent beat/miss outcomes across all 14,239 observations. The correlation is ρ=0.161 (p<0.001), the highest of any single feature and remarkable for a univariate metric. Companies in the top Elo decile beat 84% of the time; those in the bottom decile beat only 56% of the time—a 28 percentage point spread from a single numerical score.

Examining elo_momentum specifically, companies with top-quartile momentum (rising Elo trajectories) beat 81% of the time versus 62% for bottom-quartile momentum (declining trajectories). This 19 percentage point gap, combined with momentum's 36% model importance, positions it as the most powerful signal in our arsenal. Intuitively, momentum captures a company's current trajectory in a way that static features cannot: whether the business is improving, stable, or deteriorating.

The success of Elo in our context offers broader lessons for financial machine learning. Often, the most powerful features are not raw accounting ratios or price levels but derived metrics that encode dynamic processes—in our case, the historical trajectory of earnings surprises weighted by recency and information quality. This principle likely generalizes beyond earnings to other corporate events (M&A success rates, product launches, regulatory approvals) where historical track records predict future outcomes.

---

**Pages:** 32-37  
**Section:** Elo Rating System  
**Classification:** Confidential

