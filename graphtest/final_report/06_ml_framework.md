# 5. MACHINE LEARNING METHODOLOGY

## 5.1 Model Architecture and Design

We employ an ensemble of three supervised learning algorithms, each chosen for complementary strengths. Random Forest, our first model, excels at capturing non-linear interactions between features without overfitting, thanks to its bagging procedure and tree-based splits. With 300 trees of maximum depth 20, balanced class weights, and minimum samples per leaf of 2, our Random Forest configuration achieves the best probability calibration (Brier score 0.193) among our models, making it particularly suitable for Kelly-based position sizing where accurate probability estimates are paramount.

XGBoost, our second model, brings gradient boosting's sequential error correction to bear on the problem. With 200 trees of depth 5, learning rate 0.05, and scale_pos_weight calibrated to the training set's class imbalance, XGBoost achieves the highest ROC-AUC (0.848) despite slightly lower raw accuracy than Logistic Regression. This superior discriminative ability makes XGBoost valuable for ranking opportunities—identifying which earnings events have the highest beat probability relative to others—even if its absolute probability estimates require calibration adjustments.

Logistic Regression, our third model, serves dual purposes. First, it provides a transparent baseline: the learned coefficients directly indicate each feature's marginal effect on log-odds of beating, facilitating interpretation and hypothesis testing. Second, despite its linear simplicity, Logistic Regression achieves our highest test set accuracy at 82.65%, suggesting the problem has strong linear components that more complex models don't necessarily improve upon. We employ L2 regularization (C=1.0) and balanced class weights, with convergence typically achieved in under 100 iterations despite the 1,000-iteration limit.

Each model processes identically prepared input: 31 features standardized to zero mean and unit variance after median imputation of missing values. We train on 4,983 observations spanning 2011-2020 and test on 9,256 observations spanning 2020-2025, maintaining strict temporal ordering to prevent any information leakage from future to past.

## 5.2 Training Methodology and Validation

Our training protocol prioritizes temporal validity over cross-sectional validation. Standard k-fold cross-validation, while appropriate for many machine learning tasks, is unsuitable for time series financial data because it mixes past and future observations. For instance, 5-fold CV might train on 2015-2019 data and test on 2017 data—effectively predicting the past using the future, which inflates accuracy metrics but fails catastrophically in production.

Instead, we implement a temporal train/test split. After sorting all 14,239 observations by fiscal quarter end date, we allocate the first 35% (4,983 observations, covering 2011 to mid-2020) to training and the remaining 65% (9,256 observations, mid-2020 to 2025) to testing. This 35/65 split balances two competing needs: training requires sufficient data to learn patterns (hence not too small), while testing should reflect the deployment environment as closely as possible (hence not trained on overly historical data).

Within the training set, we employ TimeSeriesSplit cross-validation for hyperparameter tuning. This sklearn utility maintains temporal ordering even within CV folds, ensuring each validation fold predicts only chronologically subsequent data. We run randomized grid search over key hyperparameters—number of trees, depths, learning rates—selecting configurations that maximize 5-fold CV ROC-AUC. This two-stage process (CV for tuning, holdout test for final evaluation) guards against both overfitting to training data and overfitting to validation folds.

The resulting test set performance—82.65% accuracy, 0.846 ROC-AUC, 87.56% precision, 89.53% recall for Logistic Regression—represents genuine out-of-sample forecasting ability on data the models have never encountered. We further validate this by comparing to company-specific models (66.43% average accuracy), confirming that our global approach's superior data aggregation drives meaningful performance gains.

## 5.3 Feature Importance and Interpretation

Feature importance analysis via Random Forest's Gini impurity decomposition reveals a striking pattern: Elo metrics dominate. The top four features—elo_momentum (36.17%), elo_before (6.38%), elo_decay (6.16%), and elo_vol_4q (5.27%)—collectively account for 53.98% of predictive power. No other feature exceeds 3% individual importance. This concentration indicates that historical earnings performance, properly quantified, is the single strongest predictor of future performance, surpassing even analyst estimates themselves.

The dominance of elo_momentum in particular (36% importance, six times larger than any non-Elo feature) suggests that the trajectory of earnings performance—whether a company is on an improving or deteriorating path—matters more than its current absolute performance level. A company with mediocre historical Elo but strong recent momentum (several consecutive beats) may be entering a new performance regime, making another beat likely. Conversely, a company with high absolute Elo but declining momentum may be exhausting its ability to exceed estimates, perhaps because analysts have adjusted upward or business conditions are normalizing.

Beyond Elo, the next tier of features includes lagged EPS growth (2.66%), price momentum before announcement (2.43% and 2.38% for 1-month and 3-month lookbacks), and revenue estimates (2.18%). Analyst revision features, despite their theoretical appeal and our empirical findings in Section 7, contribute individually modest importance (0.5-1.5% each). This apparent contradiction—revision momentum matters in univariate analysis but shows low importance in the ensemble—likely reflects correlation with Elo: companies with positive revisions already have rising Elo momentum, so the revision signal is partially redundant once Elo is included.

## 5.4 Performance Metrics and Benchmarking

Our global Logistic Regression model's 82.65% test set accuracy deserves contextualization against multiple benchmarks. First, the naive baseline of always predicting "beat" (since 74.82% of test set observations are beats) yields 74.82% accuracy. Our model's 7.83 percentage point improvement represents a 32% reduction in error rate (from 25.18% baseline error to 17.35% model error)—a meaningful but not extraordinary gain that aligns with realistic expectations for earnings prediction given inherent randomness.

Second, company-specific models trained on individual ticker histories average 66.43% accuracy despite having access to company-specific patterns. Our global model's 16 percentage point superior performance (82.65% versus 66.43%) validates the cross-company learning approach: pooling data from 468 companies allows the model to learn generalizable patterns that individual-company models, starved for data with only 30 observations each, cannot reliably detect.

Third, comparing to published academic benchmarks is challenging due to differing datasets and methodologies, but studies in the earnings prediction literature typically report 60-70% accuracy for traditional regression or classification approaches. Our 82.65% thus represents state-of-the-art performance, likely attributable to our comprehensive feature set (particularly Elo) and modern ensemble methods.

From a precision-recall perspective, our 87.56% precision means that when the model predicts a beat, it's correct 87.56% of the time—reliability sufficient for trading strategies. Our 89.53% recall means we identify 89.53% of actual beats, leaving only 10.47% as false negatives. The tradeoff is lower specificity (50-60% for identifying misses), but since our strategy focuses on betting on beats rather than shorting misses, this asymmetry is acceptable and indeed expected given the class imbalance in training data.

---

**Figure 5.1: Global Model Performance Metrics**  
*See: final_report/figures/fig2_global_model_performance.png*

**Table 5.1: Model Comparison**  
*See: final_report/tables/table1_global_model_performance.txt*

The consistent performance across Random Forest (79.86%), XGBoost (79.81%), and Logistic Regression (82.65%) suggests our features genuinely predict outcomes rather than one model overfitting. Ensemble agreement bolsters confidence in predictions.

---

**Pages:** 25-31  
**Section:** Machine Learning Methodology  
**Classification:** Confidential

