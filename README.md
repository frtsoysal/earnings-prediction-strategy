# S&P 500 Earnings Prediction Strategy
## Machine Learning-Powered Alpha Generation Through Prediction Market Arbitrage

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

## 📊 Overview

A quantitative investment strategy that combines machine learning with prediction market arbitrage to generate alpha through S&P 500 earnings announcements. The system achieves **32.73% quarterly ROI** with a **93.75% win rate** by identifying mispricings between ML model predictions and Polymarket prices.

### Key Results
- **82.65%** out-of-sample prediction accuracy
- **54%** of predictive power from novel Elo rating system
- **262** real Polymarket markets backtested
- **14,239** historical earnings observations analyzed
- **$192.49** capital requirement for optimal "Sweet Spot" strategy

## 🎯 Core Innovation

### Elo Rating System for Corporate Earnings
We adapt chess Elo ratings to quantify corporate earnings performance, creating four powerful metrics:
- `elo_before`: Current performance rating
- `elo_momentum`: Recent trend (4-quarter weighted)
- `elo_decay`: Time-weighted historical average
- `elo_vol_4q`: Consistency measure

This Elo-based approach captures patterns that traditional financial metrics miss, contributing 54% of model predictive power.

## 🏗️ Project Structure

```
ML/
├── graphtest/
│   ├── global_model/           # Cross-company ML models
│   │   ├── train_global_model.py
│   │   ├── models/            # Trained models (.pkl files)
│   │   └── evaluation_report.txt
│   ├── polymarket_backtest/    # Historical backtest
│   │   ├── polymarket_vs_ml.py
│   │   ├── edge_analysis.py
│   │   ├── strategy_optimization.py
│   │   └── tables/            # Results tables
│   ├── polymarket_live/        # Live trading system
│   │   ├── live_trading_recommendations.py
│   │   ├── README.md
│   │   └── IMPLEMENTATION_SUMMARY.md
│   └── final_report/           # Investment strategy report
│       ├── COMPLETE_INVESTMENT_REPORT.md
│       ├── REFERENCES.md
│       ├── figures/           # 21 professional visualizations
│       └── generate_*.py      # Figure generation scripts
├── data/
│   └── raw/                   # 498 company CSV files (not included)
└── scripts/                   # Data fetching utilities
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib, requests
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/earnings-prediction-strategy.git
cd earnings-prediction-strategy

pip install -r requirements.txt
```

### API Keys Setup
1. Get Alpha Vantage API key: https://www.alphavantage.co/
2. Replace `YOUR_ALPHA_VANTAGE_API_KEY` in:
   - `graphtest/polymarket_live/live_trading_recommendations.py`
   - `graphtest/polymarket_backtest/fetch_missing_tickers.py`

### Run Live Trading Recommendations
```bash
cd graphtest/polymarket_live
python3 live_trading_recommendations.py
```

Output includes:
- Strategy breakdown (Sweet Spot, Positive Edge, etc.)
- Top 10 recommendations by edge
- Kelly-weighted position sizes
- Expected value calculations
- CSV file with all opportunities

## 📈 Trading Strategies

### 1. Sweet Spot Strategy (Recommended) ⭐
- **Edge Range:** 10-15%
- **Historical ROI:** +32.73%
- **Win Rate:** 93.75%
- **Capital:** $192.49 per cycle

### 2. All Positive Edge
- **Edge Range:** >0%
- **Historical ROI:** +13.07%
- **Win Rate:** 72.22%
- **Capital:** $976.72 per cycle

### 3. High Edge
- **Edge Range:** ≥8%
- **Historical ROI:** +12.23%
- **Win Rate:** 64.91%
- **Capital:** $849.12 per cycle

## 🧪 Methodology

### Data Collection
- **Source:** Alpha Vantage API (Premium)
- **Coverage:** 498 S&P 500 companies
- **Period:** 2011-2025 (14+ years)
- **Features:** 29 leak-safe predictors

### Machine Learning Models
- **Random Forest:** Robust ensemble method
- **XGBoost:** Gradient boosting for complex patterns
- **Logistic Regression:** Well-calibrated probabilities

### Temporal Validation
- **Train:** 35% (2011-2020)
- **Test:** 65% (2020-2025)
- **Leak Prevention:** Strict temporal ordering

### Position Sizing
- **Kelly Criterion:** Optimal bet sizing
- **Quarter-Kelly:** 25% of optimal (risk management)
- **Fee Adjustment:** 2% on winning trades

## 📊 Key Findings

### Feature Importance (Top 10)
1. `eps_estimate_average_90_days_ago` (9.5%)
2. `eps_estimate_analyst_count` (8.5%)
3. `eps_estimate_revision_down_trailing_30_days` (7.9%)
4. `elo_momentum` (7.5%)
5. `eps_estimate_average` (7.4%)
6. `eps_estimate_high` (6.9%)
7. `eps_estimate_low` (6.8%)
8. `elo_before` (6.6%)
9. `eps_estimate_average_30_days_ago` (6.1%)
10. `eps_estimate_average_7_days_ago` (5.9%)

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | Brier Score |
|-------|----------|-----------|--------|----------|-------------|
| Random Forest | 82.29% | 87.75% | 91.04% | 89.37% | 0.1449 |
| XGBoost | 81.48% | 89.94% | 87.68% | 88.79% | 0.1533 |
| Logistic Reg | 82.65% | 89.59% | 89.81% | 89.70% | 0.1402 |

## 📖 Documentation

### Reports
- **[Complete Investment Strategy Report](graphtest/final_report/COMPLETE_INVESTMENT_REPORT.md)** - 96-page comprehensive analysis
- **[References & Citations](graphtest/final_report/REFERENCES.md)** - Academic sources and data attributions
- **[Live Trading System](graphtest/polymarket_live/README.md)** - Implementation guide

### Key Scripts
- `train_global_model.py` - Train ML models (589 lines)
- `polymarket_vs_ml.py` - Backtest execution (456 lines)
- `edge_analysis.py` - ROI calculation (589 lines)
- `live_trading_recommendations.py` - Production system (480 lines)

## ⚠️ Risk Warnings

1. **Sample Size:** Backtest covers 1 quarter (262 markets). Multi-quarter validation recommended.
2. **Market Efficiency:** Edge may decay as markets become more efficient.
3. **Concentration Risk:** Limited to earnings events (quarterly).
4. **Liquidity:** Polymarket liquidity varies by market.
5. **Regulatory:** Prediction markets may face regulatory changes.

**Past performance does not guarantee future results.**

## 🔬 Research Contributions

### Novel Methodologies
1. **Elo for Earnings:** First application of chess ratings to corporate earnings
2. **Global Model:** 82.65% accuracy vs. 66.43% for ticker-specific models
3. **Edge Optimization:** Sweet Spot strategy identification
4. **Leak Prevention:** Comprehensive 29-feature audit protocol

### Academic Foundations
- **Elo (1978):** Rating system methodology
- **Kelly (1956):** Optimal bet sizing
- **Breiman (2001):** Random Forest algorithm
- **Chen & Guestrin (2016):** XGBoost framework

## 📊 Visualizations

The project includes 21 professional figures:
- Yearly beat rate trends
- Global model performance metrics
- Feature importance rankings
- Elo distribution and trajectories
- Edge bucket analysis
- Calibration curves
- P&L simulation results
- Strategy comparison charts
- Architecture diagrams

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-asset expansion (international markets)
- Real-time data streaming
- Additional ML models (neural networks, etc.)
- Enhanced feature engineering
- Risk management improvements

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions, collaborations, or institutional inquiries:
- Open an issue on GitHub
- See COMPLETE_INVESTMENT_REPORT.md for detailed methodology

## 🙏 Acknowledgments

### Data Sources
- **Alpha Vantage** - Financial data API
- **Polymarket** - Prediction market platform
- **Wikipedia** - S&P 500 constituent list

### Open Source Libraries
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Theoretical Foundations
- Arpad Elo - Chess rating system
- John Kelly - Kelly criterion
- Leo Breiman - Random Forest
- Tianqi Chen - XGBoost

---

**Disclaimer:** This project is for educational and research purposes. Trading involves substantial risk. The authors assume no liability for losses. Consult qualified financial advisors before implementing any trading strategy.

**© 2025 Quantitative Research Division. Released under MIT License.**

---

⭐ Star this repository if you find it useful!

