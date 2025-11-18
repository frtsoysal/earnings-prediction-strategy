# EPS Beat Prediction - Machine Learning Pipeline

Complete ML pipeline for predicting whether IBM will beat earnings per share (EPS) estimates.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# For macOS with XGBoost:
brew install libomp
```

### Run Pipeline

```bash
# Step 1: Prepare data (creates train/test split)
python3 prepare_data.py

# Step 2: Train models (Random Forest, XGBoost, Logistic Regression)
python3 train_model.py

# Step 3: Evaluate and visualize results
python3 evaluate.py
```

## 📊 Dataset

**Source:** IBM earnings estimates from Alpha Vantage API  
**File:** `ibm_earnings_estimates_with_q4.csv`  
**Samples:** 33 quarterly records (2017-2025)  
**Target:** Binary classification (Beat=1, Miss=0)

### Features (24 total)

- **Price Momentum (4):** 1-month and 3-month price changes before earnings
- **EPS Estimates (8):** Consensus estimates and historical averages
- **Analyst Revisions (4):** Up/down revisions in trailing 7 and 30 days
- **Revenue Estimates (4):** Consensus revenue expectations
- **Elo Ratings (4):** Custom performance metrics tracking beat consistency

### Data Split

- **Training:** 2017-2022 (22 samples, 90.91% beat rate)
- **Testing:** 2023-2025 (11 samples, 100% beat rate)

## 🤖 Models

### 1. Random Forest
- **Hyperparameters:** n_estimators=300, max_depth=30
- **Test Accuracy:** 100% (predicted all beats correctly)
- **Top Features:** price_3m_before, elo_decay, revenue_estimate_average

### 2. XGBoost
- **Hyperparameters:** n_estimators=200, max_depth=3, learning_rate=0.05
- **Test Accuracy:** 0% (predicted all as misses)
- **Note:** Conservative predictions due to class imbalance

### 3. Logistic Regression
- **Hyperparameters:** C=0.01, penalty='l1'
- **Test Accuracy:** 0% (predicted all as misses)
- **Note:** Linear model struggled with imbalanced data

## 📁 Project Structure

```
ML/
├── README.md                    # This file
├── ML_PIPELINE_SUMMARY.md       # Detailed analysis
├── requirements.txt             # Python dependencies
│
├── prepare_data.py              # Data preparation script
├── train_model.py               # Model training script
├── evaluate.py                  # Evaluation script
│
├── fetch_alpha_vantage.py       # Data collection script
├── ibm_earnings_estimates_with_q4.csv  # Source data
│
├── X_train.csv, y_train.csv     # Training data
├── X_test.csv, y_test.csv       # Test data
├── feature_names.txt            # Feature list
├── data_info.json               # Split metadata
├── training_results.json        # Hyperparameter search results
│
├── models/                      # Trained models
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lr_model.pkl
│   └── preprocessor.pkl
│
└── results/                     # Evaluation outputs
    ├── feature_importance_*.png
    ├── confusion_matrix_*.png
    ├── roc_curves_comparison.png
    ├── pr_curves_comparison.png
    ├── calibration_curves.png
    ├── prediction_analysis_*.png
    ├── model_comparison.csv
    ├── misclassified_samples.csv
    └── evaluation_report.txt
```

## 📈 Results

### Model Performance (Test Set)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 100%     | 100%      | 100%   | 100%     |
| XGBoost             | 0%       | 0%        | 0%     | 0%       |
| Logistic Regression | 0%       | 0%        | 0%     | 0%       |

### Top 5 Features (Random Forest)

1. **price_3m_before** (9.6%) - Stock price 3 months before earnings
2. **elo_decay** (9.1%) - Elo rating decay factor
3. **revenue_estimate_average** (8.8%) - Consensus revenue estimate
4. **elo_before** (5.8%) - Historical Elo rating
5. **revenue_estimate_low** (5.7%) - Conservative revenue estimate

## ⚠️ Important Notes

### Class Imbalance

**The Challenge:** IBM has beaten EPS estimates in 31 of 33 quarters (94% beat rate), with the test set showing 100% beats.

**Implications:**
- ROC-AUC is undefined (requires both classes)
- Random Forest "wins" by always predicting "beat"
- Difficult to evaluate false positive rate
- Model calibration uncertain

### Recommendations

1. **Expand Dataset:** Include more companies with varied performance
2. **Reframe Problem:** Predict beat magnitude (regression) instead of binary classification
3. **Alternative Metrics:** Use probability calibration and confidence intervals
4. **More Features:** Add macro indicators, sentiment analysis, options market data

## 💻 Usage Examples

### Make Predictions

```python
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('models/rf_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Prepare features (24 features required)
features = {
    'price_1m_before': 250.0,
    'price_3m_before': 245.0,
    'price_change_1m_pct': 2.0,
    'price_change_3m_pct': 2.04,
    'eps_estimate_average': 1.50,
    # ... (add all 24 features)
}

X = pd.DataFrame([features])
X_processed = preprocessor.transform(X)

# Predict
prediction = model.predict(X_processed)[0]
probability = model.predict_proba(X_processed)[0, 1]

print(f"Prediction: {'Beat' if prediction else 'Miss'}")
print(f"Confidence: {probability:.2%}")
```

### Analyze Feature Importance

```python
import joblib
import pandas as pd

model = joblib.load('models/rf_model.pkl')
features = pd.read_csv('feature_names.txt', header=None)[0].tolist()

importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Imputation:** Median strategy for missing values
2. **Scaling:** StandardScaler for feature normalization
3. **Leakage Prevention:** Excluded post-event features

### Hyperparameter Tuning
- **Method:** RandomizedSearchCV
- **CV Folds:** 5
- **Scoring:** ROC-AUC
- **Iterations:** 10-20 per model

### Class Imbalance Handling
- Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` based on class ratio
- Logistic Regression: `class_weight='balanced'`

## 📚 Dependencies

- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

## 📊 Visualizations

The pipeline generates 14 publication-quality visualizations:

1. **Feature Importance** - Random Forest and XGBoost
2. **Confusion Matrices** - All 3 models
3. **ROC Curves** - Model comparison
4. **Precision-Recall Curves** - Performance across thresholds
5. **Calibration Curves** - Probability calibration
6. **Prediction Analysis** - Per-sample prediction confidence

All saved in `results/` directory as high-resolution PNG files (300 DPI).

## 🎯 Next Steps

### Short Term
- [ ] Add cross-validation within training set
- [ ] Implement SMOTE for class balancing
- [ ] Add confidence intervals to predictions

### Medium Term
- [ ] Expand to multi-company dataset
- [ ] Add sentiment analysis features
- [ ] Implement ensemble stacking

### Long Term
- [ ] Real-time data pipeline integration
- [ ] Deploy as REST API
- [ ] Build interactive dashboard

## 📖 Documentation

- **Detailed Analysis:** See `ML_PIPELINE_SUMMARY.md`
- **Evaluation Report:** See `results/evaluation_report.txt`
- **Model Comparison:** See `results/model_comparison.csv`

## 🤝 Contributing

This is a learning/demonstration project. Key areas for improvement:

1. Data collection from multiple sources
2. Advanced feature engineering
3. Alternative problem formulations
4. Production deployment infrastructure

## 📄 License

Educational/demonstration project.

## 📧 Contact

For questions about the implementation, see the code comments or documentation files.

---

**Last Updated:** November 11, 2025  
**Data Source:** Alpha Vantage API  
**Stock:** IBM (International Business Machines Corporation)

