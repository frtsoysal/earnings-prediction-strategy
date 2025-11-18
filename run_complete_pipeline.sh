#!/bin/bash

# Complete ML Pipeline Runner
# Runs all three steps of the EPS beat prediction pipeline

echo "=========================================="
echo "EPS BEAT PREDICTION - COMPLETE PIPELINE"
echo "=========================================="
echo ""

# Step 1: Data Preparation
echo "STEP 1/3: Preparing data..."
echo "----------------------------------------"
python3 prepare_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi
echo ""

# Step 2: Model Training
echo "STEP 2/3: Training models..."
echo "----------------------------------------"
python3 train_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Model training failed!"
    exit 1
fi
echo ""

# Step 3: Model Evaluation
echo "STEP 3/3: Evaluating models..."
echo "----------------------------------------"
python3 evaluate.py
if [ $? -ne 0 ]; then
    echo "ERROR: Model evaluation failed!"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "  - Training data: X_train.csv, y_train.csv"
echo "  - Test data: X_test.csv, y_test.csv"
echo "  - Trained models: models/*.pkl (4 files)"
echo "  - Visualizations: results/*.png (14 files)"
echo "  - Reports: results/*.csv, results/*.txt"
echo ""
echo "View results:"
echo "  - cat results/evaluation_report.txt"
echo "  - cat results/model_comparison.csv"
echo "  - open results/  # (to view visualizations)"
echo ""

