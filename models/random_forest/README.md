# Random Forest Models - Customer Intelligence Platform

## Overview
This folder contains all Random Forest model experiments for the Customer Intelligence Platform project.

## Models Implemented
1. **Customer Lifetime Value (CLV) Prediction** - Random Forest Regression
2. **Churn Risk Classification** - Random Forest Classification  
3. **Customer Segmentation** - K-Means Clustering (with Random Forest feature importance)

## Folder Structure
```
random_forest/
├── random_forest_mlflow_models.ipynb    # Main notebook with all RF models
├── models/                              # Saved model artifacts (.pkl files)
│   ├── clv_random_forest_model.pkl
│   ├── churn_random_forest_model.pkl
│   ├── segmentation_kmeans_model.pkl
│   └── segmentation_scaler.pkl
├── mlruns/                              # MLflow experiment tracking
├── results/                             # Output files and reports
└── README.md                            # This file
```

## Key Results
- **CLV Model**: R² = 0.9995 (Excellent performance)
- **Churn Model**: F1-Score = 1.0000 (Perfect performance)
- **Segmentation**: Silhouette Score = 0.6078 (Good clustering)

## Data Dependencies
- Input: `../../data/processed/df_eng_customer_purchasing_features.csv`
- Output: `./results/model_predictions_random_forest.csv`

## MLflow Tracking
- Experiment Name: "Customer_Intelligence_Platform"
- Local tracking in `./mlruns/`
- Start MLflow UI: `mlflow ui --host 0.0.0.0 --port 5000`

## Usage
1. Navigate to this folder: `cd experiments/random_forest/`
2. Open Jupyter notebook: `random_forest_mlflow_models.ipynb`
3. Run all cells to train models and generate predictions
4. View MLflow dashboard for experiment tracking

## Team Collaboration Notes
- This folder is isolated for Random Forest experiments
- Other team members should create similar folders for their approaches:
  - `experiments/xgboost/`
  - `experiments/neural_networks/`
  - `experiments/ensemble/`
- All teams share the same input data from `data/processed/`
- Final model comparisons can be done in `experiments/ensemble/`
