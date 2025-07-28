# Team Model Experiments

## Folder Structure for Collaboration

Each team member should create their own folder for their modeling approach:

```
experiments/
├── random_forest/          # Random Forest models (Team Member 1)
├── xgboost/               # XGBoost models (Team Member 2)
├── neural_networks/       # Deep Learning models (Team Member 3)
├── svm/                   # Support Vector Machine models (Team Member 4)
├── ensemble/              # Final ensemble models (All team)
└── README.md              # This file
```

## Guidelines

### 1. Individual Model Folders
Each folder should contain:
- Main notebook with all models
- `models/` subfolder for saved models (.pkl files) - **isolated per experiment**
- `mlruns/` subfolder for MLflow tracking - **isolated per experiment**
- `results/` subfolder for prediction outputs and reports
- `README.md` with model results and usage

**Example folder structure:**
```
random_forest/
├── random_forest_mlflow_models.ipynb
├── models/                     # Models isolated per experiment
│   ├── clv_random_forest_model.pkl
│   ├── churn_random_forest_model.pkl
│   ├── segmentation_kmeans_model.pkl
│   └── segmentation_scaler.pkl
├── results/
│   └── model_predictions_random_forest.csv
├── mlruns/                     # MLflow tracking isolated per experiment
│   └── 807182511413173702/     # Customer_Intelligence_Platform experiment
└── README.md
```

### 2. Naming Conventions
- Models: `{model_type}_{algorithm}_{task}.pkl` (saved in `./models/` within experiment folder)
  - Example: `./models/clv_random_forest_regression.pkl`
- Predictions: `model_predictions_{algorithm}.csv` (saved in `./results/`)
  - Example: `./results/model_predictions_random_forest.csv`
- MLflow Experiments: Use descriptive names within isolated `./mlruns/` directories
  - Example: "Customer_Intelligence_Platform" (within `./mlruns/`)

### 3. Shared Resources
- **Data**: All teams use `../../data/processed/df_eng_customer_purchasing_features.csv`
- **Output**: Save predictions to `./results/model_predictions_{your_algorithm}.csv` within your experiment folder
- **Documentation**: Update this README with your results

### 4. MLflow Experiment Tracking
- Each team member uses their own experiment name within their isolated folder
- MLflow runs are stored locally in each `./mlruns/` directory for complete isolation
- Port suggestions for viewing individual experiments:
  - Random Forest: `cd random_forest && mlflow ui --port 5000`
  - XGBoost: `cd xgboost && mlflow ui --port 5001` 
  - Neural Networks: `cd neural_networks && mlflow ui --port 5002`
  - SVM: `cd svm && mlflow ui --port 5003`
  - Ensemble: `cd ensemble && mlflow ui --port 5004`

**Benefits of isolated MLflow tracking:**
- No experiment conflicts between team members
- Complete reproducibility within each algorithm folder
- Easy to compare different algorithm approaches
- Clean separation for final ensemble comparison

## Model Comparison Template

| Algorithm | CLV R² | Churn F1 | Segmentation Silhouette | Notes |
|-----------|---------|----------|-------------------------|-------|
| Random Forest | 0.9995 | 1.0000 | 0.6078 | Excellent performance |
| XGBoost | TBD | TBD | TBD | |
| Neural Networks | TBD | TBD | TBD | |
| SVM | TBD | TBD | TBD | |

## Final Ensemble Strategy
Once all individual models are complete:
1. Collect all predictions from each team's `results/` folder into `ensemble/`
2. Load all models from each team's `models/` folder for ensemble creation
3. Compare model performance across different algorithms using isolated MLflow tracking
4. Create ensemble models combining best approaches
5. Document final model selection and deployment strategy

**Benefits of complete experiment isolation:**
- Each algorithm approach is completely self-contained
- Easy to package and share individual experiments
- No conflicts between team members' work
- Simple model comparison and ensemble creation
- Reproducible results within each experiment folder
