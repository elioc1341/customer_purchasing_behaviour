# MLflow with VS Code - Local Setup Guide (No Docker)

## Prerequisites

- Python 3.8+ installed
- VS Code with Python extension
- Git (optional but recommended)

## Step 1: Project Structure Setup

Create your project directory:

```
ml-project/
├── requirements.txt
├── .env
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── utils.py
│   └── serve_model.py
├── data/
├── models/
├── notebooks/
├── mlruns/
└── logs/
```

## Step 2: Environment Setup

### Create `requirements.txt`:

```txt
mlflow==2.8.1
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
python-dotenv==1.0.0
click==8.1.7
```

### Create `.env` file:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
MLFLOW_EXPERIMENT_NAME=ml-classification-experiment

# Python Configuration
PYTHONPATH=./src
```

### Create `.gitignore`:

```
mlruns/
__pycache__/
*.pyc
*.pyo
.env
.vscode/settings.json
logs/
models/*.pkl
.DS_Store
```

### Setup Virtual Environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: MLflow Training Script

### Create `src/train.py`:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import os
import click
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking"""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default-experiment')
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def load_or_generate_data(data_path=None):
    """Load data from file or generate sample data"""
    if data_path and os.path.exists(data_path):
        # Load real data
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)
        y = data['target']
        logger.info(f"Loaded data from {data_path}")
    else:
        # Generate sample data
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=10,
            n_redundant=10,
            random_state=42
        )
        logger.info("Generated sample classification data")
    
    return X, y

def create_confusion_matrix_plot(y_true, y_pred, save_path='confusion_matrix.png'):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

@click.command()
@click.option('--n-estimators', default=100, help='Number of estimators')
@click.option('--max-depth', default=10, help='Maximum depth')
@click.option('--random-state', default=42, help='Random state')
@click.option('--data-path', default=None, help='Path to training data CSV')
@click.option('--run-name', default=None, help='MLflow run name')
@click.option('--grid-search', is_flag=True, help='Run grid search for hyperparameters')
def train_model(n_estimators, max_depth, random_state, data_path, run_name, grid_search):
    """Train model with MLflow tracking"""
    
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    with mlflow.start_run(run_name=run_name) as run:
        # Load data
        X, y = load_or_generate_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Log basic parameters
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", random_state)
        
        if grid_search:
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            model = RandomForestClassifier(random_state=random_state)
            grid_search_cv = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search_cv.fit(X_train, y_train)
            
            best_model = grid_search_cv.best_estimator_
            best_params = grid_search_cv.best_params_
            
            # Log grid search results
            mlflow.log_params(best_params)
            mlflow.log_param("grid_search", True)
            mlflow.log_param("cv_score", grid_search_cv.best_score_)
            
        else:
            # Train with specified parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("grid_search", False)
            
            best_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if len(set(y)) == 2 else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model, 
            "random_forest_model",
            registered_model_name="RandomForestClassifier"
        )
        
        # Create and log visualizations
        cm_path = create_confusion_matrix_plot(y_test, y_pred, 'confusion_matrix.png')
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)  # Clean up
        
        # Log feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(best_model.feature_importances_))],
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')
            os.remove('feature_importance.csv')  # Clean up
        
        # Log model info
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_features", X.shape[1])
        
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("Model logged successfully!")
        
        return run.info.run_id

if __name__ == "__main__":
    train_model()
```

## Step 4: Model Registry and Management

### Create `src/utils.py`:

```python
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def get_best_model(experiment_name, metric="accuracy"):
    """Get the best model from an experiment"""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in experiment")
    
    return runs[0]

def register_best_model(experiment_name, model_name, metric="accuracy"):
    """Register the best model from an experiment"""
    best_run = get_best_model(experiment_name, metric)
    model_uri = f"runs:/{best_run.info.run_id}/random_forest_model"
    
    # Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Model registered: {model_name} version {model_version.version}")
    print(f"From run: {best_run.info.run_id}")
    print(f"Best {metric}: {best_run.data.metrics[metric]:.4f}")
    
    return model_version

def compare_runs(experiment_name, top_n=5):
    """Compare top N runs from an experiment"""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=top_n
    )
    
    comparison_data = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id[:8],
            'accuracy': run.data.metrics.get('accuracy', 0),
            'f1_score': run.data.metrics.get('f1_score', 0),
            'n_estimators': run.data.params.get('n_estimators'),
            'max_depth': run.data.params.get('max_depth'),
            'status': run.info.status
        }
        comparison_data.append(run_data)
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    return df
```

### Create `src/serve_model.py`:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import click
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Global model variable
model = None
model_name = None

def load_model_by_version(name, version):
    """Load a specific version of a registered model"""
    global model, model_name
    model_uri = f"models:/{name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    model_name = f"{name}_v{version}"
    print(f"Loaded model: {model_name}")

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Make prediction
        predictions = model.predict(df)
        probabilities = model.predict_proba(df).tolist() if hasattr(model, 'predict_proba') else None
        
        response = {
            'predictions': predictions.tolist(),
            'model': model_name
        }
        
        if probabilities:
            response['probabilities'] = probabilities
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': model_name
    })

@click.command()
@click.option('--model-name', required=True, help='Name of registered model')
@click.option('--version', default='latest', help='Model version to serve')
@click.option('--port', default=5001, help='Port to run the server')
def serve(model_name, version, port):
    """Serve a registered MLflow model"""
    # Setup MLflow
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load model
    load_model_by_version(model_name, version)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    serve()
```

## Step 5: Essential Commands

### Start MLflow Server:

```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Or start with file backend (simpler)
mlflow ui --host 0.0.0.0 --port 5000
```

### Training Commands:

```bash
# Basic training
python src/train.py

# Training with custom parameters
python src/train.py --n-estimators 200 --max-depth 15 --run-name "experiment_1"

# Training with grid search
python src/train.py --grid-search --run-name "grid_search_experiment"

# Training with custom data
python src/train.py --data-path ./data/my_dataset.csv
```

### Model Management Commands:

```bash
# Compare runs (in Python)
python -c "
from src.utils import compare_runs
compare_runs('ml-classification-experiment', top_n=5)
"

# Register best model
python -c "
from src.utils import register_best_model
register_best_model('ml-classification-experiment', 'ProductionModel')
"

# Serve model
python src/serve_model.py --model-name ProductionModel --version 1 --port 5001
```

## Step 6: VS Code Integration

### `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/train.py",
            "console": "integratedTerminal",
            "args": ["--run-name", "debug_run"],
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Train with Grid Search",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/train.py",
            "console": "integratedTerminal",
            "args": ["--grid-search", "--run-name", "debug_grid_search"],
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Serve Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/serve_model.py",
            "console": "integratedTerminal",
            "args": ["--model-name", "ProductionModel", "--version", "1"],
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

### `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start MLflow Server",
            "type": "shell",
            "command": "mlflow",
            "args": ["ui", "--host", "0.0.0.0", "--port", "5000"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "isBackground": true,
            "problemMatcher": []
        },
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "pip",
            "args": ["install", "-r", "requirements.txt"],
            "group": "build"
        },
        {
            "label": "Run Training",
            "type": "shell",
            "command": "python",
            "args": ["src/train.py", "--run-name", "task_run"],
            "group": "test",
            "dependsOrder": "sequence"
        },
        {
            "label": "Compare Models",
            "type": "shell",
            "command": "python",
            "args": ["-c", "from src.utils import compare_runs; compare_runs('ml-classification-experiment')"],
            "group": "test"
        }
    ]
}
```

## Step 7: Complete Workflow

### Daily Development Workflow:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start MLflow server (in separate terminal)
mlflow ui --port 5000

# 3. Run experiments
python src/train.py --run-name "experiment_1"
python src/train.py --grid-search --run-name "grid_search_1"

# 4. Compare results
python -c "from src.utils import compare_runs; compare_runs('ml-classification-experiment')"

# 5. Register best model
python -c "from src.utils import register_best_model; register_best_model('ml-classification-experiment', 'MyModel')"

# 6. Serve model
python src/serve_model.py --model-name MyModel --version 1

# 7. Test model endpoint
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'
```

## Step 8: Jupyter Notebook Integration

### Create notebook with MLflow:

```python
# In Jupyter notebook
import mlflow
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment('notebook-experiments')

# Your ML code with MLflow tracking
with mlflow.start_run():
    # ... your code here
    mlflow.log_param("notebook", True)
    mlflow.log_metric("accuracy", 0.95)
```

## Best Practices

1. **Virtual Environment**: Always use virtual environments
2. **Environment Variables**: Use `.env` files for configuration
3. **Version Control**: Exclude `mlruns/` from git
4. **Naming**: Use descriptive run names and tags
5. **Organization**: Group related experiments
6. **Cleanup**: Regularly clean old experiments
7. **Documentation**: Document your experiments and parameters

## Troubleshooting

- **Port conflicts**: Use different ports if 5000 is taken
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Database locks**: Stop MLflow server before moving files
- **Memory issues**: Limit concurrent runs for large models

This local setup is much simpler than Docker while providing all the same MLflow functionality!