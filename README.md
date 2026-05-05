# mlops-house-prices

1. Project Overview
This project implements a full Machine Learning pipeline using MLOps best practices.
It includes data versioning (DVC), experiment tracking (MLflow), and model deployment (FastAPI).

The goal is to predict housing prices using the Kaggle House Prices dataset.

2. System Architecture

                         ┌──────────────────────────┐
                         │      GitHub Repo         │
                         │  (source of truth)       │
                         │  - code versioning       │
                         │  - README                │
                         └──────────┬───────────────┘
                                    │ clone/push
                                    ▼
                         ┌──────────────────────────┐
                         │     WSL Ubuntu           │
                         │ (execution environment)   │
                         └──────────┬───────────────┘
                                    │
                                    ▼
                   ┌─────────────────────────────────┐
                   │        Kaggle Dataset           │
                   │ House Prices (raw data)         │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │          DVC Layer              │
                   │  - version raw data             │
                   │  - reproducible pipeline        │
                   │  - processed dataset            │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │     Feature Engineering         │
                   │  - cleaning                     │
                   │  - encoding                    │
                   │  - feature selection           │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │       Model Training            │
                   │ (Scikit-learn models)           │
                   └──────────────┬──────────────────┘
                                  │
                 ┌────────────────┴─────────────────┐
                 ▼                                  ▼
 ┌────────────────────────────┐     ┌────────────────────────────┐
 │      MLflow Tracking       │     │   Model Artifacts          │
 │ - params                   │     │ - best model (.pkl)       │
 │ - metrics (MAE, RMSE)     │     │ - saved locally            │
 │ - experiment comparison    │     └────────────┬───────────────┘
 └──────────────┬─────────────┘                  │
                ▼                                ▼
     ┌────────────────────────────┐   ┌────────────────────────────┐
     │   Model Selection Layer    │   │   Model Registry (MLflow)  │
     │   best run chosen          │   │   production candidate      │
     └──────────────┬─────────────┘   └────────────┬───────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                     ┌────────────────────────────┐
                     │       FastAPI Service      │
                     │   /predict endpoint        │
                     └────────────┬───────────────┘
                                  ▼
                     ┌────────────────────────────┐
                     │     Client (curl / app)    │
                     └────────────────────────────┘
                     
3. Dataset
Source:
Kaggle - House Prices: Advanced Regression Techniques

Type:
Regression problem

Size:
Several thousand samples

Target:
SalePrice

4. Pipeline Overview

4.1. Raw data downloaded from Kaggle
4.2. Data versioned using DVC
4.3. Preprocessing pipeline applied
4.4. Feature engineering performed
4.5. Multiple ML models trained
4.6. Experiments tracked using MLflow
4.7. Best model selected
4.8. Model deployed using FastAPI

5. Project Structure
mlops-house-prices/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   ├── features/
│   └── models/
│
├── api/
│   └── main.py
│
├── models/
├── notebooks/
├── dvc.yaml
├── mlruns/
├── README.md
└── requirements.txt

6. Experiment Tracking (MLflow)
MLflow was used to track:
- model parameters
- evaluation metrics (MAE, RMSE)
- trained models

Multiple experiments were executed to compare models and optimize hyperparameters.

7. Model Selection Strategy
The best model was selected based on RMSE and MAE metrics.
RandomForest was used as baseline and compared with alternative configurations.

8. API Deployment (FastAPI)
The trained model is exposed via a REST API built with FastAPI.

Endpoint:
/predict

Method:
POST

Input:
House features (JSON)

Output:
Predicted price

9. How to Run the Project
# Clone repository
git clone <repo-url>
cd mlops-house-prices

# Create environment
conda create -n mlops-house python=3.10 -y
conda activate mlops-house

# Install dependencies
pip install -r requirements.txt

# Pull data (DVC)
dvc pull

# Train model
python src/models/train.py

# Run MLflow UI
mlflow ui

# Run API
uvicorn api.main:app --reload

10. API Testing
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"GrLivArea": 1500, "OverallQual": 7}'

11. Reproducibility Statement
This project is fully reproducible.
Any user can clone the repository and reproduce the entire pipeline using DVC and MLflow tracking.

12. Author
MLOps academic project - House Prices Prediction System
