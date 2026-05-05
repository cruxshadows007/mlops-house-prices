import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn

DATA_PATH = "data/processed/train.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"], errors="ignore")

    # solo numéricas
    X = X.select_dtypes(include=["int64", "float64"]).fillna(0)

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

def save_artifacts(model, features):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(features, "models/features.pkl")

    print("Model and features saved!")

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return mae, rmse

def train():
    (X_train, X_test, y_train, y_test), feature_names = load_data()

    mlflow.set_experiment("house-prices-multi-model")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "GradientBoosting": GradientBoostingRegressor(),
        "LinearRegression": LinearRegression()
    }

    best_rmse = float("inf")
    best_model = None
    best_name = None

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            mae, rmse = evaluate(model, X_train, X_test, y_train, y_test)

            mlflow.log_param("model", name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name

    print(f"\nBEST MODEL: {best_name} | RMSE: {best_rmse}")

    # reentrenar mejor modelo en todo el dataset
    best_model.fit(X_train, y_train)

    save_artifacts(best_model, feature_names)

if __name__ == "__main__":
    train()