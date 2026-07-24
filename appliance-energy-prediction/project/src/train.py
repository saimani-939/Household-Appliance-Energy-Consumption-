"""
End-to-end training pipeline for the appliance energy prediction models.

Run directly to reproduce the full workflow used in the notebook:
    load data -> engineer features -> train/test split -> scale ->
    train Linear Regression + Random Forest -> evaluate -> save artifacts.

Usage
-----
    python -m src.train
"""

import json
import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import load_data
from src.evaluate import evaluate_all
from src.features import build_feature_matrix


def train_models(X_train, y_train):
    """
    Fit a Linear Regression baseline and a Random Forest Regressor.

    The Linear Regression model expects pre-scaled input; the Random
    Forest is fit directly on raw (unscaled) feature values.

    Returns
    -------
    dict
        {"scaler": StandardScaler, "linear_regression": LinearRegression,
         "random_forest": RandomForestRegressor}
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
    rf.fit(X_train, y_train)

    return {"scaler": scaler, "linear_regression": lr, "random_forest": rf}


def run_pipeline(data_path: str = config.RAW_DATA_PATH) -> dict:
    """Run the full training pipeline and persist model artifacts to disk."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    df = load_data(data_path)
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    artifacts = train_models(X_train, y_train)
    scaler, lr, rf = (
        artifacts["scaler"],
        artifacts["linear_regression"],
        artifacts["random_forest"],
    )

    X_test_scaled = scaler.transform(X_test)
    predictions = {
        "Linear Regression": lr.predict(X_test_scaled),
        "Random Forest": rf.predict(X_test),
    }

    metrics_df = evaluate_all(y_test, predictions)
    best_model_name = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]

    importances = (
        __import__("pandas")
        .Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    # Persist model artifacts
    joblib.dump(rf, config.RANDOM_FOREST_MODEL_PATH)
    joblib.dump(lr, config.LINEAR_REGRESSION_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(list(X.columns), config.FEATURE_COLUMNS_PATH)

    summary = {
        "best_model": best_model_name,
        "n_features": X.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "metrics": metrics_df.to_dict(orient="records"),
        "top5_features": importances.head(5).to_dict(),
    }
    with open(config.SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Artifacts saved to: {config.MODEL_DIR}")

    return summary


if __name__ == "__main__":
    run_pipeline()
