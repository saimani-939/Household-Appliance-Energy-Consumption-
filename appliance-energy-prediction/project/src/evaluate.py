"""
Model evaluation utilities: regression metrics used to compare candidate
models (Linear Regression vs. Random Forest).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred, name: str) -> dict:
    """Compute MSE, RMSE, MAE, and R2 for a single set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": name,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_all(y_true, predictions: dict) -> pd.DataFrame:
    """
    Evaluate multiple models at once.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    predictions : dict
        Mapping of model name -> predicted values.

    Returns
    -------
    pd.DataFrame
        One row per model with MSE, RMSE, MAE, and R2 columns.
    """
    return pd.DataFrame([evaluate(y_true, p, n) for n, p in predictions.items()])
