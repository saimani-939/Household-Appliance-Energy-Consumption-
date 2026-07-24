"""
Inference helper: load the saved artifacts and generate predictions for
new, unseen sensor readings.

Usage
-----
    python -m src.predict --input path/to/new_readings.csv --output preds.csv
"""

import argparse

import joblib
import pandas as pd

from src import config
from src.features import add_time_features, drop_noise_columns


def load_artifacts(model_name: str = "random_forest"):
    """
    Load a trained model plus the fitted scaler and feature column order.

    Parameters
    ----------
    model_name : str
        Either "random_forest" (default, best performer) or
        "linear_regression".
    """
    model_path = (
        config.RANDOM_FOREST_MODEL_PATH
        if model_name == "random_forest"
        else config.LINEAR_REGRESSION_MODEL_PATH
    )
    model = joblib.load(model_path)
    scaler = joblib.load(config.SCALER_PATH)
    feature_columns = joblib.load(config.FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


def prepare_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Apply the same feature engineering used at training time."""
    df = df.copy()
    df[config.DATE_COLUMN] = pd.to_datetime(
        df[config.DATE_COLUMN], format=config.DATE_FORMAT
    )
    df = add_time_features(df)
    df = drop_noise_columns(df)
    df = df.drop(columns=[config.DATE_COLUMN])
    return df[feature_columns]


def predict(df: pd.DataFrame, model_name: str = "random_forest") -> pd.Series:
    """
    Generate appliance-energy-use predictions (Wh) for new sensor readings.

    Parameters
    ----------
    df : pd.DataFrame
        Raw readings with the same columns as the training data
        (including a ``date`` column), minus the target.
    model_name : str
        "random_forest" (default) or "linear_regression".
    """
    model, scaler, feature_columns = load_artifacts(model_name)
    X = prepare_features(df, feature_columns)

    if model_name == "linear_regression":
        X = scaler.transform(X)

    return pd.Series(model.predict(X), index=df.index, name="predicted_appliances_wh")


def main():
    parser = argparse.ArgumentParser(description="Predict appliance energy use.")
    parser.add_argument("--input", required=True, help="CSV of new sensor readings.")
    parser.add_argument("--output", default=None, help="Where to write predictions CSV.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "linear_regression"],
        help="Which trained model to use.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds = predict(df, model_name=args.model)

    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    else:
        print(preds)


if __name__ == "__main__":
    main()
