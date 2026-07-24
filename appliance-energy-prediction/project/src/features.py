"""
Feature engineering for the appliance energy prediction dataset.

Adds time-based features (hour, day-of-week, cyclical hour encoding,
weekend flag) and drops the known noise columns before the
train/target split.
"""

import numpy as np
import pandas as pd

from src import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, cyclical hour, and weekend features."""
    df = df.copy()
    df["hour"] = df[config.DATE_COLUMN].dt.hour
    df["dayofweek"] = df[config.DATE_COLUMN].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encoding so 23:00 and 00:00 are recognized as close together
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rv1/rv2 random noise columns confirmed during EDA."""
    return df.drop(columns=config.NOISE_COLUMNS, errors="ignore")


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full feature engineering pipeline and split into X, y.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by ``data_loader.load_data``.

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
        Feature matrix and target series, ready for train/test split.
    """
    df = add_time_features(df)
    df = drop_noise_columns(df)

    model_df = df.drop(columns=[config.DATE_COLUMN])
    X = model_df.drop(columns=[config.TARGET_COLUMN])
    y = model_df[config.TARGET_COLUMN]
    return X, y
