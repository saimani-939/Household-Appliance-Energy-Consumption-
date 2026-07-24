"""
Data loading utilities for the appliance energy prediction dataset.
"""

import pandas as pd

from src import config


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw energy dataset from CSV and parse the timestamp column.

    Parameters
    ----------
    path : str
        Path to the ``energydata_complete.csv`` file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with the ``date`` column parsed as a datetime.
    """
    df = pd.read_csv(path)
    df[config.DATE_COLUMN] = pd.to_datetime(
        df[config.DATE_COLUMN], format=config.DATE_FORMAT
    )
    return df


def data_summary(df: pd.DataFrame) -> dict:
    """Return a small dict summary of the dataset, useful for logging/EDA."""
    return {
        "shape": df.shape,
        "date_min": str(df[config.DATE_COLUMN].min()),
        "date_max": str(df[config.DATE_COLUMN].max()),
        "missing_values": int(df.isna().sum().sum()),
    }


if __name__ == "__main__":
    dataframe = load_data()
    print(data_summary(dataframe))
