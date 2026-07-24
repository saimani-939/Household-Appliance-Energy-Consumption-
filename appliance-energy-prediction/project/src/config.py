"""
Central configuration: paths, constants, and hyperparameters used across the
data loading, feature engineering, training, and prediction modules.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

RAW_DATA_PATH = os.path.join(DATA_DIR, "energydata_complete.csv")

RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
LINEAR_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SUMMARY_PATH = os.path.join(REPORTS_DIR, "summary.json")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
DATE_COLUMN = "date"
DATE_FORMAT = "%d-%m-%Y %H:%M"  # source timestamps are day-first
TARGET_COLUMN = "Appliances"

# Random noise columns included in the original UCI dataset as a
# distractor / sanity-check pair -- confirmed to have ~0 correlation
# with the target during EDA, so they are dropped before modeling.
NOISE_COLUMNS = ["rv1", "rv2"]

# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
