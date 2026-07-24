# Appliance Energy Prediction

Predicting household appliance energy consumption (Wh) from indoor sensor
readings, outdoor weather conditions, and time-of-day patterns, using the
UCI "Appliances Energy Prediction" dataset.

## Overview

- **Dataset:** 19,735 readings taken at 10-minute intervals over ~4.5 months
  (Jan–May 2016) from a low-energy house in Belgium, combined with weather
  data from a nearby airport station.
- **Target:** `Appliances` — appliance energy use in Wh per 10-minute interval.
- **Models compared:** Linear Regression (baseline) vs. Random Forest Regressor.
- **Result:** Random Forest outperforms Linear Regression on every metric
  (RMSE, MAE, R²), confirming that the relationship between sensor readings
  and appliance usage is non-linear and interaction-driven.

## Key findings

1. **Time of day is the strongest predictor** — usage peaks in the morning
   (~8am) and evening (~6–8pm), consistent with typical occupied-home routines.
2. Laundry room temperature/humidity and kitchen/bathroom humidity carry more
   signal than outdoor weather variables.
3. Behavioral/occupancy timing dominates over climate conditions, suggesting
   energy-saving programs should prioritize **usage-timing interventions**
   (smart scheduling, time-of-use pricing) over weather-based ones.
4. ~43% of variance remains unexplained, concentrated in rare high-usage
   spikes — likely needing appliance-level submetering or occupancy sensors
   to resolve further.

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                      # place energydata_complete.csv here (not tracked)
├── models/                    # trained model artifacts (generated, not tracked)
├── reports/                   # summary.json metrics report (generated, not tracked)
├── notebooks/
│   └── appliance_energy_prediction.ipynb   # original EDA / exploration notebook
└── src/
    ├── __init__.py
    ├── config.py              # paths, constants, hyperparameters
    ├── data_loader.py         # CSV loading + timestamp parsing
    ├── features.py            # time-based feature engineering
    ├── evaluate.py            # regression metrics (MSE, RMSE, MAE, R2)
    ├── train.py                # full training pipeline (fit, evaluate, save)
    └── predict.py              # load saved model + score new readings
```

## Setup

```bash
git clone <your-repo-url>
cd appliance-energy-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset ([UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction))
and place `energydata_complete.csv` in the `data/` folder.

## Usage

### Train the models

```bash
python -m src.train
```

This loads the data, engineers features (cyclical hour encoding, weekend
flag, dropping the `rv1`/`rv2` noise columns), trains both models, evaluates
them on a held-out test set, and saves the following to `models/` and
`reports/`:

- `random_forest_model.pkl`
- `linear_regression_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`
- `summary.json` (metrics + top feature importances)

### Predict on new data

```bash
python -m src.predict --input path/to/new_readings.csv --output predictions.csv
```

The input CSV must have the same raw columns as the training data (including
`date`), minus the `Appliances` target. Use `--model linear_regression` to
score with the baseline model instead of the default Random Forest.

### Explore the notebook

The original exploratory notebook (`notebooks/appliance_energy_prediction.ipynb`)
contains the full EDA, visualizations, and narrative walkthrough that the
`src/` modules were refactored from.

## Methodology

1. **EDA** — target distribution, correlation heatmap, and hourly/weekly
   usage patterns.
2. **Feature engineering** — cyclical hour encoding (`hour_sin`, `hour_cos`),
   `is_weekend` flag, dropping the `rv1`/`rv2` noise columns confirmed to
   carry no signal.
3. **Modeling** — Linear Regression on standardized features vs. Random
   Forest (200 trees) on raw features.
4. **Evaluation** — MSE, RMSE, MAE, R² on an 80/20 train/test split
   (`random_state=42`).
5. **Feature importance** — Random Forest mean-decrease-in-impurity scores.

## Possible next steps

- Hyperparameter tuning (grid/random search on the Random Forest)
- Gradient boosting models (XGBoost, LightGBM)
- Lag features capturing short-term usage momentum
- Appliance-level submetering or occupancy sensor data to explain high-usage spikes

## License

Add a license of your choice (e.g. MIT) before publishing.
