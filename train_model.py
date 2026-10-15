"""
train_model.py
Trains a rainfall model with Scikit-learn and saves it with Joblib.

Run it from the project folder:
    python train_model.py                          (uses data/sample_weather.csv)
    python train_model.py data/uploaded_weather.csv
"""
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import data_utils

# The model ALWAYS uses these 4 inputs in this order (training and prediction must match)
FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]
TARGET = "rainfall"

MODEL_PATH = os.path.join(data_utils.BASE_DIR, "models", "rainfall_model.joblib")
TEST_SIZE = 0.2                 # last 20% of the days are kept for testing
MIN_ROWS = 30                   # fewer rows than this: refuse to train
MIN_ROWS_FOR_FOREST = 100       # fewer rows than this: use the simple baseline instead


def rainfall_category(mm):
    """
    Simple categories used in this project (our own thresholds, NOT official ones):
        below 0.5 mm   -> No rain
        0.5 to 7.5 mm  -> Light rain
        7.5 to 35 mm   -> Moderate rain
        35 mm or more  -> Heavy rain
    """
    if mm < 0.5:
        return "No rain"
    elif mm < 7.5:
        return "Light rain"
    elif mm < 35:
        return "Moderate rain"
    else:
        return "Heavy rain"


def evaluate(y_true, y_predicted):
    """MAE = average size of the error. RMSE = like MAE but punishes big errors more."""
    mae = mean_absolute_error(y_true, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2)}


def chronological_split(df, test_size=TEST_SIZE):
    """
    Time-series split: train on the OLDER days, test on the NEWER days.
    (A random split would let the model 'peek' at the future.)
    """
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[:split_index], df.iloc[split_index:]


def train_and_save(data_path=None):
    """Train, evaluate and save the model. Returns the saved information as a dictionary."""
    if data_path is None:
        data_path = data_utils.DATA_PATH

    df = data_utils.load_weather_data(data_path)      # loads and cleans the data
    if len(df) < MIN_ROWS:
        raise ValueError("Only " + str(len(df)) + " rows found. At least "
                         + str(MIN_ROWS) + " rows are needed to train a model.")

    train, test = chronological_split(df)
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    # Baseline: always predicts the average rainfall of the training days
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    baseline_scores = evaluate(y_test, baseline.predict(x_test))

    # Main model
    if len(df) >= MIN_ROWS_FOR_FOREST:
        model = RandomForestRegressor(n_estimators=100, max_depth=8,
                                      min_samples_leaf=3, random_state=42)
        model_name = "Random Forest Regressor"
    else:
        model = DummyRegressor(strategy="mean")
        model_name = "Baseline (average rainfall) - dataset too small for a forest"
    model.fit(x_train, y_train)
    scores = evaluate(y_test, model.predict(x_test))

    importance = {}
    if hasattr(model, "feature_importances_"):
        for name, value in zip(FEATURES, model.feature_importances_):
            importance[name] = round(float(value), 3)

    bundle = {
        "model": model,
        "features": FEATURES,
        "model_name": model_name,
        "metrics": scores,
        "baseline_metrics": baseline_scores,
        "beats_baseline": scores["mae"] < baseline_scores["mae"],
        "importance": importance,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_period": test["date"].min().strftime("%Y-%m-%d") + " to " + test["date"].max().strftime("%Y-%m-%d"),
        "data_file": os.path.basename(data_path),
        "trained_at": datetime.now().strftime("%d %b %Y %H:%M"),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model():
    """Load the saved model. Returns None if it has not been trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict_rainfall(bundle, values):
    """
    values is a dictionary such as
        {"temperature": 30, "humidity": 80, "pressure": 1005, "wind_speed": 12}
    Returns the estimated rainfall in mm (never below 0).
    """
    row = pd.DataFrame([values], columns=bundle["features"])   # same column order as training
    estimate = float(bundle["model"].predict(row)[0])
    return max(0.0, round(estimate, 1))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        result = train_and_save(path)
    except ValueError as error:
        print("Training failed:", error)
        sys.exit(1)

    print("Model trained and saved to:", MODEL_PATH)
    print("Model            :", result["model_name"])
    print("Data file        :", result["data_file"])
    print("Training rows    :", result["train_rows"])
    print("Testing rows     :", result["test_rows"], "(" + result["test_period"] + ")")
    print("Model  MAE / RMSE:", result["metrics"]["mae"], "/", result["metrics"]["rmse"], "mm")
    print("Baseline MAE/RMSE:", result["baseline_metrics"]["mae"], "/", result["baseline_metrics"]["rmse"], "mm")
    if result["beats_baseline"]:
        print("Result           : the model has a lower MAE than the baseline on the test days.")
    else:
        print("Result           : the model is NOT better than the baseline. Do not rely on it.")
    if result["importance"]:
        print("Feature importance:", result["importance"])