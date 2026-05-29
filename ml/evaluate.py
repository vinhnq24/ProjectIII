import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report
)
from config import MODEL_FORECAST, MODEL_CLASSIFIER
from database.db import get_all
from ml.preprocess import (
    rows_to_dataframe, clean_data, add_features,
    make_forecast_target, make_classifier_labels
)


def _time_split_xy(X, y, test_size=0.2):
    n = len(X)
    if n < 2:
        raise ValueError("Not enough samples to evaluate.")

    n_test = max(1, int(n * test_size))
    n_test = min(n_test, n - 1)
    split_idx = n - n_test

    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def evaluate_forecast():
    print("[Evaluate] Forecast model...")

    if not os.path.exists(MODEL_FORECAST):
        print("[Evaluate] Model not found. Run train.py first.")
        return

    with open(MODEL_FORECAST, "rb") as f:
        model, feature_cols = pickle.load(f)

    rows = get_all()
    df   = rows_to_dataframe(rows)
    df   = clean_data(df)
    df   = add_features(df)

    target = make_forecast_target(df, steps_ahead=1)
    df["target"] = target
    df = df.dropna(subset=["target"])

    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = _time_split_xy(X, y)
    y_pred = model.predict(X_test)
    naive_pred = X_test[:, feature_cols.index("pm25")]

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    naive_mae = mean_absolute_error(y_test, naive_pred)

    print(f"  MAE (test)       : {mae:.2f} µg/m³")
    print(f"  RMSE (test)      : {rmse:.2f} µg/m³")
    print(f"  Naive MAE (test) : {naive_mae:.2f} µg/m³")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "naive_mae": float(naive_mae),
        "n_test": int(len(y_test)),
    }


def evaluate_classifier():
    print("[Evaluate] Classifier model...")

    if not os.path.exists(MODEL_CLASSIFIER):
        print("[Evaluate] Model not found. Run train.py first.")
        return

    with open(MODEL_CLASSIFIER, "rb") as f:
        model, feature_cols = pickle.load(f)

    rows = get_all()
    df   = rows_to_dataframe(rows)
    df   = clean_data(df)
    df   = add_features(df)

    labels = make_classifier_labels(df)
    X = df[feature_cols].values
    y = labels.values
    X_train, X_test, y_train, y_test = _time_split_xy(X, y)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_map = {0: "Good", 1: "Normal", 2: "Bad", 3: "Danger"}
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [class_map.get(int(lbl), str(lbl)) for lbl in labels_present]
    report = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0
    )

    print(f"  Accuracy (test) : {accuracy:.2%}")
    print(report)

    return {"accuracy": float(accuracy), "report": report, "n_test": int(len(y_test))}


if __name__ == "__main__":
    evaluate_forecast()
    evaluate_classifier()