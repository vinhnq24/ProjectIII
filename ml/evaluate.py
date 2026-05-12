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

    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"  MAE  : {mae:.2f} µg/m³")
    print(f"  RMSE : {rmse:.2f} µg/m³")

    return {"mae": mae, "rmse": rmse}


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

    labels   = make_classifier_labels(df)
    X        = df[feature_cols].values
    y        = labels.values
    y_pred   = model.predict(X)

    accuracy = accuracy_score(y, y_pred)

    print(f"  Accuracy : {accuracy:.2%}")
    print(classification_report(
        y, y_pred,
        target_names=["Good", "Normal", "Bad", "Danger"],
        zero_division=0
    ))

    return {"accuracy": accuracy}


if __name__ == "__main__":
    evaluate_forecast()
    evaluate_classifier()