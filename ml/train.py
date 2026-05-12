import pickle
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

from config import (
    MODEL_FORECAST, MODEL_ANOMALY, MODEL_CLASSIFIER,
    SAVED_MODELS_DIR, MIN_RECORDS_TO_TRAIN, TEST_SIZE, RANDOM_STATE
)
from database.db import get_all
from ml.preprocess import (
    rows_to_dataframe, clean_data, add_features,
    fit_and_save_scaler, make_classifier_labels, make_forecast_target
)


os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


# =====================================================
# TRAIN FORECAST MODEL
# Dự đoán PM2.5 tiếp theo
# =====================================================
def train_forecast(df):
    print("\n[Train] Forecast model...")

    target = make_forecast_target(df, steps_ahead=1)
    df_train = df.copy()
    df_train["target"] = target

    df_train = df_train.dropna(subset=["target"])

    feature_cols = [c for c in df_train.columns
                    if c not in ["id", "timestamp", "target"]]

    X = df_train[feature_cols].values
    y = df_train["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)

    print(f"[Train] Forecast MAE: {mae:.2f} µg/m³")

    with open(MODEL_FORECAST, "wb") as f:
        pickle.dump((model, feature_cols), f)

    print(f"[Train] Forecast model saved → {MODEL_FORECAST}")

    return mae


# =====================================================
# TRAIN ANOMALY MODEL
# Phát hiện bất thường (Isolation Forest)
# =====================================================
def train_anomaly(df):
    print("\n[Train] Anomaly detection model...")

    feature_cols = [c for c in df.columns
                    if c not in ["id", "timestamp"]]

    X = df[feature_cols].values

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=RANDOM_STATE
    )

    model.fit(X)

    scores  = model.decision_function(X)
    preds   = model.predict(X)
    n_anom  = np.sum(preds == -1)

    print(f"[Train] Anomalies found in training data: {n_anom}/{len(X)}")

    with open(MODEL_ANOMALY, "wb") as f:
        pickle.dump((model, feature_cols), f)

    print(f"[Train] Anomaly model saved → {MODEL_ANOMALY}")

    return n_anom


# =====================================================
# TRAIN CLASSIFIER MODEL
# Phân loại: Good / Normal / Bad / Danger
# =====================================================
def train_classifier(df):
    print("\n[Train] Classifier model...")

    labels = make_classifier_labels(df)
    df_train = df.copy()
    df_train["label"] = labels

    feature_cols = [c for c in df_train.columns
                    if c not in ["id", "timestamp", "label"]]

    X = df_train[feature_cols].values
    y = df_train["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[Train] Classifier accuracy: {accuracy:.2%}")

    with open(MODEL_CLASSIFIER, "wb") as f:
        pickle.dump((model, feature_cols), f)

    print(f"[Train] Classifier model saved → {MODEL_CLASSIFIER}")

    return accuracy


# =====================================================
# MAIN TRAIN PIPELINE
# =====================================================
def run_training():
    print("=" * 50)
    print("[Train] Starting training pipeline...")
    print("=" * 50)

    rows = get_all()

    if len(rows) < MIN_RECORDS_TO_TRAIN:
        msg = (f"[Train] Not enough data: "
               f"{len(rows)}/{MIN_RECORDS_TO_TRAIN} records")
        print(msg)
        return {"success": False, "message": msg}

    print(f"[Train] Records loaded: {len(rows)}")

    # Preprocess
    df = rows_to_dataframe(rows)
    df = clean_data(df)
    df = add_features(df)

    _, feature_cols = fit_and_save_scaler(df)

    # Train
    mae      = train_forecast(df)
    n_anom   = train_anomaly(df)
    accuracy = train_classifier(df)

    result = {
        "success":           True,
        "records_used":      len(df),
        "forecast_mae":      round(mae, 2),
        "anomalies_found":   int(n_anom),
        "classifier_accuracy": round(accuracy, 4),
    }

    print("\n" + "=" * 50)
    print("[Train] Done!")
    print(result)
    print("=" * 50)

    return result


if __name__ == "__main__":
    run_training()