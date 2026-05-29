import pickle
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
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


def save_model_with_backup(model, feature_cols, primary_path):
    """
    Lưu model vào path chính và đồng thời tạo một bản sao lưu (archive) có đánh dấu thời gian.
    """
    import shutil
    from datetime import datetime

    # 1. Lưu vào path chính
    with open(primary_path, "wb") as f:
        pickle.dump((model, feature_cols), f)

    # 2. Tạo bản sao lưu (archive)
    try:
        archive_dir = os.path.join(os.path.dirname(primary_path), "archive")
        os.makedirs(archive_dir, exist_ok=True)

        base_name = os.path.basename(primary_path)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(archive_dir, f"{name}_v{timestamp}{ext}")

        shutil.copy2(primary_path, archive_path)
        print(f"[Backup] Saved versioned backup to {archive_path}")
        return archive_path
    except Exception as e:
        print(f"[Backup] Lỗi sao lưu model: {e}")
        return None


def log_training_history(metrics: dict):
    """
    Ghi nhận lại lịch sử huấn luyện chi tiết vào file model_history.json
    """
    import json
    from datetime import datetime

    history_path = os.path.join(SAVED_MODELS_DIR, "model_history.json")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics
    }

    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append(record)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        print(f"[History] Training metrics logged to {history_path}")
    except Exception as e:
        print(f"[History] Lỗi ghi lịch sử training: {e}")



def _time_split_xy(X, y, test_size):
    """
    Split theo thời gian: train = quá khứ, test = tương lai.
    """
    n = len(X)
    if n < 2:
        raise ValueError("Not enough samples to split train/test.")

    n_test = max(1, int(n * test_size))
    n_test = min(n_test, n - 1)
    split_idx = n - n_test

    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


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

    X_train, X_test, y_train, y_test = _time_split_xy(X, y, TEST_SIZE)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    naive_pred = X_test[:, feature_cols.index("pm25")]
    naive_mae = mean_absolute_error(y_test, naive_pred)

    print(f"[Train] Forecast MAE (test): {mae:.2f} µg/m³")
    print(f"[Train] Naive MAE (test):    {naive_mae:.2f} µg/m³")

    save_model_with_backup(model, feature_cols, MODEL_FORECAST)

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

    save_model_with_backup(model, feature_cols, MODEL_ANOMALY)

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

    X_train, X_test, y_train, y_test = _time_split_xy(X, y, TEST_SIZE)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[Train] Classifier accuracy (test): {accuracy:.2%}")

    save_model_with_backup(model, feature_cols, MODEL_CLASSIFIER)

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

    log_training_history(result)

    print("\n" + "=" * 50)
    print("[Train] Done!")
    print(result)
    print("=" * 50)

    return result


if __name__ == "__main__":
    run_training()