import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_FORECAST, MODEL_ANOMALY,
    MODEL_CLASSIFIER, THRESHOLDS
)


# =====================================================
# LOAD MODELS
# =====================================================
def load_model(path):
    if not os.path.exists(path):
        return None, None
    # File rỗng hoặc corrupt → trả về None thay vì crash
    if os.path.getsize(path) == 0:
        return None, None
    try:
        with open(path, "rb") as f:
            model, feature_cols = pickle.load(f)
        return model, feature_cols
    except (EOFError, pickle.UnpicklingError):
        return None, None


# =====================================================
# BUILD FEATURE VECTOR
# =====================================================
def build_feature_vector(data: dict, feature_cols: list) -> np.ndarray:
    """
    Tạo vector features từ dict dữ liệu.
    Các cột không có trong data sẽ được điền 0.
    """
    row = []
    for col in feature_cols:
        row.append(data.get(col, 0.0))
    return np.array(row).reshape(1, -1)


# =====================================================
# PREDICT
# =====================================================
def predict_forecast(data: dict) -> dict:
    model, feature_cols = load_model(MODEL_FORECAST)

    if model is None:
        return {"available": False, "pm25_next": None}

    X = build_feature_vector(data, feature_cols)
    pm25_next = float(model.predict(X)[0])

    return {
        "available": True,
        "pm25_next": round(pm25_next, 1),
    }


def predict_anomaly(data: dict) -> dict:
    model, feature_cols = load_model(MODEL_ANOMALY)

    if model is None:
        return {"available": False, "is_anomaly": None, "score": None}

    X = build_feature_vector(data, feature_cols)

    pred  = model.predict(X)[0]           # 1 = normal, -1 = anomaly
    score = float(model.decision_function(X)[0])

    return {
        "available":  True,
        "is_anomaly": bool(pred == -1),
        "score":      round(score, 4),
    }


def predict_class(data: dict) -> dict:
    model, feature_cols = load_model(MODEL_CLASSIFIER)

    if model is None:
        return {"available": False, "air_class": None, "label": None}

    CLASS_MAP = {0: "Good", 1: "Normal", 2: "Bad", 3: "Danger"}

    X = build_feature_vector(data, feature_cols)

    pred   = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]

    return {
        "available":   True,
        "air_class":   CLASS_MAP.get(pred, "Unknown"),
        "label":       pred,
        "confidence":  round(float(np.max(probas)), 3),
    }


# =====================================================
# FULL PREDICTION (gọi 1 lần, trả về tất cả)
# =====================================================
def run_all_predictions(data: dict) -> dict:
    """
    data = {
        "pm25": float, "pm10": float,
        "temp": float, "hum":  float,
        "mq":   float
    }
    """
    forecast = predict_forecast(data)
    anomaly  = predict_anomaly(data)
    classify = predict_class(data)

    # Rule-based alert level (luôn có, không cần model)
    alert_level = _rule_based_alert(data)

    return {
        "input":       data,
        "forecast":    forecast,
        "anomaly":     anomaly,
        "classify":    classify,
        "alert_level": alert_level,
    }


# =====================================================
# RULE-BASED FALLBACK (khi chưa có model)
# =====================================================
def _rule_based_alert(data: dict) -> dict:
    t = THRESHOLDS

    def level_pm25(v):
        if v < t["pm25"]["good"]:   return 0
        if v < t["pm25"]["normal"]: return 1
        if v < t["pm25"]["bad"]:    return 2
        return 3

    def level_pm10(v):
        if v < t["pm10"]["good"]:   return 0
        if v < t["pm10"]["normal"]: return 1
        if v < t["pm10"]["bad"]:    return 2
        return 3

    def level_mq(v):
        if v < t["mq"]["good"]:     return 0
        if v < t["mq"]["normal"]:   return 1
        if v < t["mq"]["bad"]:      return 2
        return 3

    def level_temp(v):
        if t["temp"]["min_ok"] <= v <= t["temp"]["max_ok"]: return 0
        return 2

    def level_hum(v):
        if t["hum"]["min_ok"] <= v <= t["hum"]["max_ok"]: return 0
        return 1

    levels = {
        "pm25": level_pm25(data.get("pm25", 0)),
        "pm10": level_pm10(data.get("pm10", 0)),
        "mq":   level_mq(data.get("mq", 0)),
        "temp": level_temp(data.get("temp", 25)),
        "hum":  level_hum(data.get("hum", 50)),
    }

    LABEL = {0: "Good", 1: "Normal", 2: "Bad", 3: "Danger"}
    max_level = max(levels.values())

    return {
        "level":          max_level,
        "label":          LABEL[max_level],
        "detail":         levels,
        "triggered_by":   [k for k, v in levels.items() if v == max_level],
    }