import pickle
# pyrefly: ignore [missing-import]
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime

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
# ENRICH DATA (Fix Training-Serving Skew)
# =====================================================
def get_enriched_data(current_data: dict) -> dict:
    """
    Lấy dữ liệu lịch sử từ SQLite, kết hợp với dữ liệu hiện tại,
    và chạy hàm add_features để sinh các đặc trưng động (rolling, lag, time).
    Tránh hiện tượng Training-Serving Skew khi đưa vào mô hình AI thực tế.
    """
    try:
        from database.db import get_latest
        # Lấy tối đa 4 dòng dữ liệu lịch sử gần nhất
        history = get_latest(limit=4)
        # get_latest trả về danh sách có id giảm dần (mới nhất trước)
        # Cần đảo ngược lại để có thứ tự thời gian tăng dần
        history = list(reversed(history))
    except Exception as e:
        print(f"[Predict] Lỗi khi lấy dữ liệu lịch sử: {e}")
        history = []

    # Tạo bản sao dữ liệu hiện tại để tránh thay đổi dictionary gốc
    curr = current_data.copy()
    if "timestamp" not in curr:
        curr["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Kiểm tra xem bản ghi hiện tại đã có trong database chưa
    # (Trường hợp ESP32 gửi lên được lưu trước khi chạy dự báo)
    is_already_in_db = False
    if history:
        last_rec = history[-1]
        # So sánh các chỉ số cơ bản để xác định trùng lặp
        if (abs(last_rec.get("pm25", -999) - curr.get("pm25", 0.0)) < 1e-4 and
            abs(last_rec.get("pm10", -999) - curr.get("pm10", 0.0)) < 1e-4 and
            abs(last_rec.get("temp", -999) - curr.get("temp", 0.0)) < 1e-4):
            is_already_in_db = True

    if not is_already_in_db:
        # Nếu chưa có trong DB (ví dụ: test từ API /predict), thêm vào cuối history
        history.append(curr)

    # Chuyển đổi sang DataFrame để xử lý đồng nhất với lúc train
    from ml.preprocess import add_features

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Tính toán đặc trưng trễ (lag), trung bình trượt (rolling), giờ tuần hoàn
    df = add_features(df)

    # Trả về bản ghi cuối cùng dưới dạng dictionary đầy đủ đặc trưng
    enriched_row = df.iloc[-1].to_dict()

    # Chuyển đổi timestamp ngược lại dạng chuỗi nếu cần
    if isinstance(enriched_row.get("timestamp"), pd.Timestamp):
        enriched_row["timestamp"] = enriched_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return enriched_row


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
    # Làm giàu đặc trưng trước khi chạy dự báo để tránh lệch đặc trưng (Training-Serving Skew)
    enriched_data = get_enriched_data(data)

    forecast = predict_forecast(enriched_data)
    anomaly  = predict_anomaly(enriched_data)
    classify = predict_class(enriched_data)

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
