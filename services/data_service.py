import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import (
    insert_sensor_data, insert_prediction,
    get_latest, get_all, get_count, get_range
)
from ml.predict import run_all_predictions
from config import MIN_RECORDS_TO_TRAIN


# =====================================================
# NHẬN DỮ LIỆU TỪ ESP32
# =====================================================
def process_incoming_data(pm25: float, pm10: float,
                           temp: float, hum: float,
                           mq: float) -> dict:
    # 1. Lưu vào database
    timestamp = insert_sensor_data(pm25, pm10, temp, hum, mq)

    # 2. Chạy AI prediction
    data = {
        "pm25": pm25, "pm10": pm10,
        "temp": temp, "hum":  hum,
        "mq":   mq,
    }

    predictions = run_all_predictions(data)

    # 3. Lưu prediction vào database
    forecast  = predictions["forecast"]
    anomaly   = predictions["anomaly"]
    classify  = predictions["classify"]

    insert_prediction(
        pm25_input    = pm25,
        pm25_forecast = forecast.get("pm25_next"),
        air_class     = classify.get("air_class"),
        anomaly       = 1 if anomaly.get("is_anomaly") else 0,
        anomaly_score = anomaly.get("score"),
    )

    # 4. Đếm tổng records
    count = get_count()

    return {
        "timestamp":   timestamp,
        "saved":       True,
        "count":       count,
        "predictions": predictions,
        "ready_to_train": count >= MIN_RECORDS_TO_TRAIN,
    }


# =====================================================
# LẤY DỮ LIỆU CHO DASHBOARD
# =====================================================
def get_dashboard_data(limit: int = 50) -> dict:
    rows = get_latest(limit)
    count = get_count()

    if not rows:
        return {"rows": [], "count": 0, "summary": None}

    latest = rows[0]

    predictions = run_all_predictions({
        "pm25": latest["pm25"],
        "pm10": latest["pm10"],
        "temp": latest["temp"],
        "hum":  latest["hum"],
        "mq":   latest["mq"],
    })

    return {
        "rows":        rows,
        "count":       count,
        "latest":      latest,
        "predictions": predictions,
    }


# =====================================================
# EXPORT DATA
# =====================================================
def export_to_csv(filepath: str = "data/exports/export.csv"):
    import csv

    rows = get_all()

    if not rows:
        return {"success": False, "message": "No data"}

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return {
        "success":  True,
        "filepath": filepath,
        "rows":     len(rows),
    }