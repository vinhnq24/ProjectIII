import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import (
    insert_sensor_data, insert_prediction,
    get_latest, get_all, get_count, get_range
)
from ml.predict import run_all_predictions
from config import MIN_RECORDS_TO_TRAIN

logger = logging.getLogger(__name__)


# =====================================================
# NHẬN DỮ LIỆU TỪ ESP32
# =====================================================
def process_incoming_data(pm25: float, pm10: float,
                           temp: float, hum: float,
                           mq: float,
                           gps_fix: int = 0,
                           lat: float = 0.0,
                           lng: float = 0.0) -> dict:
    # 1. Luôn lưu sensor trước — không phụ thuộc AI
    timestamp = insert_sensor_data(pm25, pm10, temp, hum, mq,
                                   gps_fix=gps_fix, lat=lat, lng=lng)
    count = get_count()
    logger.info(
        "Sensor data saved to SQLite: timestamp=%s pm25=%s pm10=%s temp=%s hum=%s mq=%s gps_fix=%s lat=%s lng=%s count=%s",
        timestamp, pm25, pm10, temp, hum, mq, gps_fix, lat, lng, count,
    )

    data = {
        "pm25": pm25, "pm10": pm10,
        "temp": temp, "hum":  hum,
        "mq":   mq,
        "gps_fix": gps_fix, "lat": lat, "lng": lng,
    }

    predictions = None
    try:
        # 2. Chạy AI prediction (lỗi ở đây không được chặn lưu sensor)
        predictions = run_all_predictions(data)

        forecast = predictions["forecast"]
        anomaly  = predictions["anomaly"]
        classify = predictions["classify"]

        insert_prediction(
            pm25_input    = pm25,
            pm25_forecast = forecast.get("pm25_next"),
            air_class     = classify.get("air_class"),
            anomaly       = 1 if anomaly.get("is_anomaly") else 0,
            anomaly_score = anomaly.get("score"),
        )

        if anomaly.get("is_anomaly"):
            logger.warning(
                "[ALERT - ANOMALY] PM2.5=%s PM10=%s Temp=%s Hum=%s MQ135=%s",
                pm25, pm10, temp, hum, mq,
            )

        if count >= MIN_RECORDS_TO_TRAIN and count % 1000 == 0:
            import threading
            from ml.train import run_training

            logger.info(
                "[Auto-Retrain] %s records reached — retraining models in background",
                count,
            )
            threading.Thread(target=run_training, daemon=True).start()

    except Exception as e:
        logger.exception("AI prediction failed after sensor save: %s", e)

    return {
        "timestamp":      timestamp,
        "saved":          True,
        "count":          count,
        "predictions":    predictions,
        "ready_to_train": count >= MIN_RECORDS_TO_TRAIN,
    }


# =====================================================
# LẤY DỮ LIỆU CHO DASHBOARD
# =====================================================
def get_dashboard_data(limit: int = 50) -> dict:
    rows = get_latest(limit)
    count = get_count()

    if not rows:
        return {"rows": [], "count": 0, "latest": None, "predictions": None}

    latest = rows[0]

    predictions = None
    try:
        predictions = run_all_predictions({
            "pm25": latest["pm25"],
            "pm10": latest["pm10"],
            "temp": latest["temp"],
            "hum":  latest["hum"],
            "mq":   latest["mq"],
        })
    except Exception as e:
        logger.exception("Dashboard AI prediction failed; returning sensor data only: %s", e)

    from config import STATION_LAT, STATION_LNG
    return {
        "rows":        rows,
        "count":       count,
        "latest":      latest,
        "predictions": predictions,
        "station_lat": STATION_LAT,
        "station_lng": STATION_LNG,
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
