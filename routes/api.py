from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_service import (
    process_incoming_data,
    get_dashboard_data,
    export_to_csv,
)
from database.db import (
    get_latest, get_count, get_range,
    get_latest_predictions
)
from ml.predict import run_all_predictions
from ml.train import run_training

router = APIRouter()


# =====================================================
# SCHEMA
# =====================================================
class SensorPayload(BaseModel):
    pm25: float = Field(..., ge=0, le=1000, description="PM2.5 µg/m³")
    pm10: float = Field(..., ge=0, le=1000, description="PM10 µg/m³")
    temp: float = Field(..., ge=-40, le=80,  description="Nhiệt độ °C")
    hum:  float = Field(..., ge=0,  le=100,  description="Độ ẩm %")
    mq:   float = Field(..., ge=0,           description="MQ135 raw value")


# =====================================================
# POST /sensor-data
# ESP32 gửi dữ liệu lên đây
# =====================================================
@router.post("/sensor-data")
async def receive_sensor_data(payload: SensorPayload):
    try:
        result = process_incoming_data(
            pm25 = payload.pm25,
            pm10 = payload.pm10,
            temp = payload.temp,
            hum  = payload.hum,
            mq   = payload.mq,
        )
        return {"status": "ok", **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# GET /latest
# Lấy N bản ghi gần nhất
# =====================================================
@router.get("/latest")
async def get_latest_data(limit: int = 50):
    rows = get_latest(limit)
    return {
        "status": "ok",
        "count":  len(rows),
        "data":   rows,
    }


# =====================================================
# GET /dashboard
# Dữ liệu tổng hợp cho trang web
# =====================================================
@router.get("/dashboard")
async def dashboard(limit: int = 100):
    data = get_dashboard_data(limit)
    return {"status": "ok", **data}


# =====================================================
# GET /stats
# Thống kê nhanh
# =====================================================
@router.get("/stats")
async def stats():
    count = get_count()
    rows  = get_latest(1)
    latest = rows[0] if rows else None

    return {
        "status":         "ok",
        "total_records":  count,
        "latest":         latest,
    }


# =====================================================
# POST /predict
# Dự đoán từ dữ liệu tùy ý (không cần lưu)
# =====================================================
@router.post("/predict")
async def predict(payload: SensorPayload):
    data = payload.model_dump()
    result = run_all_predictions(data)
    return {"status": "ok", "result": result}


# =====================================================
# GET /predictions
# Lịch sử predictions
# =====================================================
@router.get("/predictions")
async def get_predictions(limit: int = 20):
    preds = get_latest_predictions(limit)
    return {
        "status": "ok",
        "count":  len(preds),
        "data":   preds,
    }


# =====================================================
# POST /train
# Trigger training thủ công
# =====================================================
@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    count = get_count()

    background_tasks.add_task(run_training)

    return {
        "status":  "ok",
        "message": "Training started in background",
        "records": count,
    }


# =====================================================
# GET /train/status
# Kiểm tra model đã được train chưa
# =====================================================
@router.get("/train/status")
async def training_status():
    from config import MODEL_FORECAST, MODEL_ANOMALY, MODEL_CLASSIFIER
    import os

    return {
        "status": "ok",
        "models": {
            "forecast":   os.path.exists(MODEL_FORECAST),
            "anomaly":    os.path.exists(MODEL_ANOMALY),
            "classifier": os.path.exists(MODEL_CLASSIFIER),
        },
        "records": get_count(),
    }


# =====================================================
# GET /export
# Xuất toàn bộ data ra CSV
# =====================================================
@router.get("/export")
async def export():
    result = export_to_csv()
    return {"status": "ok", **result}


# =====================================================
# GET /range
# Lấy data trong khoảng thời gian
# =====================================================
@router.get("/range")
async def get_range_data(start: str, end: str):
    """
    Ví dụ: /range?start=2024-01-01 00:00:00&end=2024-01-02 00:00:00
    """
    try:
        rows = get_range(start, end)
        return {
            "status": "ok",
            "count":  len(rows),
            "data":   rows,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))