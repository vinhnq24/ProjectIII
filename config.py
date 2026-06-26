import os

# =====================================================
# DATABASE
# =====================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "database", "air.db")

# =====================================================
# SERVER
# =====================================================
HOST        = "0.0.0.0"
PORT        = 8000
DEBUG       = True

# =====================================================
# STATION LOCATION (Tọa độ trạm cảm biến thực tế)
# Mẹo: Click chuột phải trên Google Maps tại vị trí trạm để lấy tọa độ.
# =====================================================
STATION_LAT = 21.0285  # Thay bằng Vĩ độ chính xác của bạn
STATION_LNG = 105.8542 # Thay bằng Kinh độ chính xác của bạn

# =====================================================
# MODEL
# =====================================================
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_FORECAST   = os.path.join(SAVED_MODELS_DIR, "forecast_model.pkl")
MODEL_ANOMALY    = os.path.join(SAVED_MODELS_DIR, "anomaly_model.pkl")
MODEL_CLASSIFIER = os.path.join(SAVED_MODELS_DIR, "classifier_model.pkl")
SCALER_PATH      = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")

# =====================================================
# TRAINING
# =====================================================
# Số bản ghi tối thiểu để bắt đầu train
MIN_RECORDS_TO_TRAIN = 500

# Tỉ lệ train/test
TEST_SIZE = 0.2

# Random seed
RANDOM_STATE = 42

# =====================================================
# AIR QUALITY THRESHOLDS (theo WHO 2021)
# =====================================================
THRESHOLDS = {
    "pm25": {"good": 35,  "normal": 75,  "bad": 150},
    "pm10": {"good": 54,  "normal": 154, "bad": 254},
    "temp": {"min_ok": 18, "max_ok": 28},
    "hum":  {"min_ok": 40, "max_ok": 70},
    "mq":   {"good": 800, "normal": 1500, "bad": 2500},
}

# =====================================================
# MQTT
# =====================================================
MQTT_BROKER    = os.environ.get("MQTT_BROKER",    "22f19246dfe745fd9bbc373e63c12f9f.s1.eu.hivemq.cloud")
MQTT_PORT      = int(os.environ.get("MQTT_PORT",  8883))
MQTT_TOPIC     = os.environ.get("MQTT_TOPIC",     "airquality/data")
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "air-quality-ai-server")
MQTT_USERNAME  = os.environ.get("MQTT_USERNAME",  "ngoquangvinh")
MQTT_PASSWORD  = os.environ.get("MQTT_PASSWORD",  "Vinh4953")
MQTT_TLS       = os.environ.get("MQTT_TLS",       "true").lower() in ("1", "true", "yes", "on")

# =====================================================
# LOGGING
# =====================================================
LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")
