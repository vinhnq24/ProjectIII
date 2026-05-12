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
# LOGGING
# =====================================================
LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")