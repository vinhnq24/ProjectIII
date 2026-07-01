import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "database", "air.db"))

# =====================================================
# Server
# =====================================================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
DEBUG = _env_bool("DEBUG", True)

# =====================================================
# Station location
# Update these values to the real sensor location.
# =====================================================
STATION_LAT = float(os.environ.get("STATION_LAT", 21.0285))
STATION_LNG = float(os.environ.get("STATION_LNG", 105.8542))

# =====================================================
# Models
# =====================================================
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_FORECAST = os.path.join(SAVED_MODELS_DIR, "forecast_model.pkl")
MODEL_ANOMALY = os.path.join(SAVED_MODELS_DIR, "anomaly_model.pkl")
MODEL_CLASSIFIER = os.path.join(SAVED_MODELS_DIR, "classifier_model.pkl")
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")

# =====================================================
# Training
# =====================================================
MIN_RECORDS_TO_TRAIN = int(os.environ.get("MIN_RECORDS_TO_TRAIN", 500))
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))

# =====================================================
# Air quality thresholds
# =====================================================
THRESHOLDS = {
    "pm25": {"good": 35, "normal": 75, "bad": 150},
    "pm10": {"good": 54, "normal": 154, "bad": 254},
    "temp": {"min_ok": 18, "max_ok": 28},
    "hum": {"min_ok": 40, "max_ok": 70},
    "mq": {"good": 800, "normal": 1500, "bad": 2500},
}

# =====================================================
# MQTT
# Keep credentials in environment variables or .env tooling.
# Do not commit real usernames/passwords to source control.
# =====================================================
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "airquality/data")
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "air-quality-ai-server")
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")
MQTT_TLS = _env_bool("MQTT_TLS", False)

# =====================================================
# Logging
# =====================================================
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(BASE_DIR, "logs", "app.log"))
