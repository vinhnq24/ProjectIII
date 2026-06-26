import json
import logging
import re
import ssl

try:
    import paho.mqtt.client as mqtt
except ModuleNotFoundError:
    mqtt = None

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_CLIENT_ID,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TLS,
)
from services.data_service import process_incoming_data

logger = logging.getLogger(__name__)


# =====================================================
# CALLBACKS
# =====================================================

def on_connect(client, userdata, flags, reason_code, properties):
    """Gọi khi kết nối tới broker thành công (hoặc thất bại)."""
    if reason_code == 0:
        logger.info(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        # Subscribe ngay sau khi connect — đây là cách đúng với paho v2
        client.subscribe(MQTT_TOPIC)
        logger.info(f"MQTT subscribed to topic: {MQTT_TOPIC}")
    else:
        logger.error(f"MQTT connection failed, reason code: {reason_code}")


def on_disconnect(client, userdata, flags, reason_code, properties):
    """Gọi khi mất kết nối với broker."""
    if reason_code != 0:
        logger.warning(f"MQTT disconnected unexpectedly (rc={reason_code}). Will auto-reconnect...")
    else:
        logger.info("MQTT disconnected cleanly.")


def parse_mqtt_payload(raw: bytes) -> dict:
    """Parse JSON từ ESP32 — hỗ trợ cả key có/không dấu ngoặc kép."""
    raw_str = raw.decode("utf-8", errors="replace").strip()

    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        pass

    # ESP32 cũ hoặc payload thủ công: {pm25:35.5, pm10:48.2, ...}
    normalized = re.sub(
        r'([{,]\s*)([a-zA-Z0-9_]+)\s*:',
        r'\1"\2":',
        raw_str,
    )
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    cleaned = raw_str.replace("\\", "").replace('"', "").replace("{", "").replace("}", "")
    pairs = re.findall(
        r"\b(pm25|pm10|temp|hum|mq)\s*:\s*(-?\d+(?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if pairs:
        return {key.lower(): float(value) for key, value in pairs}

    return json.loads(normalized)


def on_message(client, userdata, msg):
    """Gọi khi nhận được message từ broker."""
    try:
        payload = parse_mqtt_payload(msg.payload)
        logger.info("MQTT message received on '%s': %s", msg.topic, payload)

        process_incoming_data(
            pm25=float(payload["pm25"]),
            pm10=float(payload["pm10"]),
            temp=float(payload["temp"]),
            hum=float(payload["hum"]),
            mq=float(payload["mq"]),
            gps_fix=int(payload.get("gps_fix", 0)),
            lat=float(payload.get("lat", 0.0)),
            lng=float(payload.get("lng", 0.0)),
        )

        logger.info("MQTT data processed and saved successfully.")

    except KeyError as e:
        logger.error("MQTT payload missing field: %s | raw: %s", e, msg.payload)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error("MQTT payload is not valid JSON: %s | raw: %s", e, msg.payload)
    except Exception as e:
        logger.exception("MQTT on_message unhandled error: %s", e)


# =====================================================
# CLIENT SETUP
# =====================================================

client = None

if mqtt is not None:
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=MQTT_CLIENT_ID,
    )

    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    if MQTT_TLS:
        client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.tls_insecure_set(False)

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message


# =====================================================
# START / STOP  (gọi từ app.py lifespan)
# =====================================================

def start():
    """Kết nối tới MQTT broker và bắt đầu event loop ngầm."""
    if client is None:
        logger.error("paho-mqtt is not installed. Run: pip install -r requirements.txt")
        return

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        logger.info("MQTT loop started.")
    except OSError as e:
        # Broker chưa chạy — log cảnh báo, KHÔNG crash app
        logger.warning(f"MQTT broker không khả dụng ({MQTT_BROKER}:{MQTT_PORT}): {e}. "
                       "Server vẫn chạy bình thường, MQTT sẽ retry khi reconnect.")
    except Exception as e:
        logger.exception(f"MQTT start error: {e}")


def stop():
    """Dừng event loop và ngắt kết nối broker sạch sẽ."""
    if client is None:
        return

    try:
        client.loop_stop()
        client.disconnect()
        logger.info("MQTT disconnected cleanly on shutdown.")
    except Exception as e:
        logger.warning(f"MQTT stop error (ignored): {e}")
