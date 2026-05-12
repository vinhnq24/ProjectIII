from datetime import datetime
from config import THRESHOLDS


# =====================================================
# AIR QUALITY LABEL
# =====================================================
def get_air_label(pm25: float) -> str:
    t = THRESHOLDS["pm25"]
    if pm25 < t["good"]:   return "Good"
    if pm25 < t["normal"]: return "Normal"
    if pm25 < t["bad"]:    return "Bad"
    return "Danger"


def get_alert_level(pm25: float) -> int:
    t = THRESHOLDS["pm25"]
    if pm25 < t["good"]:   return 0
    if pm25 < t["normal"]: return 1
    if pm25 < t["bad"]:    return 2
    return 3


# =====================================================
# TIMESTAMP
# =====================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# =====================================================
# VALIDATE SENSOR DATA
# =====================================================
def validate_sensor(data: dict) -> tuple[bool, str]:
    required = ["pm25", "pm10", "temp", "hum", "mq"]

    for key in required:
        if key not in data:
            return False, f"Missing field: {key}"

    if not (0 <= data["pm25"] <= 1000):
        return False, "pm25 out of range"

    if not (0 <= data["pm10"] <= 1000):
        return False, "pm10 out of range"

    if not (-40 <= data["temp"] <= 80):
        return False, "temp out of range"

    if not (0 <= data["hum"] <= 100):
        return False, "hum out of range"

    return True, "ok"


# =====================================================
# FORMAT RESPONSE
# =====================================================
def format_rows_for_chart(rows: list) -> dict:
    """
    Chuyển list rows thành format cho Chart.js:
    { labels: [...timestamps], datasets: {pm25: [...], ...} }
    """
    if not rows:
        return {"labels": [], "datasets": {}}

    # rows đã sắp xếp theo thời gian mới → cũ, đảo lại
    rows_asc = list(reversed(rows))

    labels = [r["timestamp"] for r in rows_asc]

    datasets = {
        "pm25": [r["pm25"]  for r in rows_asc],
        "pm10": [r["pm10"]  for r in rows_asc],
        "temp": [r["temp"]  for r in rows_asc],
        "hum":  [r["hum"]   for r in rows_asc],
        "mq":   [r["mq"]    for r in rows_asc],
    }

    return {"labels": labels, "datasets": datasets}