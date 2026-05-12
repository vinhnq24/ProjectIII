import sqlite3
import os
from datetime import datetime
from config import DB_PATH


# =====================================================
# INIT DATABASE
# =====================================================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            pm25        REAL    NOT NULL,
            pm10        REAL    NOT NULL,
            temp        REAL    NOT NULL,
            hum         REAL    NOT NULL,
            mq          REAL    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            pm25_input      REAL,
            pm25_forecast   REAL,
            air_class       TEXT,
            anomaly         INTEGER,
            anomaly_score   REAL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_sensor_timestamp ON sensor_data(timestamp)
    """)

    conn.commit()
    conn.close()

    print(f"[DB] Initialized at {DB_PATH}")


# =====================================================
# INSERT
# =====================================================
def insert_sensor_data(pm25, pm10, temp, hum, mq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO sensor_data
        (timestamp, pm25, pm10, temp, hum, mq)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, pm25, pm10, temp, hum, mq))

    conn.commit()
    conn.close()

    return timestamp


# =====================================================
# SELECT
# =====================================================
def get_latest(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sensor_data
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows


def get_all():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sensor_data
        ORDER BY timestamp ASC
    """)

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows


def get_count():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sensor_data")
    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_range(start: str, end: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sensor_data
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (start, end))

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows


# =====================================================
# INSERT PREDICTION
# =====================================================
def insert_prediction(pm25_input, pm25_forecast,
                      air_class, anomaly, anomaly_score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions
        (timestamp, pm25_input, pm25_forecast,
         air_class, anomaly, anomaly_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, pm25_input, pm25_forecast,
          air_class, anomaly, anomaly_score))

    conn.commit()
    conn.close()


def get_latest_predictions(limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows