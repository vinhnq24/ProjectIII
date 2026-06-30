# 🌬️ Air Quality AI — Hệ thống Giám sát Chất lượng Không khí Thông minh

> **Project 3 — Đại học Bách khoa Hà Nội (HUST)**  
> Hệ thống IoT thu thập dữ liệu cảm biến môi trường thời gian thực, truyền qua MQTT, lưu trữ trong SQLite, dự báo và phân loại chất lượng không khí bằng AI/ML, hiển thị trên Dashboard web chuyên nghiệp.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Phần cứng ESP32](#-phần-cứng-esp32)
- [API Endpoints](#-api-endpoints)
- [AI/ML Pipeline](#-aiml-pipeline)
- [Dashboard](#-dashboard)
- [Testing](#-testing)

---

## 🎯 Tổng quan

Hệ thống **Air Quality AI** giám sát chất lượng không khí end-to-end:

1. **ESP32** đọc dữ liệu từ 5 cảm biến (PM2.5, PM10, nhiệt độ, độ ẩm, khí gas) + GPS
2. **MQTT** truyền dữ liệu real-time mỗi 5 giây đến server
3. **FastAPI** nhận, xử lý, lưu vào **SQLite**
4. **3 mô hình AI** dự báo, phân loại, phát hiện bất thường
5. **Web Dashboard** hiển thị real-time với bản đồ GPS, biểu đồ, cảnh báo

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────┐     MQTT (5s)     ┌──────────────┐     REST API     ┌──────────────┐
│   ESP32      │ ─────────────────▶│  Mosquitto   │◀───────────────▶│   FastAPI    │
│  (5 sensors  │                   │   Broker     │                  │   Server     │
│   + GPS)     │                   └──────────────┘                  └──────┬───────┘
│   + OLED     │                                                           │
│   + RGB LED  │                                                    ┌──────┴───────┐
│   + Buzzer   │                                                    │              │
└──────────────┘                                              ┌─────┴─────┐  ┌─────┴─────┐
                                                              │  SQLite   │  │ ML Models │
                                                              │  Database │  │ (3 models)│
                                                              └───────────┘  └───────────┘
                                                                     │
                                                              ┌──────┴───────┐
                                                              │     Web      │
                                                              │  Dashboard   │
                                                              │ (Chart.js +  │
                                                              │  Leaflet)    │
                                                              └──────────────┘
```

---

## 🛠️ Công nghệ sử dụng

### Phần cứng
| Thành phần | Mô tả |
|-----------|-------|
| ESP32 DevKit V1 | Vi điều khiển chính |
| PMS7003 | Cảm biến bụi mịn PM2.5, PM10 (UART) |
| DHT22 | Cảm biến nhiệt độ và độ ẩm |
| MQ135 | Cảm biến khí gas (CO2, NH3, benzene) |
| NEO-6M GPS | Module định vị vệ tinh |
| SSD1306 OLED | Màn hình hiển thị 128x64 (I2C) |
| RGB LED | Chỉ thị trạng thái (Common Cathode) |
| Buzzer | Cảnh báo âm thanh |

### Phần mềm
| Công nghệ | Phiên bản | Vai trò |
|-----------|-----------|---------|
| Python | 3.10+ | Ngôn ngữ chính backend |
| FastAPI | 0.111.0 | Web framework (async) |
| Uvicorn | 0.29.0 | ASGI server |
| SQLite | Built-in | Cơ sở dữ liệu |
| scikit-learn | 1.7.2 | Machine Learning |
| pandas | 2.3.3 | Xử lý dữ liệu |
| numpy | 2.3.5 | Tính toán số |
| paho-mqtt | 2.1.0 | MQTT client (Python) |
| Jinja2 | 3.1.4 | Template engine |
| Chart.js | CDN | Biểu đồ |
| Leaflet.js | 1.9.4 | Bản đồ GIS |
| Tailwind CSS | CDN | Giao diện |
| Arduino IDE | - | Firmware ESP32 |
| PubSubClient | - | MQTT client (ESP32) |
| Mosquitto | - | MQTT Broker |

---

## 📁 Cấu trúc thư mục

```
air-quality-ai/
├── app.py                  # Entry point — FastAPI server
├── config.py               # Cấu hình tập trung (env variables)
├── mqtt_client.py          # MQTT subscriber (nhận data từ ESP32)
├── mqtt_test.py            # Test kết nối MQTT nhanh
├── mosquitto.conf          # Cấu hình Mosquitto broker
├── requirements.txt        # Python dependencies
│
├── database/
│   ├── db.py               # SQLite CRUD operations
│   └── air.db              # Database file (~7000 records)
│
├── routes/
│   └── api.py              # REST API endpoints (12 routes)
│
├── services/
│   └── data_service.py     # Business logic layer
│
├── ml/
│   ├── preprocess.py       # Tiền xử lý & Feature Engineering
│   ├── train.py            # Training pipeline (3 models)
│   ├── predict.py          # Inference + Rule-based fallback
│   └── evaluate.py         # Đánh giá model (MAE, RMSE, Accuracy)
│
├── models/
│   └── ai_model.py         # Data model definitions
│
├── saved_models/
│   ├── forecast_model.pkl  # Random Forest Regressor
│   ├── classifier_model.pkl # Random Forest Classifier
│   ├── anomaly_model.pkl   # Isolation Forest
│   ├── scaler.pkl          # StandardScaler
│   ├── model_history.json  # Lịch sử training
│   └── archive/            # Backup các phiên bản model cũ
│
├── utils/
│   └── helpers.py          # Hàm tiện ích (validate, format, label)
│
├── templates/
│   └── index.html          # Dashboard web (1459 dòng)
│
├── tests/
│   ├── test_api.py         # Test API endpoints
│   └── test_model.py       # Test ML pipeline & helpers
│
├── data/
│   ├── raw/                # Dữ liệu thô
│   ├── processed/          # Dữ liệu đã xử lý
│   └── exports/            # File CSV xuất ra
│
├── logs/
│   └── app.log             # Log hệ thống
│
└── esp32/
    └── esp32_mqtt/
        ├── esp32_mqtt.ino  # Firmware ESP32 (412 dòng)
        └── secrets.example.h # Template cấu hình WiFi/MQTT
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- Mosquitto MQTT Broker
- Arduino IDE (cho ESP32)

### 1. Clone & cài đặt

```bash
git clone <repository-url>
cd air-quality-ai

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Khởi động MQTT Broker

```bash
# Windows (sau khi cài Mosquitto)
mosquitto -c mosquitto.conf

# Hoặc chạy với config mặc định
mosquitto -p 1883
```

### 3. Chạy server

```bash
python app.py
```

Server sẽ chạy tại:
- **Dashboard**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 4. Nạp firmware ESP32

1. Mở `esp32/esp32_mqtt/esp32_mqtt.ino` bằng Arduino IDE
2. Tạo file `secrets.h` từ `secrets.example.h`, điền thông tin WiFi và MQTT
3. Cài đặt thư viện: `PubSubClient`, `Adafruit_SSD1306`, `DHT`, `TinyGPS++`
4. Chọn board `ESP32 Dev Module` và nạp code

### 5. Biến môi trường (tùy chọn)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DB_PATH` | `database/air.db` | Đường dẫn database |
| `MQTT_BROKER` | `localhost` | MQTT broker address |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `airquality/data` | MQTT topic |
| `MQTT_USERNAME` | *(trống)* | MQTT username |
| `MQTT_PASSWORD` | *(trống)* | MQTT password |
| `MQTT_TLS` | `false` | Bật mã hóa TLS |
| `MIN_RECORDS_TO_TRAIN` | `500` | Số bản ghi tối thiểu để train |

---

## 🔌 Phần cứng ESP32

### Sơ đồ kết nối

| Cảm biến | Pin ESP32 | Giao tiếp |
|----------|-----------|-----------|
| PMS7003 RX | GPIO 16 | UART Serial2 |
| PMS7003 TX | GPIO 17 | UART Serial2 |
| DHT22 Data | GPIO 4 | Digital |
| MQ135 Analog | GPIO 34 | ADC |
| GPS RX | GPIO 32 | UART Serial1 |
| GPS TX | GPIO 33 | UART Serial1 |
| OLED SDA | GPIO 21 | I2C |
| OLED SCL | GPIO 22 | I2C |
| LED R | GPIO 25 | Digital |
| LED G | GPIO 26 | Digital |
| LED B | GPIO 14 | Digital |
| Buzzer | GPIO 27 | Digital |

### Tính năng firmware
- **Lọc EMA** (Exponential Moving Average) giảm nhiễu sensor
- **GPS HDOP filtering** — chỉ chấp nhận tọa độ khi HDOP < 5.0 và ≥ 4 vệ tinh
- **Checksum PMS7003** — xác minh tính toàn vẹn frame dữ liệu
- **OLED display** — hiển thị PM2.5, PM10, nhiệt độ, độ ẩm, MQ135, trạng thái GPS
- **RGB LED** — xanh (WiFi OK), xanh dương (MQTT OK), đỏ (lỗi)
- **Buzzer** — beep khi publish thành công/thất bại
- **Gửi dữ liệu JSON** qua MQTT mỗi 5 giây

### Format payload MQTT

```json
{
  "ts": 12345,
  "pm25": 25.5,
  "pm10": 40.2,
  "temp": 28.3,
  "hum": 65.0,
  "mq": 450.0,
  "gps_fix": 1,
  "lat": 21.028500,
  "lng": 105.854200
}
```

---

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Trang Dashboard HTML |
| `POST` | `/sensor-data` | Nhận dữ liệu cảm biến từ ESP32 |
| `GET` | `/latest?limit=50` | Lấy N bản ghi mới nhất |
| `GET` | `/dashboard?limit=100` | Dữ liệu tổng hợp cho Dashboard |
| `GET` | `/stats` | Thống kê nhanh (tổng records, bản ghi mới nhất) |
| `POST` | `/predict` | Dự đoán AI từ input bất kỳ |
| `GET` | `/predictions?limit=20` | Lịch sử kết quả dự đoán |
| `POST` | `/train` | Kích hoạt training model (background) |
| `GET` | `/train/status` | Trạng thái các model đã train |
| `GET` | `/metrics` | Đánh giá chất lượng model |
| `GET` | `/export` | Xuất toàn bộ dữ liệu ra CSV |
| `GET` | `/range?start=...&end=...` | Query dữ liệu theo khoảng thời gian |

### Ví dụ sử dụng

```bash
# Gửi dữ liệu cảm biến
curl -X POST http://localhost:8000/sensor-data \
  -H "Content-Type: application/json" \
  -d '{"pm25": 25.5, "pm10": 40.2, "temp": 28.0, "hum": 65.0, "mq": 450.0}'

# Dự đoán AI
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm25": 80.0, "pm10": 120.0, "temp": 32.0, "hum": 85.0, "mq": 1200.0}'

# Kích hoạt train model
curl -X POST http://localhost:8000/train

# Xem metrics
curl http://localhost:8000/metrics
```

---

## 🤖 AI/ML Pipeline

### 3 Mô hình Machine Learning

| Model | Thuật toán | Input | Output | Metric |
|-------|-----------|-------|--------|--------|
| **Forecast** | Random Forest Regressor | 13 features | PM2.5 tiếp theo (µg/m³) | MAE |
| **Classifier** | Random Forest Classifier | 13 features | Good / Normal / Bad / Danger | Accuracy |
| **Anomaly** | Isolation Forest | 13 features | Normal / Anomaly + Score | Contamination |

### Feature Engineering (13 features)

| Feature | Nguồn | Mô tả |
|---------|-------|-------|
| `pm25`, `pm10`, `temp`, `hum`, `mq` | Cảm biến | 5 giá trị gốc |
| `gps_fix`, `lat`, `lng` | GPS | Thông tin vị trí |
| `pm25_roll5`, `pm10_roll5`, `mq_roll5` | Rolling | Trung bình trượt 5 bản ghi |
| `pm25_lag1`, `pm10_lag1` | Lag | Giá trị trước đó |
| `hour_sin`, `hour_cos` | Time | Giờ mã hóa tuần hoàn |

### Kết quả training (lần cuối — 23/06/2026)

| Metric | Giá trị |
|--------|---------|
| Records sử dụng | 6,995 |
| Forecast MAE | **0.81 µg/m³** |
| Classifier Accuracy | **100%** |
| Anomalies detected | 350 / 6,995 (5%) |

### Kỹ thuật đáng chú ý
- **Time-based split**: Train/test chia theo thời gian (80/20), không random — đúng chuẩn time series
- **Training-Serving Skew prevention**: Hàm `get_enriched_data()` tính features động giống lúc train
- **Auto-retrain**: Tự động huấn luyện lại mỗi 1,000 bản ghi mới
- **Model versioning**: Backup tự động với timestamp vào thư mục `archive/`
- **Rule-based fallback**: Hệ thống cảnh báo dựa trên ngưỡng vẫn hoạt động ngay cả khi chưa train model

### Train thủ công

```bash
# Từ command line
python -m ml.train

# Hoặc qua API
curl -X POST http://localhost:8000/train

# Đánh giá model
python -m ml.evaluate
```

---

## 🖥️ Dashboard

### Tính năng giao diện
- **AQI Gauge** — chỉ số chất lượng không khí dạng vòng tròn với mã màu
- **Bản đồ GPS** (Leaflet) — định vị trạm cảm biến bằng GPS thực, reverse geocoding hiển thị địa chỉ
- **Biểu đồ xu hướng** (Chart.js) — PM2.5/PM10 theo thời gian (12/30/60 bản tin)
- **Thẻ thông số** — PM2.5, PM10, nhiệt độ, độ ẩm, MQ135 cập nhật real-time
- **Kết quả AI** — dự báo PM2.5, phân loại chất lượng, phát hiện bất thường
- **Nhật ký hệ thống** — log MQTT, GPS, cảnh báo dạng terminal
- **Nhật ký cảnh báo** — ghi nhận các sự kiện vượt ngưỡng nguy hại
- **Giả lập khẩn cấp** — demo kịch bản ô nhiễm nguy hại cho bài báo cáo
- **Xuất báo cáo** — export TXT + CSV
- **Nút Train AI** — kích hoạt training trực tiếp từ giao diện
- **Polling tự động** mỗi 4 giây
- **Dark mode** — thiết kế tối hiện đại, responsive

---

## 🧪 Testing

### Chạy toàn bộ test

```bash
pytest tests/ -v
```

### Chạy riêng từng module

```bash
# Test API endpoints
pytest tests/test_api.py -v

# Test ML pipeline & helpers
pytest tests/test_model.py -v
```

### Phạm vi test

**test_api.py** (12 test classes):
- Dashboard page trả về HTML
- POST `/sensor-data` — valid, missing fields, out of range, danger level
- GET `/latest` — default, custom limit, data structure
- GET `/dashboard` — data + predictions
- GET `/stats` — total records
- POST `/predict` — valid, invalid, danger input, alert structure
- GET `/predictions` — history
- POST `/train` — trigger + status
- GET `/metrics` — model evaluation
- GET `/export` — CSV export
- GET `/range` — time range query

**test_model.py** (30+ test cases):
- Preprocess: rows_to_dataframe, clean negatives/duplicates/outliers
- Feature engineering: rolling, lag, cyclic time
- Forecast target, classifier labels, thresholds
- Scaler fit/save/load
- Predict: load model, build features, rule-based alerts
- run_all_predictions structure
- Helpers: air label, alert level, validate, format
- Config: thresholds, model paths, MQTT defaults

---

## 📝 Ghi chú

- Database mặc định: `database/air.db` (~1.2 MB, ~7000 records thực)
- Logs: `logs/app.log` (ghi lại toàn bộ hoạt động server)
- Swagger Docs tự động: http://localhost:8000/docs
- Mosquitto config cho phép anonymous trên local network

---

## 👤 Tác giả

- **Sinh viên**: Ngô Quang Vinh
- **Môn học**: Project 3
- **Năm**: 2026

---

## 📄 License

Project phục vụ mục đích học tập tại HUST.
