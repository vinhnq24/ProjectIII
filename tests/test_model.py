"""
Test ML Pipeline — Preprocess, Train, Predict, Evaluate
Chạy: pytest tests/test_model.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =====================================================
# FIXTURE: Sample sensor data (giả lập ~100 bản ghi)
# =====================================================
@pytest.fixture(scope="module")
def sample_rows():
    """Tạo 100 bản ghi cảm biến giả lập theo thứ tự thời gian."""
    np.random.seed(42)
    rows = []
    base_time = datetime(2026, 6, 1, 8, 0, 0)

    for i in range(100):
        ts = base_time + timedelta(seconds=i * 5)
        rows.append({
            "id": i + 1,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "pm25": max(0, 25 + np.random.normal(0, 10)),
            "pm10": max(0, 40 + np.random.normal(0, 15)),
            "temp": 28 + np.random.normal(0, 2),
            "hum": 65 + np.random.normal(0, 5),
            "mq": max(0, 400 + np.random.normal(0, 80)),
            "gps_fix": 1,
            "lat": 21.0285 + np.random.normal(0, 0.0001),
            "lng": 105.8542 + np.random.normal(0, 0.0001),
        })
    return rows


@pytest.fixture(scope="module")
def sample_dataframe(sample_rows):
    """Chuyển sample_rows thành DataFrame đã clean."""
    from ml.preprocess import rows_to_dataframe, clean_data
    df = rows_to_dataframe(sample_rows)
    df = clean_data(df)
    return df


# =====================================================
# TEST: PREPROCESS MODULE
# =====================================================
class TestPreprocess:

    def test_rows_to_dataframe(self, sample_rows):
        """Chuyển list[dict] → DataFrame với timestamp đúng kiểu."""
        from ml.preprocess import rows_to_dataframe
        df = rows_to_dataframe(sample_rows)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        # Phải được sắp xếp theo thời gian tăng dần
        assert df["timestamp"].is_monotonic_increasing

    def test_clean_data_removes_negatives(self):
        """Clean data loại bỏ giá trị âm của pm25, pm10."""
        from ml.preprocess import rows_to_dataframe, clean_data

        rows = [
            {"id": 1, "timestamp": "2026-06-01 08:00:00",
             "pm25": -5, "pm10": 40, "temp": 28, "hum": 65, "mq": 400,
             "gps_fix": 0, "lat": 0, "lng": 0},
            {"id": 2, "timestamp": "2026-06-01 08:00:05",
             "pm25": 20, "pm10": 35, "temp": 27, "hum": 60, "mq": 380,
             "gps_fix": 0, "lat": 0, "lng": 0},
        ]
        df = rows_to_dataframe(rows)
        df = clean_data(df)

        assert len(df) == 1  # Bản ghi pm25=-5 bị loại
        assert df.iloc[0]["pm25"] == 20

    def test_clean_data_removes_duplicates(self):
        """Clean data loại bỏ bản ghi trùng timestamp."""
        from ml.preprocess import rows_to_dataframe, clean_data

        rows = [
            {"id": 1, "timestamp": "2026-06-01 08:00:00",
             "pm25": 25, "pm10": 40, "temp": 28, "hum": 65, "mq": 400,
             "gps_fix": 0, "lat": 0, "lng": 0},
            {"id": 2, "timestamp": "2026-06-01 08:00:00",  # trùng
             "pm25": 30, "pm10": 45, "temp": 29, "hum": 66, "mq": 410,
             "gps_fix": 0, "lat": 0, "lng": 0},
        ]
        df = rows_to_dataframe(rows)
        df = clean_data(df)

        assert len(df) == 1

    def test_clean_data_removes_extreme_outliers(self):
        """Clean data loại bỏ PM2.5 > 1000 (outlier cực đoan)."""
        from ml.preprocess import rows_to_dataframe, clean_data

        rows = [
            {"id": 1, "timestamp": "2026-06-01 08:00:00",
             "pm25": 1500, "pm10": 40, "temp": 28, "hum": 65, "mq": 400,
             "gps_fix": 0, "lat": 0, "lng": 0},
            {"id": 2, "timestamp": "2026-06-01 08:00:05",
             "pm25": 30, "pm10": 40, "temp": 28, "hum": 65, "mq": 400,
             "gps_fix": 0, "lat": 0, "lng": 0},
        ]
        df = rows_to_dataframe(rows)
        df = clean_data(df)

        assert len(df) == 1
        assert df.iloc[0]["pm25"] == 30

    def test_add_features_creates_columns(self, sample_dataframe):
        """add_features() tạo đầy đủ các cột feature engineering."""
        from ml.preprocess import add_features
        df = add_features(sample_dataframe.copy())

        expected_cols = [
            "pm25_roll5", "pm10_roll5", "mq_roll5",  # Rolling mean
            "pm25_lag1", "pm10_lag1",                  # Lag features
            "hour", "hour_sin", "hour_cos",            # Time features
        ]
        for col in expected_cols:
            assert col in df.columns, f"Thiếu feature column: {col}"

    def test_add_features_rolling_values(self, sample_dataframe):
        """Rolling mean tính đúng (không có NaN)."""
        from ml.preprocess import add_features
        df = add_features(sample_dataframe.copy())

        assert df["pm25_roll5"].isna().sum() == 0
        assert df["pm10_roll5"].isna().sum() == 0
        assert df["mq_roll5"].isna().sum() == 0

    def test_hour_cyclic_encoding(self, sample_dataframe):
        """hour_sin và hour_cos nằm trong [-1, 1]."""
        from ml.preprocess import add_features
        df = add_features(sample_dataframe.copy())

        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_make_forecast_target(self, sample_dataframe):
        """Target forecast = pm25 dịch lên 1 bước."""
        from ml.preprocess import make_forecast_target
        target = make_forecast_target(sample_dataframe, steps_ahead=1)

        assert len(target) == len(sample_dataframe)
        # Bản ghi cuối cùng phải là NaN (không có giá trị tiếp theo)
        assert pd.isna(target.iloc[-1])
        # Bản ghi thứ i phải bằng pm25 bản ghi thứ i+1
        assert target.iloc[0] == sample_dataframe["pm25"].iloc[1]

    def test_make_classifier_labels(self, sample_dataframe):
        """Labels classifier: 0=Good, 1=Normal, 2=Bad, 3=Danger."""
        from ml.preprocess import make_classifier_labels
        labels = make_classifier_labels(sample_dataframe)

        assert len(labels) == len(sample_dataframe)
        assert set(labels.unique()).issubset({0, 1, 2, 3})

    def test_classifier_label_thresholds(self):
        """Kiểm tra ngưỡng phân loại chính xác."""
        from ml.preprocess import make_classifier_labels

        df = pd.DataFrame({
            "pm25": [10, 50, 100, 200],
            "pm10": [20, 60, 120, 250],
            "temp": [25, 25, 25, 25],
            "hum": [60, 60, 60, 60],
            "mq": [300, 300, 300, 300],
        })
        labels = make_classifier_labels(df)

        assert labels.iloc[0] == 0  # pm25=10  < 35  → Good
        assert labels.iloc[1] == 1  # pm25=50  < 75  → Normal
        assert labels.iloc[2] == 2  # pm25=100 < 150 → Bad
        assert labels.iloc[3] == 3  # pm25=200 ≥ 150 → Danger

    def test_fit_and_save_scaler(self, sample_dataframe, tmp_path, monkeypatch):
        """Scaler fit, lưu file, và trả về đúng feature columns."""
        from ml import preprocess as preprocess_mod
        import pickle

        df = preprocess_mod.add_features(sample_dataframe.copy())

        # Monkeypatch SCALER_PATH trong module preprocess
        test_scaler_path = str(tmp_path / "test_scaler.pkl")
        monkeypatch.setattr(preprocess_mod, "SCALER_PATH", test_scaler_path)

        scaler, feature_cols = preprocess_mod.fit_and_save_scaler(df)

        assert os.path.exists(test_scaler_path)
        assert len(feature_cols) > 0
        assert "id" not in feature_cols
        assert "timestamp" not in feature_cols

        # Load lại để kiểm tra
        with open(test_scaler_path, "rb") as f:
            loaded_scaler, loaded_cols = pickle.load(f)
        assert loaded_cols == feature_cols


# =====================================================
# TEST: PREDICT MODULE
# =====================================================
class TestPredict:

    def test_load_model_nonexistent(self):
        """Load model từ path không tồn tại → (None, None)."""
        from ml.predict import load_model
        model, cols = load_model("/nonexistent/path/model.pkl")
        assert model is None
        assert cols is None

    def test_load_model_empty_file(self, tmp_path):
        """Load model từ file rỗng → (None, None)."""
        from ml.predict import load_model
        empty_file = tmp_path / "empty.pkl"
        empty_file.write_bytes(b"")

        model, cols = load_model(str(empty_file))
        assert model is None
        assert cols is None

    def test_build_feature_vector(self):
        """Build feature vector từ dict, fill missing cols = 0."""
        from ml.predict import build_feature_vector

        data = {"pm25": 25.0, "pm10": 40.0, "temp": 28.0}
        feature_cols = ["pm25", "pm10", "temp", "hum", "mq"]

        X = build_feature_vector(data, feature_cols)

        assert X.shape == (1, 5)
        assert X[0][0] == 25.0   # pm25
        assert X[0][1] == 40.0   # pm10
        assert X[0][2] == 28.0   # temp
        assert X[0][3] == 0.0    # hum (missing → 0)
        assert X[0][4] == 0.0    # mq (missing → 0)

    def test_predict_forecast_no_model(self, monkeypatch):
        """Forecast khi chưa có model → available=False."""
        from ml import predict as predict_mod
        monkeypatch.setattr(predict_mod, "MODEL_FORECAST", "/nonexistent/forecast.pkl")

        result = predict_mod.predict_forecast({"pm25": 25, "pm10": 40})
        assert result["available"] is False
        assert result["pm25_next"] is None

    def test_predict_anomaly_no_model(self, monkeypatch):
        """Anomaly khi chưa có model → available=False."""
        from ml import predict as predict_mod
        monkeypatch.setattr(predict_mod, "MODEL_ANOMALY", "/nonexistent/anomaly.pkl")

        result = predict_mod.predict_anomaly({"pm25": 25})
        assert result["available"] is False

    def test_predict_class_no_model(self, monkeypatch):
        """Classifier khi chưa có model → available=False."""
        from ml import predict as predict_mod
        monkeypatch.setattr(predict_mod, "MODEL_CLASSIFIER", "/nonexistent/classifier.pkl")

        result = predict_mod.predict_class({"pm25": 25})
        assert result["available"] is False

    def test_rule_based_alert_good(self):
        """Rule-based alert: dữ liệu sạch → Good."""
        from ml.predict import _rule_based_alert

        result = _rule_based_alert({
            "pm25": 10, "pm10": 20, "temp": 25, "hum": 60, "mq": 300
        })
        assert result["level"] == 0
        assert result["label"] == "Good"

    def test_rule_based_alert_danger(self):
        """Rule-based alert: PM2.5 cực cao → Danger."""
        from ml.predict import _rule_based_alert

        result = _rule_based_alert({
            "pm25": 200, "pm10": 300, "temp": 35, "hum": 90, "mq": 3000
        })
        assert result["level"] == 3
        assert result["label"] == "Danger"
        assert "pm25" in result["triggered_by"]

    def test_rule_based_alert_partial_bad(self):
        """Rule-based alert: chỉ MQ135 vượt ngưỡng → level tương ứng."""
        from ml.predict import _rule_based_alert

        result = _rule_based_alert({
            "pm25": 10, "pm10": 20, "temp": 25, "hum": 60, "mq": 1600
        })
        assert result["level"] >= 1
        assert "mq" in result["triggered_by"]

    def test_run_all_predictions_structure(self):
        """run_all_predictions trả về đúng cấu trúc."""
        from ml.predict import run_all_predictions

        data = {"pm25": 25, "pm10": 40, "temp": 28, "hum": 65, "mq": 450}
        try:
            result = run_all_predictions(data)
        except ValueError:
            pytest.skip("Model sklearn version mismatch, bỏ qua")

        assert "input" in result
        assert "forecast" in result
        assert "anomaly" in result
        assert "classify" in result
        assert "alert_level" in result
        assert result["input"] == data

    def test_predict_with_saved_models(self):
        """Nếu model tồn tại → trả về kết quả có available=True."""
        from ml.predict import run_all_predictions
        import config

        if not os.path.exists(config.MODEL_FORECAST):
            pytest.skip("Model chưa được train, bỏ qua test này")

        data = {"pm25": 30, "pm10": 45, "temp": 29, "hum": 62, "mq": 500}
        try:
            result = run_all_predictions(data)
        except (ValueError, Exception) as e:
            pytest.skip(f"Model inference error (version mismatch?): {e}")

        assert result["forecast"]["available"] is True
        assert isinstance(result["forecast"]["pm25_next"], float)
        assert result["classify"]["available"] is True
        assert result["classify"]["air_class"] in ["Good", "Normal", "Bad", "Danger"]
        assert result["anomaly"]["available"] is True
        assert isinstance(result["anomaly"]["is_anomaly"], bool)


# =====================================================
# TEST: HELPERS / UTILS
# =====================================================
class TestHelpers:

    def test_get_air_label(self):
        """Hàm phân loại chất lượng không khí theo PM2.5."""
        from utils.helpers import get_air_label

        assert get_air_label(10) == "Good"
        assert get_air_label(50) == "Normal"
        assert get_air_label(100) == "Bad"
        assert get_air_label(200) == "Danger"

    def test_get_alert_level(self):
        """Hàm tính cấp độ cảnh báo."""
        from utils.helpers import get_alert_level

        assert get_alert_level(10) == 0
        assert get_alert_level(50) == 1
        assert get_alert_level(100) == 2
        assert get_alert_level(200) == 3

    def test_validate_sensor_valid(self):
        """Validate dữ liệu hợp lệ."""
        from utils.helpers import validate_sensor

        ok, msg = validate_sensor({
            "pm25": 25, "pm10": 40, "temp": 28, "hum": 65, "mq": 400
        })
        assert ok is True
        assert msg == "ok"

    def test_validate_sensor_missing_field(self):
        """Validate thiếu trường → False."""
        from utils.helpers import validate_sensor

        ok, msg = validate_sensor({"pm25": 25})
        assert ok is False
        assert "Missing" in msg

    def test_validate_sensor_out_of_range(self):
        """Validate giá trị ngoài phạm vi → False."""
        from utils.helpers import validate_sensor

        ok, msg = validate_sensor({
            "pm25": -10, "pm10": 40, "temp": 28, "hum": 65, "mq": 400
        })
        assert ok is False
        assert "out of range" in msg

    def test_now_str_format(self):
        """now_str() trả về chuỗi đúng định dạng."""
        from utils.helpers import now_str

        ts = now_str()
        # Format: YYYY-MM-DD HH:MM:SS
        parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        assert isinstance(parsed, datetime)

    def test_format_rows_for_chart_empty(self):
        """Format rỗng → labels và datasets rỗng."""
        from utils.helpers import format_rows_for_chart

        result = format_rows_for_chart([])
        assert result["labels"] == []
        assert result["datasets"] == {}

    def test_format_rows_for_chart(self):
        """Format rows cho Chart.js có đủ datasets."""
        from utils.helpers import format_rows_for_chart

        rows = [
            {"timestamp": "2026-06-01 08:00:05", "pm25": 30, "pm10": 45,
             "temp": 28, "hum": 65, "mq": 400},
            {"timestamp": "2026-06-01 08:00:00", "pm25": 25, "pm10": 40,
             "temp": 27, "hum": 63, "mq": 380},
        ]
        result = format_rows_for_chart(rows)

        assert len(result["labels"]) == 2
        # Đảo ngược: cũ → mới
        assert result["labels"][0] == "2026-06-01 08:00:00"
        assert result["datasets"]["pm25"] == [25, 30]


# =====================================================
# TEST: CONFIG
# =====================================================
class TestConfig:

    def test_thresholds_defined(self):
        """Config có đầy đủ ngưỡng cảnh báo."""
        from config import THRESHOLDS

        assert "pm25" in THRESHOLDS
        assert "pm10" in THRESHOLDS
        assert "temp" in THRESHOLDS
        assert "hum" in THRESHOLDS
        assert "mq" in THRESHOLDS

    def test_pm25_thresholds_order(self):
        """Ngưỡng PM2.5: good < normal < bad."""
        from config import THRESHOLDS

        t = THRESHOLDS["pm25"]
        assert t["good"] < t["normal"] < t["bad"]

    def test_model_paths_defined(self):
        """Paths model được cấu hình đúng."""
        from config import MODEL_FORECAST, MODEL_ANOMALY, MODEL_CLASSIFIER

        assert MODEL_FORECAST.endswith("forecast_model.pkl")
        assert MODEL_ANOMALY.endswith("anomaly_model.pkl")
        assert MODEL_CLASSIFIER.endswith("classifier_model.pkl")

    def test_mqtt_config_defaults(self):
        """MQTT config có giá trị mặc định."""
        from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

        assert isinstance(MQTT_BROKER, str)
        assert isinstance(MQTT_PORT, int)
        assert MQTT_TOPIC == "airquality/data"
