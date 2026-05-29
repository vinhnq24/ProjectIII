import pandas as pd
import numpy as np
import pickle
import os
from config import SCALER_PATH
from sklearn.preprocessing import StandardScaler


FEATURES = ["pm25", "pm10", "temp", "hum", "mq"]


# =====================================================
# LOAD DATA FROM DB ROWS
# =====================================================
def rows_to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =====================================================
# CLEAN DATA
# =====================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Xóa duplicate
    df = df.drop_duplicates(subset=["timestamp"])

    # Xóa dòng có giá trị âm hoặc bất hợp lý
    df = df[df["pm25"]  >= 0]
    df = df[df["pm10"]  >= 0]
    df = df[df["mq"]    >= 0]
    df = df[df["temp"]  > -40]
    df = df[df["temp"]  < 80]
    df = df[df["hum"]   >= 0]
    df = df[df["hum"]   <= 100]

    # Giới hạn outlier cực đoan (PM2.5 > 1000 là không thực tế)
    df = df[df["pm25"] <= 1000]
    df = df[df["pm10"] <= 1000]

    # Fill NaN bằng giá trị trước đó
    df[FEATURES] = df[FEATURES].fillna(method="ffill").fillna(method="bfill")

    df = df.reset_index(drop=True)
    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling mean 5 bản ghi
    for col in ["pm25", "pm10", "mq"]:
        df[f"{col}_roll5"] = df[col].rolling(5, min_periods=1).mean()

    # Lag 1 (giá trị trước đó)
    df["pm25_lag1"] = df["pm25"].shift(1).fillna(df["pm25"])
    df["pm10_lag1"] = df["pm10"].shift(1).fillna(df["pm10"])

    # Giờ trong ngày (nếu có timestamp)
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# =====================================================
# SCALE
# =====================================================
def fit_and_save_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()

    feature_cols = [c for c in df.columns
                    if c not in ["id", "timestamp"]]

    scaler.fit(df[feature_cols])

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump((scaler, feature_cols), f)

    print(f"[Preprocess] Scaler saved -> {SCALER_PATH}")

    return scaler, feature_cols


def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler, feature_cols = pickle.load(f)
    return scaler, feature_cols


def scale_data(df: pd.DataFrame, scaler, feature_cols) -> pd.DataFrame:
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    return df_scaled


# =====================================================
# PREPARE LABELS
# =====================================================
def make_classifier_labels(df: pd.DataFrame) -> pd.Series:
    """
    0 = Good   (pm25 < 35)
    1 = Normal (pm25 < 75)
    2 = Bad    (pm25 < 150)
    3 = Danger (pm25 >= 150)
    """
    def label(row):
        if row["pm25"] < 35:  return 0
        if row["pm25"] < 75:  return 1
        if row["pm25"] < 150: return 2
        return 3

    return df.apply(label, axis=1)


def make_forecast_target(df: pd.DataFrame,
                         steps_ahead: int = 1) -> pd.Series:
    """
    Target: giá trị pm25 sau `steps_ahead` bản ghi
    """
    return df["pm25"].shift(-steps_ahead)


# =====================================================
# FULL PIPELINE
# =====================================================
def full_preprocess(rows: list):
    df = rows_to_dataframe(rows)
    df = clean_data(df)
    df = add_features(df)

    scaler, feature_cols = fit_and_save_scaler(df)
    df_scaled = scale_data(df, scaler, feature_cols)

    return df, df_scaled, feature_cols