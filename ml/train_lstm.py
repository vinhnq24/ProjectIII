"""
LSTM vs Random Forest — So sánh mô hình dự báo PM2.5 (PyTorch)
Chạy: python -m ml.train_lstm

Script này train một mô hình LSTM (PyTorch) và so sánh với Random Forest hiện tại
trên cùng tập dữ liệu, cùng cách chia train/test (time-based split).

Kết quả: In bảng so sánh MAE, RMSE và lưu biểu đồ vào saved_models/comparison/
"""

import os
import sys

# Đảm bảo import được các module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thiết lập môi trường để tránh lỗi OpenMP DLL conflict trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import torch đầu tiên để load DLLs trước các thư viện khác
try:
    import torch
except Exception:
    pass

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from config import (
    MODEL_FORECAST, SAVED_MODELS_DIR,
    TEST_SIZE, RANDOM_STATE
)
from database.db import get_all
from ml.preprocess import rows_to_dataframe, clean_data, add_features


# =====================================================
# CẤU HÌNH LSTM
# =====================================================
SEQUENCE_LENGTH = 10        # Số bước thời gian nhìn lại (lookback)
HIDDEN_SIZE = 64            # Số hidden units trong LSTM
NUM_LAYERS = 2              # Số lớp LSTM chồng nhau
DROPOUT = 0.2               # Tỷ lệ dropout
EPOCHS = 50                 # Số epoch huấn luyện tối đa
BATCH_SIZE = 32             # Batch size
LEARNING_RATE = 0.001       # Tốc độ học (learning rate)
PATIENCE = 10               # Số epoch chờ để early stopping

COMPARISON_DIR = os.path.join(SAVED_MODELS_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)


# =====================================================
# TẠO SEQUENCES CHO LSTM
# =====================================================
def create_sequences(X, y, seq_length):
    """
    Tạo chuỗi thời gian dạng (samples, sequence_length, features) cho LSTM.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# =====================================================
# HUẤN LUYỆN MÔ HÌNH LSTM (PyTorch)
# =====================================================
def build_and_train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, n_features):
    """Xây dựng và huấn luyện mô hình LSTM bằng PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_STATE)

    # Chuyển đổi dữ liệu sang PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train_seq)
    y_train_t = torch.FloatTensor(y_train_seq)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_test_t = torch.FloatTensor(y_test_seq)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Định nghĩa cấu trúc mạng LSTM
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super(LSTMForecaster, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size // 2, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # Lấy đầu ra của bước thời gian cuối cùng
            out = self.fc1(last_hidden)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out.squeeze()

    model = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

    # In thông tin cấu trúc mạng
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[LSTM] Model Architecture:")
    print(f"  Input size:  {n_features} (features) x {SEQUENCE_LENGTH} (timesteps)")
    print(f"  LSTM Layers: {NUM_LAYERS} layers x {HIDDEN_SIZE} units")
    print(f"  Total Params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Vòng lặp huấn luyện có Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

    print(f"\n[LSTM] Training model...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Kiểm tra Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == PATIENCE:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  -> Kích hoạt Early stopping ở epoch {epoch+1}")
            break

    # Phục hồi trọng số tốt nhất
    model.load_state_dict(best_state)
    actual_epochs = epoch + 1

    # Dự báo trên tập test
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    return model, y_pred, train_losses, val_losses, actual_epochs


# =====================================================
# EVALUATE RANDOM FOREST
# =====================================================
def evaluate_random_forest(X_test, y_test):
    """Đánh giá Random Forest trên cùng tập test."""
    if not os.path.exists(MODEL_FORECAST):
        print("[RF] Model chưa được train!")
        return None

    with open(MODEL_FORECAST, "rb") as f:
        rf_model, rf_feature_cols = pickle.load(f)

    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    return {
        "model": "Random Forest",
        "mae": round(mae_rf, 4),
        "rmse": round(rmse_rf, 4),
        "y_pred": y_pred_rf,
    }


# =====================================================
# VẼ BIỂU ĐỒ SO SÁNH
# =====================================================
def plot_comparison(y_test, y_pred_rf, y_pred_lstm, mae_rf, mae_lstm,
                    rmse_rf, rmse_lstm):
    """Vẽ biểu đồ so sánh dự đoán vs thực tế."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('So sánh LSTM (PyTorch) vs Random Forest — Dự báo PM2.5',
                     fontsize=14, fontweight='bold')

        # --- 1. RF: Actual vs Predicted ---
        n_show = min(100, len(y_pred_rf))
        ax1 = axes[0, 0]
        ax1.plot(y_test[:n_show], label='Thực tế', color='#1e293b', linewidth=1.5)
        ax1.plot(y_pred_rf[:n_show], label='Random Forest', color='#3b82f6',
                 linewidth=1.5, alpha=0.8)
        ax1.set_title(f'Random Forest: Dự báo vs Thực tế ({n_show} mẫu)')
        ax1.set_xlabel('Mẫu')
        ax1.set_ylabel('PM2.5 (ug/m3)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # --- 2. LSTM: Actual vs Predicted ---
        n_show_lstm = min(100, len(y_pred_lstm))
        ax2 = axes[0, 1]
        ax2.plot(y_test[-len(y_pred_lstm):][:n_show_lstm], label='Thực tế',
                 color='#1e293b', linewidth=1.5)
        ax2.plot(y_pred_lstm[:n_show_lstm], label='LSTM Model', color='#ef4444',
                 linewidth=1.5, alpha=0.8)
        ax2.set_title(f'LSTM: Dự báo vs Thực tế ({n_show_lstm} mẫu)')
        ax2.set_xlabel('Mẫu')
        ax2.set_ylabel('PM2.5 (ug/m3)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # --- 3. MAE & RMSE Comparison ---
        ax3 = axes[1, 0]
        models = ['Random Forest', 'LSTM (PyTorch)']
        mae_vals = [mae_rf, mae_lstm]
        rmse_vals = [rmse_rf, rmse_lstm]

        x = np.arange(len(models))
        width = 0.35
        bars1 = ax3.bar(x - width/2, mae_vals, width, label='MAE',
                        color=['#3b82f6', '#ef4444'])
        bars2 = ax3.bar(x + width/2, rmse_vals, width, label='RMSE',
                        color=['#60a5fa', '#f87171'])
        ax3.set_title('So sánh MAE & RMSE')
        ax3.set_ylabel('Error (ug/m3)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        for bar in bars1:
            ax3.annotate(f'{bar.get_height():.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold')
        for bar in bars2:
            ax3.annotate(f'{bar.get_height():.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold')

        # --- 4. Scatter Plot ---
        ax4 = axes[1, 1]
        n_scatter = min(len(y_pred_lstm), len(y_pred_rf), len(y_test))
        ax4.scatter(y_test[:n_scatter], y_pred_lstm[:n_scatter],
                    alpha=0.5, s=10, color='#ef4444', label='LSTM')
        ax4.scatter(y_test[:n_scatter], y_pred_rf[:n_scatter],
                    alpha=0.5, s=10, color='#3b82f6', label='Random Forest')
        min_val = min(y_test[:n_scatter].min(), 0)
        max_val = max(y_test[:n_scatter].max(), 50)
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                 label='Dự báo hoàn hảo')
        ax4.set_title('Scatter Plot: Thực tế vs Dự báo')
        ax4.set_xlabel('PM2.5 Thực tế (ug/m3)')
        ax4.set_ylabel('PM2.5 Dự báo (ug/m3)')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(COMPARISON_DIR, "lstm_vs_rf_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[Chart] Đã lưu biểu đồ so sánh: {chart_path}")
        return chart_path
    except ImportError:
        print("[Chart] matplotlib chưa cài. Bỏ qua vẽ biểu đồ.")
        return None


# =====================================================
# MAIN: TRAIN & COMPARE
# =====================================================
def run_comparison():
    print("=" * 60)
    print("  LSTM (PyTorch) vs Random Forest - PM2.5 Forecast Comparison")
    print("=" * 60)

    # 1. Load data
    print("\n[Data] Loading data from database...")
    rows = get_all()
    if len(rows) < 100:
        print(f"[Error] Không đủ dữ liệu: {len(rows)} records (cần >= 100)")
        return

    df = rows_to_dataframe(rows)
    df = clean_data(df)
    df = add_features(df)
    print(f"[Data] Loaded {len(df)} records sau khi tiền xử lý")

    # 2. Features & Target
    feature_cols = [c for c in df.columns if c not in ["id", "timestamp"]]
    target_col = "pm25"

    df["target"] = df[target_col].shift(-1)
    df = df.dropna(subset=["target"])

    X = df[feature_cols].values
    y = df["target"].values

    # 3. Split
    n = len(X)
    n_test = max(1, int(n * TEST_SIZE))
    split_idx = n - n_test

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)} (ratio: {TEST_SIZE})")

    # =====================================================
    # RANDOM FOREST EVALUATION
    # =====================================================
    print("\n" + "-" * 50)
    print("  RANDOM FOREST (Mô hình hiện tại)")
    print("-" * 50)

    rf_result = evaluate_random_forest(X_test, y_test)
    if rf_result:
        print(f"  MAE  = {rf_result['mae']:.4f} ug/m3")
        print(f"  RMSE = {rf_result['rmse']:.4f} ug/m3")

    # =====================================================
    # LSTM TRAINING
    # =====================================================
    print("\n" + "-" * 50)
    print("  LSTM (Deep Learning - PyTorch)")
    print("-" * 50)

    # Scale dữ liệu cho LSTM (MinMax scaling là tốt nhất cho LSTM)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Tạo sequences chuỗi thời gian
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    print(f"[LSTM] Sequences: Train {X_train_seq.shape} | Test {X_test_seq.shape}")

    if len(X_test_seq) < 1:
        print("[Error] Không đủ dữ liệu test cho chuỗi LSTM!")
        return

    n_features = X_train_seq.shape[2]

    # Huấn luyện model LSTM
    lstm_model, y_pred_lstm_scaled, train_losses, val_losses, actual_epochs = \
        build_and_train_lstm(
            X_train_seq, y_train_seq,
            X_test_seq, y_test_seq,
            n_features
        )

    # Khôi phục tỷ lệ (inverse scale) về đơn vị đo gốc
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
    y_test_lstm = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))

    print(f"\n[LSTM] Kết quả trên tập test:")
    print(f"  MAE  = {mae_lstm:.4f} ug/m3")
    print(f"  RMSE = {rmse_lstm:.4f} ug/m3")
    print(f"  Số Epoch đã chạy: {actual_epochs}/{EPOCHS}")

    # Lưu model LSTM
    import torch
    lstm_path = os.path.join(COMPARISON_DIR, "lstm_forecast_model.pt")
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"[LSTM] Đã lưu model: {lstm_path}")

    # Lưu scalers
    scaler_path = os.path.join(COMPARISON_DIR, "lstm_scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y,
                      "feature_cols": feature_cols,
                      "seq_length": SEQUENCE_LENGTH}, f)

    # =====================================================
    # COMPARISON RESULTS
    # =====================================================
    print("\n" + "=" * 60)
    print("  KẾT QUẢ SO SÁNH")
    print("=" * 60)

    # Naive baseline
    naive_pred = X_test[:len(y_test), feature_cols.index(target_col)]
    mae_naive = mean_absolute_error(y_test, naive_pred)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))

    print(f"\n  {'Model':<20} {'MAE (ug/m3)':<15} {'RMSE (ug/m3)':<15} {'vs Naive':<15}")
    print(f"  {'-' * 65}")
    print(f"  {'Naive Baseline':<20} {mae_naive:<15.4f} {rmse_naive:<15.4f} {'---':<15}")

    winner = "N/A"
    if rf_result:
        rf_improve = ((mae_naive - rf_result['mae']) / mae_naive * 100) if mae_naive > 0 else 0
        print(f"  {'Random Forest':<20} {rf_result['mae']:<15.4f} {rf_result['rmse']:<15.4f} {rf_improve:+.1f}%")

    lstm_improve = ((mae_naive - mae_lstm) / mae_naive * 100) if mae_naive > 0 else 0
    print(f"  {'LSTM (PyTorch)':<20} {mae_lstm:<15.4f} {rmse_lstm:<15.4f} {lstm_improve:+.1f}%")

    if rf_result:
        if rf_result['mae'] < mae_lstm:
            winner = "Random Forest"
            diff = mae_lstm - rf_result['mae']
        else:
            winner = "LSTM"
            diff = rf_result['mae'] - mae_lstm
        print(f"\n  >>> Winner: {winner} (thấp hơn {diff:.4f} ug/m3)")

    # Lưu kết quả dạng json
    comparison_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_records": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "sequence_length": SEQUENCE_LENGTH,
        "lstm_hidden_size": HIDDEN_SIZE,
        "lstm_num_layers": NUM_LAYERS,
        "epochs_trained": int(actual_epochs),
        "framework": "PyTorch (LSTM)",
        "results": {
            "naive": {"mae": float(mae_naive), "rmse": float(rmse_naive)},
            "random_forest": rf_result and {
                "mae": float(rf_result["mae"]), "rmse": float(rf_result["rmse"])
            },
            "lstm": {"mae": float(mae_lstm), "rmse": float(rmse_lstm)},
        },
        "winner": winner,
    }

    result_path = os.path.join(COMPARISON_DIR, "comparison_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=4, ensure_ascii=False)
    print(f"\n[Save] Đã lưu kết quả so sánh: {result_path}")

    # Vẽ biểu đồ so sánh
    if rf_result:
        plot_comparison(
            y_test_lstm, rf_result["y_pred"][-len(y_pred_lstm):],
            y_pred_lstm,
            rf_result["mae"], mae_lstm,
            rf_result["rmse"], rmse_lstm
        )

    print("\n" + "=" * 60)
    print("  LSTM vs Random Forest comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_comparison()
