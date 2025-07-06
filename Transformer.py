""
# satellite_transformer_training_v2.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Step 1: Load and normalize data
df = pd.read_csv("drive/MyDrive/PFE/satellite_dataset.csv", parse_dates=["timestamp"])
df.sort_values(by=["norad_id", "timestamp"], inplace=True)

# Features: position, velocity, sidereal time, earth rotation angle
features = [
    "x_km", "y_km", "z_km",
    "vx_km_s", "vy_km_s", "vz_km_s",
    "sidereal_time_hours", "earth_rotation_angle_rad"
]
targets = ["x_km", "y_km", "z_km"]

# Use StandardScaler for feature normalization
feature_scaler = StandardScaler()
df[features] = feature_scaler.fit_transform(df[features])

# ✅ Step 2: Create sequence dataset
def build_sequences(df, input_steps=48):
    X, y = [], []
    for _, group in df.groupby("norad_id"):
        data = group[features + targets].values
        for i in range(len(data) - input_steps):
            X.append(data[i:i+input_steps, :len(features)])  # input
            y.append(data[i+input_steps, :3])                # target x, y, z
    return np.array(X), np.array(y)

X, y = build_sequences(df, input_steps=48)

# ✅ Step 3: Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

def to_loader(X, y, batch_size=64):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

train_loader = to_loader(X_train, y_train)
val_loader = to_loader(X_val, y_val)
test_loader = to_loader(X_test, y_test)

# ✅ Step 4: Transformer model
class SatelliteTransformer(nn.Module):
    def __init__(self, seq_len=48, d_model=64, nhead=4, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(8, d_model)  # 8 features as input
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)  # Output x, y, z

    def forward(self, x):
        x = self.input_proj(x.to(device)) + self.pos_encoder
        x = self.transformer(x)
        return self.fc(x[:, -1])  # Predict only last step

model = SatelliteTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
scaler = GradScaler()  

# ✅ Step 5: Training loop
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        print(f"\n⏱️ Epoch {epoch+1}")
        model.train()
        total_train_loss = 0

        for xb, yb in tqdm(train_loader, desc="Training"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item() * xb.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # ✅ Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Validation"):
                xb, yb = xb.to(device), yb.to(device)
                with autocast():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                total_val_loss += loss.item() * xb.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)
        print(f"📉 Epoch {epoch+1:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

train_model(model, train_loader, val_loader, epochs=1)

# ✅ Step 6: Evaluation
def evaluate(model, test_loader, feature_scaler):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            all_preds.append(pred)
            all_targets.append(yb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)

    mse = total_loss / len(test_loader.dataset)
    rmse_norm = np.sqrt(mse)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Padding with zeros to use inverse_transform
    preds_km = feature_scaler.inverse_transform(np.concatenate([preds, np.zeros((preds.shape[0], 5))], axis=1))[:, :3]
    targets_km = feature_scaler.inverse_transform(np.concatenate([targets, np.zeros((targets.shape[0], 5))], axis=1))[:, :3]

    real_rmse_axis = np.sqrt(np.mean((preds_km - targets_km)**2, axis=0))
    real_rmse_total = np.sqrt(np.mean(np.sum((preds_km - targets_km)**2, axis=1)))

    print(f"\n📊 Normalized MSE: {mse:.4f}")
    print(f"📏 Normalized RMSE: {rmse_norm:.4f}")
    print(f"🚐 Real RMSE per axis: X={real_rmse_axis[0]:.2f} km, Y={real_rmse_axis[1]:.2f} km, Z={real_rmse_axis[2]:.2f} km")
    print(f"🧽 Total Euclidean 3D error: {real_rmse_total:.2f} km")

evaluate(model, test_loader, feature_scaler)
