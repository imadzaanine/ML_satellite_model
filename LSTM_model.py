import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1: Load CSV
# -------------------------------
df = pd.read_csv("satellite_dataset.csv", parse_dates=["timestamp"])
df.sort_values(by=["norad_id", "timestamp"], inplace=True)

# -------------------------------
# STEP 2: Build Time Series Sequences
# -------------------------------
def build_sequences(df, input_steps=5):
    X, y = [], []
    grouped = df.groupby("norad_id")
    for _, group in grouped:
        group = group.reset_index(drop=True)
        coords = group[["x_km", "y_km", "z_km"]].values
        for i in range(len(coords) - input_steps):
            X.append(coords[i:i+input_steps])
            y.append(coords[i+input_steps])
    return np.array(X), np.array(y)

X, y = build_sequences(df, input_steps=5)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# -------------------------------
# STEP 3: Train/Val/Test Split
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# -------------------------------
# STEP 4: Convert to PyTorch
# -------------------------------
def to_loader(X, y, batch_size=64):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

train_loader = to_loader(X_train, y_train)
val_loader = to_loader(X_val, y_val)
test_loader = to_loader(X_test, y_test)

# -------------------------------
# STEP 5: Define LSTM Model
# -------------------------------
class SatelliteLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last timestep

model = SatelliteLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# -------------------------------
# STEP 6: Training Loop
# -------------------------------
def train_model(model, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

train_model(model, train_loader, val_loader)

# -------------------------------
# STEP 7: Evaluate on Test Set
# -------------------------------
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\n🧪 Test Loss (MSE): {avg_loss:.4f}")
    print(f"🛰️ Approx error per coordinate: {np.sqrt(avg_loss):.2f} km")

evaluate(model, test_loader)
