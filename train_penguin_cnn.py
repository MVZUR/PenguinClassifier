# train_penguin_cnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Ustawienia
CSV_PATH = "data/penguins_size.csv"
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset
class PenguinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# Model - 1D CNN
class Penguin1DCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Załadowanie datasetu i preprocessing
df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["species"])
numeric_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

# Wypełnienie pustych danych numerycznych ich medianą
for c in numeric_cols:
    if c not in df.columns:
        raise SystemExit(f"Column {c} not found in CSV. Check file.")
    df[c] = df[c].fillna(df[c].median())

# Dane wejściowe i klasy
X = df[numeric_cols].values
y_labels = df["species"].values
le = LabelEncoder()
y = le.fit_transform(y_labels)  # map species -> integers

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Standaryzacja
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_ds = PenguinDataset(X_train, y_train)
test_ds = PenguinDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

n_features = X_train.shape[1]
n_classes = len(le.classes_)

model = Penguin1DCNN(n_features=n_features, n_classes=n_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Trenowanie
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total += yb.size(0)
    return correct/total

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*yb.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(model, test_loader)
    if val_acc > best_acc:
        best_acc = val_acc
    print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Test acc: {val_acc:.4f}")

# Ewaluacja i zapis
test_acc = evaluate(model, test_loader)
print(f"Final test accuracy (float model): {test_acc:.4f}")
os.makedirs("models", exist_ok=True)
torch.save(model, "models/model.pt")

# Rozmiar
float_size = os.path.getsize("models/model.pt")
print(f"Float model file size: {float_size/1024:.2f} KB")

# Dynamiczna kwantyzacja
model_cpu = model.to('cpu')
model_q = torch.quantization.quantize_dynamic(
    model_cpu, {torch.nn.Linear}, dtype=torch.qint8
)
# Test skwantyfikowanego modelu
def evaluate_cpu(m, loader):
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = m(xb)  # loader yields CPU tensors by default
            preds = out.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total += yb.size(0)
    return correct/total


q_acc = evaluate_cpu(model_q, test_loader)
torch.save(model_q, "models/model_quantized_dynamic.pt")
quant_size = os.path.getsize("models/model_quantized_dynamic.pt")
print(f"Quantized model test accuracy: {q_acc:.4f}")
print(f"Quantized model file size: {quant_size/1024:.2f} KB")
print(f"Size reduction: {float_size/1024:.2f} KB -> {quant_size/1024:.2f} KB ({100*(quant_size/float_size):.1f}% of original)")

