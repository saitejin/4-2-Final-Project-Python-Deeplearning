from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "models" / "grid_processed_data.pkl"
MODEL_PATH = BASE_DIR / "conv_lstm_model.pt"
RANDOM_STATE = 42
VAL_SPLIT = 0.2
BATCH_SIZE = 64
MAX_EPOCHS = 250
PATIENCE = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMNet(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        pooled_seq_len = seq_len // 2
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (batch, channels, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, channels)
        lstm_out, (h_n, _) = self.lstm(x)
        features = h_n[-1]  # last layer hidden state
        features = self.dropout(features)
        logits = self.fc(features).squeeze(-1)
        return logits


@dataclass
class TrainState:
    best_loss: float = float("inf")
    patience_counter: int = 0


def create_dataloaders(X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader]:
    tensors = TensorDataset(
        torch.from_numpy(X).float().permute(0, 2, 1),  # (batch, channels, seq)
        torch.from_numpy(y).float(),
    )
    val_size = int(len(tensors) * VAL_SPLIT)
    train_size = len(tensors) - val_size
    train_ds, val_ds = random_split(tensors, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_STATE))
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().permute(0, 2, 1).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 FINAL ACCURACY: {acc:.4f} (Paper Target: ~0.9130)")
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds, digits=4))


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Loading processed data…")
    data = joblib.load(DATA_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    train_loader, val_loader = create_dataloaders(X_train, y_train)
    model = ConvLSTMNet(seq_len=X_train.shape[1]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5)

    state = TrainState()
    best_state_dict = None

    print("\n--- Training Proposed Vertical Hybrid Conv-LSTM (PyTorch) ---")
    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {state.best_loss:.4f}")

        if val_loss + 1e-5 < state.best_loss:
            state.best_loss = val_loss
            state.patience_counter = 0
            best_state_dict = model.state_dict()
            if epoch % 10 != 0 and epoch != 1:
                print(f"Epoch {epoch:03d} | New best! Val Loss: {val_loss:.4f}")
        else:
            state.patience_counter += 1
            if state.patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                print(f"Best validation loss: {state.best_loss:.4f}")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved best model weights to {MODEL_PATH}")

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()