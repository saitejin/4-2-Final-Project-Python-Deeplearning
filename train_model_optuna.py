from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "models" / "grid_processed_data.pkl"
MODEL_PATH = BASE_DIR / "conv_lstm_model_optimized.pt"
OPTUNA_STUDY_PATH = BASE_DIR / "models" / "optuna_study.db"
RANDOM_STATE = 42
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optuna configuration (as per paper: 50 trials, 50 epochs each)
N_TRIALS = 50
OPTUNA_EPOCHS = 50


class ConvLSTMNet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        conv_channels: int,
        lstm_hidden: int,
        lstm_layers: int,
        fc_hidden: int,
        dropout_conv: float,
        dropout_lstm: float,
        dropout_fc: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_conv),
        )

        pooled_seq_len = seq_len // 2
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        features = h_n[-1]
        features = self.dropout(features)
        logits = self.fc(features).squeeze(-1)
        return logits


@dataclass
class TrainState:
    best_loss: float = float("inf")
    patience_counter: int = 0


def create_dataloaders(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    tensors = TensorDataset(
        torch.from_numpy(X).float().permute(0, 2, 1),
        torch.from_numpy(y).float(),
    )
    val_size = int(len(tensors) * VAL_SPLIT)
    train_size = len(tensors) - val_size
    train_ds, val_ds = random_split(
        tensors, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
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
            optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, seq_len: int):
    """Optuna objective function for hyperparameter optimization."""
    
    # Hyperparameters to optimize (based on paper Table 4)
    conv_channels = trial.suggest_categorical("conv_channels", [64, 128, 256])
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [50, 100, 150, 200])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    fc_hidden = trial.suggest_categorical("fc_hidden", [256, 512, 1024])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    
    dropout_conv = trial.suggest_float("dropout_conv", 0.1, 0.5)
    dropout_lstm = trial.suggest_float("dropout_lstm", 0.1, 0.5)
    dropout_fc = trial.suggest_float("dropout_fc", 0.2, 0.6)
    
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size)
    
    # Create model
    model = ConvLSTMNet(
        seq_len=seq_len,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        fc_hidden=fc_hidden,
        dropout_conv=dropout_conv,
        dropout_lstm=dropout_lstm,
        dropout_fc=dropout_fc,
        kernel_size=kernel_size,
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    
    state = TrainState()
    patience = 10
    
    # Train for OPTUNA_EPOCHS (50 as per paper)
    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        if val_loss + 1e-5 < state.best_loss:
            state.best_loss = val_loss
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            if state.patience_counter >= patience:
                break
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return state.best_loss


def evaluate(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().permute(0, 2, 1).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 FINAL ACCURACY: {acc:.4f} (Paper Target: ~0.9130)")
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds, digits=4))
    return acc


def train_with_best_params(
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int,
    max_epochs: int = 250,
):
    """Train final model with best hyperparameters."""
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 80)
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, best_params["batch_size"]
    )
    
    model = ConvLSTMNet(
        seq_len=seq_len,
        conv_channels=best_params["conv_channels"],
        lstm_hidden=best_params["lstm_hidden"],
        lstm_layers=best_params["lstm_layers"],
        fc_hidden=best_params["fc_hidden"],
        dropout_conv=best_params["dropout_conv"],
        dropout_lstm=best_params["dropout_lstm"],
        dropout_fc=best_params["dropout_fc"],
        kernel_size=best_params["kernel_size"],
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    
    state = TrainState()
    best_state_dict = None
    patience = 20
    
    print(f"Training for up to {max_epochs} epochs with patience={patience}...")
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss + 1e-5 < state.best_loss:
            state.best_loss = val_loss
            state.patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            state.patience_counter += 1
            if state.patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved optimized model weights to {MODEL_PATH}")
    
    return evaluate(model, X_test, y_test)


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR ConvLSTM")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {OPTUNA_EPOCHS}")
    print("=" * 80)
    
    print("\nLoading processed data…")
    data = joblib.load(DATA_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    seq_len = X_train.shape[1]
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Sequence length: {seq_len}")
    
    # Create Optuna study
    study_storage = f"sqlite:///{OPTUNA_STUDY_PATH}"
    study = optuna.create_study(
        study_name="convlstm_optimization",
        direction="minimize",
        storage=study_storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    print(f"\nStarting Optuna optimization with {N_TRIALS} trials...")
    print("This may take a while...\n")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, seq_len),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest trial:")
    print(f"  Value (Val Loss): {study.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    final_accuracy = train_with_best_params(
        study.best_params,
        X_train,
        y_train,
        X_test,
        y_test,
        seq_len,
        max_epochs=250,
    )
    
    print("\n" + "=" * 80)
    print(f"FINAL TEST ACCURACY: {final_accuracy:.4f}")
    print(f"Paper Target: 0.9130")
    print(f"Improvement needed: {(0.9130 - final_accuracy):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
