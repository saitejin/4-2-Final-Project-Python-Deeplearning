"""
Model Loading for Smart Grid Anomaly Detection
"""

import joblib
import torch
import torch.nn as nn
from pathlib import Path
import streamlit as st


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
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        features = h_n[-1]
        features = self.dropout(features)
        logits = self.fc(features).squeeze(-1)
        return logits


@st.cache_resource
def load_model():
    """Load trained PyTorch model and processed data"""
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_PATH = BASE_DIR / "conv_lstm_model.pt"
        DATA_PATH = BASE_DIR / "models" / "grid_processed_data.pkl"
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data
        data = joblib.load(DATA_PATH)
        seq_len = data['X_train'].shape[1]
        
        # Load model
        model = ConvLSTMNet(seq_len=seq_len)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        return model, data, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None
