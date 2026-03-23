import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for [B, T, D]."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerRegressor(nn.Module):
    """
    Windowed sequence regressor:
    [B, t, input_dim] -> MLP -> [B, t, hidden_dim] -> Transformer -> [B, t]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        window_size: int,
        mlp_hidden_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        mlp_hidden_dim = mlp_hidden_dim or hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_mlp(x)
        x = self.positional_encoding(x)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=mask)
        return self.output_head(x).squeeze(-1)
