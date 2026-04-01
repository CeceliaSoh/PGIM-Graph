import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MyLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):
    __constants__ = ['num_parameters']

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.alpha = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.alpha)


def get_activation(activation):
    if activation == "prelu":
        return MyPReLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation is None:
        return nn.Identity()
    elif callable(activation):
        return activation()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")


class MyMLP(nn.Module):
    def __init__(self, in_channels, units_list, activation, drop_rate, bn,
                 output_activation, output_drop_rate, output_bn, output_bias=True,
                 ln=False, output_ln=False, eps=1e-5):
        super().__init__()

        layers = []
        units_list = [in_channels] + units_list

        for i in range(len(units_list) - 1):
            if i < len(units_list) - 2:
                layers.append(MyLinear(units_list[i], units_list[i + 1]))
                if bn:
                    layers.append(nn.BatchNorm1d(units_list[i + 1], eps=eps))
                if ln:
                    layers.append(nn.LayerNorm(units_list[i + 1], eps=eps))
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(drop_rate))
            else:
                layers.append(MyLinear(units_list[i], units_list[i + 1], bias=output_bias))
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i + 1], eps=eps))
                if output_ln:
                    layers.append(nn.LayerNorm(units_list[i + 1], eps=eps))
                layers.append(get_activation(output_activation))
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Causal Transformer layer  (dynamic sequence length — no fixed buffer)
# ---------------------------------------------------------------------------

def _sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding, computed at runtime for any T."""
    position = torch.arange(T, device=device).unsqueeze(1).float()       # (T, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )                                                                       # (d_model/2,)
    pe = torch.zeros(T, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)                                                  # (1, T, d_model)


class CausalTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            MyLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            MyLinear(hidden_dim * 4, hidden_dim),
        )
        self.ln1     = nn.LayerNorm(hidden_dim)
        self.ln2     = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, hidden_dim)"""
        T = x.size(1)
        # upper-triangle = -inf  →  causal mask (attend only to past & current)
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# GNN Regressor  (hop encoder  →  mean pool  →  causal transformer  →  head)
# ---------------------------------------------------------------------------

class GNNRegressor(nn.Module):
    """
    Full spatial-temporal regressor:
      1. Per-hop MLP encodes each GNN hop independently.
      2. Mean pool across hops  →  (batch, T, hidden_dim).
      3. Causal Transformer layers model temporal dependencies (GPT-style).
      4. Linear head  →  next-step price prediction at every token.

    Input:  (batch, num_hops, T, feat_dim)
    Output: (batch, T, out_dim)
    """

    def __init__(self, num_hops: int, feat_dim: int, hidden_dim: int, out_dim: int,
                 mlp_layers: int = 1, num_transformer_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()

        units_list = [hidden_dim] * mlp_layers

        # --- spatial encoder (one MLP per hop) ---
        self.hop_mlps = nn.ModuleList([
            MyMLP(
                in_channels=feat_dim,
                units_list=units_list,
                activation="relu",
                drop_rate=dropout,
                bn=False,
                output_activation=None,
                output_drop_rate=0.0,
                output_bn=False,
            )
            for _ in range(num_hops)
        ])

        # --- temporal encoder ---
        self.transformer_layers = nn.ModuleList([
            CausalTransformerLayer(hidden_dim, heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.out_proj = MyLinear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_hops, T, feat_dim)
        Returns:
            (batch, T, out_dim)
        """
        batch, num_hops, T, feat_dim = x.shape

        # --- per-hop spatial encoding ---
        hop_outputs = []
        for i, mlp in enumerate(self.hop_mlps):
            h = x[:, i, :, :].reshape(batch * T, feat_dim)   # (batch*T, feat_dim)
            h = mlp(h)                                         # (batch*T, hidden_dim)
            h = h.reshape(batch, T, -1)                        # (batch, T, hidden_dim)
            hop_outputs.append(h)

        # mean pool over hops → (batch, T, hidden_dim)
        pooled = torch.stack(hop_outputs, dim=1).mean(dim=1)

        # add positional encoding
        pooled = pooled + _sinusoidal_pe(T, pooled.size(-1), pooled.device)

        # --- causal temporal modeling ---
        h = pooled
        for layer in self.transformer_layers:
            h = layer(h)

        return self.out_proj(h)                                # (batch, T, out_dim)
