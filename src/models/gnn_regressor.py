import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.alpha)


def get_activation(name):
    if name == "prelu":
        return MyPReLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name is None or name == "identity":
        return nn.Identity()
    raise NotImplementedError(f"Activation {name} not implemented")


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        activation="prelu",
        drop_rate=0.0,
        output_activation=None,
        output_drop_rate=0.0,
    ):
        super().__init__()
        layers = []
        channels = [in_channels, *units_list]
        for i in range(len(channels) - 1):
            is_last = i == len(channels) - 2
            layers.append(MyLinear(channels[i], channels[i + 1]))
            layers.append(get_activation(output_activation if is_last else activation))
            layers.append(nn.Dropout(output_drop_rate if is_last else drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def _sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(T, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(T, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class CausalTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            MyLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            MyLinear(hidden_dim * 4, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + self.dropout(attn_out))
        return self.ln2(x + self.dropout(self.ff(x)))


class RpHGNNSpatialEncoder(nn.Module):
    """RpHGNN-style group encoder for precomputed relation context tensors."""

    def __init__(
        self,
        num_groups,
        group_size,
        feat_dim,
        hidden_dim,
        mlp_layers=2,
        conv_filters=2,
        dropout=0.1,
        merge_mode="concat",
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.merge_mode = merge_mode
        self.group_convs = nn.ModuleList(
            [nn.Conv1d(group_size, conv_filters, kernel_size=1) for _ in range(num_groups)]
        )
        for conv in self.group_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

        group_hidden = [hidden_dim] * max(mlp_layers, 1)
        self.group_mlps = nn.ModuleList(
            [
                MLP(
                    in_channels=conv_filters * feat_dim,
                    units_list=group_hidden,
                    activation="prelu",
                    drop_rate=dropout,
                    output_activation="prelu",
                    output_drop_rate=dropout,
                )
                for _ in range(num_groups)
            ]
        )

        if merge_mode == "concat":
            fusion_in = hidden_dim * num_groups
        elif merge_mode == "mean":
            fusion_in = hidden_dim
        else:
            raise ValueError(f"Unsupported merge_mode={merge_mode}")

        self.fusion = MLP(
            in_channels=fusion_in,
            units_list=[hidden_dim],
            activation="prelu",
            drop_rate=dropout,
            output_activation=None,
        )

    def forward(self, x):
        batch, T, num_groups, group_size, feat_dim = x.shape
        if num_groups != self.num_groups:
            raise ValueError(f"Expected {self.num_groups} groups, got {num_groups}")
        if group_size != self.group_size:
            raise ValueError(f"Expected group size {self.group_size}, got {group_size}")

        x = x.reshape(batch * T, num_groups, group_size, feat_dim)
        group_outputs = []
        for group_idx, (conv, mlp) in enumerate(zip(self.group_convs, self.group_mlps)):
            h = conv(x[:, group_idx])
            h = h.reshape(batch * T, -1)
            group_outputs.append(mlp(h))

        if self.merge_mode == "concat":
            fused = torch.cat(group_outputs, dim=-1)
        else:
            fused = torch.stack(group_outputs, dim=0).mean(dim=0)
        return self.fusion(fused).reshape(batch, T, -1)


class GNNRegressor(nn.Module):
    """
    RpHGNN-style spatial encoder plus causal temporal regressor.

    Input:  (batch, T, num_groups, group_size, feat_dim)
    Legacy input `(batch, T, num_graphs, num_hops, feat_dim)` is also accepted;
    in the new loader, graph/relation groups are already RpHGNN groups.
    Output: (batch, T, out_dim)
    """

    def __init__(
        self,
        num_hops: int,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        mlp_layers: int = 2,
        num_transformer_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        num_graphs: int = 1,
        conv_filters: int = 2,
        merge_mode: str = "concat",
    ):
        super().__init__()
        self.num_groups = num_graphs
        self.group_size = num_hops
        self.spatial_encoder = RpHGNNSpatialEncoder(
            num_groups=num_graphs,
            group_size=num_hops,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            mlp_layers=mlp_layers,
            conv_filters=conv_filters,
            dropout=dropout,
            merge_mode=merge_mode,
        )
        self.transformer_layers = nn.ModuleList(
            [CausalTransformerLayer(hidden_dim, heads, dropout) for _ in range(num_transformer_layers)]
        )
        self.out_proj = MyLinear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input with 5 dims (batch, T, groups, group_size, feat_dim), got {x.shape}")

        h = self.spatial_encoder(x)
        h = h + _sinusoidal_pe(h.size(1), h.size(-1), h.device)
        for layer in self.transformer_layers:
            h = layer(h)
        return self.out_proj(h)
