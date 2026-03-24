import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsgnn.models.common import MyLinear

class PositionalEncoder(nn.Module):
    """
    Standard Positional Encoding for time series.
    """
    def __init__(self, units, len_ts, pos_rate=0.3):
        super().__init__()
        self.units = units
        self.pos_rate = pos_rate
        self.pos_encoding = nn.Parameter(
            PositionalEncoder.positional_encoding(len_ts, units),
            requires_grad=False
        )

    @classmethod
    def get_angles(cls, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def positional_encoding(cls, position, d_model):
        angle_rads = cls.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)

        # Apply sin to even indices in the array (2i)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array (2i+1)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding.astype(np.float32))

    def forward(self, inputs):
        # h = inputs * sqrt(units) * pos_rate + positional_encoding
        h = inputs * torch.sqrt(torch.tensor(self.units, dtype=torch.float32)) * self.pos_rate + self.pos_encoding
        return h

class Transformer(nn.Module):
    """
    Modular Transformer layer for multivariate time series.
    Assumes input shape: [Batch, Seq_Len, In_Channels]
    """
    def __init__(self, in_channels, units, len_ts,
                 heads=1,
                 drop_rate=0.0,
                 att_drop_rate=0.0,
                 residual=True,
                 fast_forward=False,
                 fast_forward_hidden_dim=None,
                 fast_forward_rate=4.0,
                 layer_norm=True):
        super().__init__()

        if units % heads != 0:
            raise ValueError(f"units ({units}) must be divisible by heads ({heads}).")

        self.units = units
        self.heads = heads
        self.residual = residual
        self.fast_forward = fast_forward
        self.layer_norm = layer_norm
        self.fast_forward_hidden_dim = fast_forward_hidden_dim

        self.dropout = nn.Dropout(drop_rate)
        self.att_dropout = nn.Dropout(att_drop_rate)

        self.dense_key = MyLinear(in_channels, units)
        self.dense_query = MyLinear(in_channels, units)
        self.dense_value = MyLinear(in_channels, units)

        if fast_forward:
            ff_hidden_dim = (
                int(units * fast_forward_rate)
                if fast_forward_hidden_dim is None
                else fast_forward_hidden_dim
            )
            self.ff = nn.Sequential(
                MyLinear(units, ff_hidden_dim),
                nn.ReLU(),
                MyLinear(ff_hidden_dim, units)
            )

        # Causal mask (tril)
        self.register_buffer("tri", torch.tril(torch.ones(len_ts, len_ts).bool()))

        if layer_norm:
            self.ln = nn.LayerNorm([len_ts, units])
            if fast_forward:
                self.ff_ln = nn.LayerNorm([len_ts, units])

    def forward(self, inputs):
        if isinstance(inputs, list):
            queries, keys = inputs
        else:
            queries = inputs
            keys = inputs

        Q = self.dense_query(queries)
        K = self.dense_key(keys)
        V = self.dense_value(keys)

        if self.heads > 1:
            # Concatenate heads along batch dimension as per reference
            Q_ = torch.concat(torch.split(Q, self.units // self.heads, dim=-1), dim=0)
            K_ = torch.concat(torch.split(K, self.units // self.heads, dim=-1), dim=0)
            V_ = torch.concat(torch.split(V, self.units // self.heads, dim=-1), dim=0)
        else:
            Q_ = Q
            K_ = K
            V_ = V

        # Scaled Dot-Product Attention
        sim_matrix = Q_ @ K_.transpose(-1, -2) / torch.sqrt(torch.tensor(Q_.size(-1), dtype=torch.float32))

        # Apply causal mask
        mask = self.tri.unsqueeze(dim=0).expand(sim_matrix.size(0), -1, -1)
        sim_matrix = torch.where(mask, sim_matrix, torch.tensor(-1e9, device=sim_matrix.device))
        sim_matrix = F.softmax(sim_matrix, dim=-1)
        sim_matrix = self.att_dropout(sim_matrix)

        outputs_ = sim_matrix @ V_

        if self.heads > 1:
            # Split heads back from batch dimension and concatenate along feature dimension
            outputs = torch.concat(torch.split(outputs_, outputs_.size(0) // self.heads, dim=0), dim=-1)
        else:
            outputs = outputs_

        outputs = self.dropout(outputs)

        if self.residual:
            outputs = outputs + queries
        if self.layer_norm:
            outputs = self.ln(outputs)

        if self.fast_forward:
            ff_outputs = self.ff(outputs)
            ff_outputs = self.dropout(ff_outputs)
            outputs = outputs + ff_outputs
            if self.layer_norm:
                outputs = self.ff_ln(outputs)

        return outputs
