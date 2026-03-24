import torch
import torch.nn as nn
from tsgnn.models.transformer import Transformer, PositionalEncoder
from tsgnn.models.common import MyLinear

class TransformerRegressor(nn.Module):
    """
    Transformer-based model for time series regression (e.g., next-step prediction).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        units,
        ff_hidden_dim,
        len_ts,
        num_layers=2,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()
        
        # Project input channels to hidden units
        self.input_projection = MyLinear(in_channels, units)
        self.pos_encoder = PositionalEncoder(units, len_ts)
        
        # Stacked Transformer layers
        self.transformer_layers = nn.ModuleList([
            Transformer(
                in_channels=units, 
                units=units, 
                len_ts=len_ts, 
                heads=heads, 
                drop_rate=dropout, 
                fast_forward=True,
                fast_forward_hidden_dim=ff_hidden_dim,
            ) for _ in range(num_layers)
        ])
        
        # Final regression head
        self.output_projection = MyLinear(units, out_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, In_Channels]
        Returns:
            Predictions of shape [Batch, Seq_Len, Out_Channels]
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
            
        x = self.output_projection(x)
        return x
