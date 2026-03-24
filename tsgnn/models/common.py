import torch
import torch.nn as nn

class MyLinear(nn.Linear):
    """
    Linear layer with Xavier Normal initialization.
    """
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class MyConv1d(nn.Conv1d):
    """
    1D Convolutional layer with Xavier Normal initialization.
    """
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class Residual(nn.Module):
    """
    Simple residual connection wrapper.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return x + self.model(x)

class Lambda(nn.Module):
    """
    Wrapper for using arbitrary functions as nn.Modules.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
