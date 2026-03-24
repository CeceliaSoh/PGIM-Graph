import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, pred, target, mask):
        """
        pred:   [B, ...]
        target: [B, ...]
        mask:   [B, ...] (0 or 1)

        returns scalar loss
        """
        mask = mask.float()

        loss = (pred - target) ** 2
        loss = loss * mask

        return loss.sum() / (mask.sum() + self.eps)