"""损失函数: BCEWithLogitsLoss + DiceLoss 组合."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dims)
        card = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return (1 - dice).mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0, dice_weight: float = 1.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.dice_weight = dice_weight
        self._pos_weight = torch.tensor([pos_weight])
        self._bce = None
        self._bce_device = None
        self.dice = DiceLoss()

    def _get_bce(self, device: torch.device) -> nn.BCEWithLogitsLoss:
        if self._bce is None or self._bce_device != device:
            self._bce = nn.BCEWithLogitsLoss(
                pos_weight=self._pos_weight.to(device))
            self._bce_device = device
        return self._bce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self._get_bce(logits.device)
        return bce(logits, target) + self.dice_weight * self.dice(logits, target)