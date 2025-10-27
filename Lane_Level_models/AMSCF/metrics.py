# -*- coding: utf-8 -*-
# metrics.py — 指标（数值稳定）
import torch


def masked_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - true))


def masked_rmse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - true) ** 2))


def masked_mape(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # 防 0 与爆炸；控制在 [0, 1000]%
    denom = torch.clamp(torch.abs(true), min=1e-3)
    mape = torch.mean(torch.abs(pred - true) / denom) * 100.0
    return torch.clamp(mape, 0.0, 1000.0)
