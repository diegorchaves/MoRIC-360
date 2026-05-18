"""Utility functions for the training pipeline."""

import os
import random
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed=1029):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_mgrid(w_sidelen, h_sidelen, dim=2):
    """
    Generate a meshgrid of normalized coordinates.

    Args:
        w_sidelen: width of the grid
        h_sidelen: height of the grid
        dim: dimensionality (default 2)

    Returns:
        Normalized coordinate grid of shape (1, dim, h_sidelen, w_sidelen)
    """
    x = torch.linspace(-1, 1, steps=w_sidelen)
    y = torch.linspace(-1, 1, steps=h_sidelen)
    tensors = (x, y) if dim == 2 else (x,) * dim

    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.unsqueeze(0).permute(0, 3, 2, 1)

    return mgrid


def make_path(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    return 0


def loss_to_psnr(loss, max_val=1):
    """Convert MSE loss to PSNR in dB."""
    return 10 * np.log10(max_val**2 / np.asarray(loss))


def ws_mse_loss(x: Tensor, y: Tensor) -> Tensor:

    def __weights(height, width):
        phis = np.arange(height + 1) * np.pi / height
        deltaTheta = 2 * np.pi / width
        column = np.asarray(
            [
                deltaTheta * (-np.cos(phis[j + 1]) + np.cos(phis[j]))
                for j in range(height)
            ]
        )
        return np.repeat(column[:, np.newaxis], width, 1)

    height, width = x.shape[-2], x.shape[-1]
    w = __weights(height, width)
    w = torch.from_numpy(w).to(x.device, dtype=x.dtype)
    w = w.unsqueeze(0).unsqueeze(0)
    mse = torch.mean(((x - y) ** 2 * w), axis=1)
    wsmse = torch.sum(mse) / (4 * np.pi)

    return wsmse


def ws_mse_loss_flat(pred: Tensor, target: Tensor, H: int, W: int) -> Tensor:
    """
    Wrapper para ws_mse_loss que aceita tensores no formato flat (B, H*W, 3).
    Faz reshape para (B, 3, H, W), chama ws_mse_loss, retorna escalar.
    """
    # (B, H*W, 3) → (B, 3, H, W)
    pred_2d = pred.permute(0, 2, 1).view(-1, 3, H, W)
    target_2d = target.permute(0, 2, 1).view(-1, 3, H, W)
    return ws_mse_loss(pred_2d, target_2d)
