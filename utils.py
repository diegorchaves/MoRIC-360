"""Utility functions for the training pipeline."""

import os
import random
import numpy as np
import torch


def seed_everything(seed=1029):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
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
    return 10 * np.log10(max_val ** 2 / np.asarray(loss))
