"""
utils.py
--------
Helper functions for:
  - Gaussian heatmap generation
  - Seed setting
  - Checkpoint saving / loading
  - Parameter counting / model size
"""

import random
import os
from typing import Optional, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------
#  Gaussian heatmap
# ------------------------------------------------------------------
def generate_gaussian_heatmap(
    keypoints: np.ndarray,      # (K, 2)   keypoint coordinates (pixels)
    visibility: np.ndarray,     # (K,)     0=invisible, 1=occluded, 2=visible
    heatmap_size: int = 56,
    img_size: int = 224,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Create a stack of Gaussian heatmaps (one per keypoint).

    Returns:
        heatmaps: np.ndarray of shape (K, H, W)
    """
    K = keypoints.shape[0]
    heatmaps = np.zeros((K, heatmap_size, heatmap_size), dtype=np.float32)

    scale = heatmap_size / img_size

    # Build coordinate grids (heatmap-space)
    ys, xs = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing='ij',
    )

    for k in range(K):
        if visibility[k] == 0:          # keypoint missing
            continue

        # Scale keypoint coordinates to heatmap size
        cx = keypoints[k, 0] * scale
        cy = keypoints[k, 1] * scale

        # 2-D Gaussian
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        heatmaps[k] = g

    return heatmaps


# ------------------------------------------------------------------
#  Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
#  Checkpoint handling
# ------------------------------------------------------------------
def save_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
):
    """Save a checkpoint (always saves latest, optionally saves best)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    """
    Load checkpoint.
    Returns (start_epoch, best_loss).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))
    return epoch, best_loss


# ------------------------------------------------------------------
#  Model information
# ------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """Return model size in MB (parameters only)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)