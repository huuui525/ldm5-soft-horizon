"""Horizon-prior extraction utilities for conditional latent diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def soft_horizon_map(x: torch.Tensor) -> torch.Tensor:
    """Extract a one-channel soft horizon map from a five-channel batch.

    Input shape is [B, C, H, W]. The output is [B, 1, H, W]. We emphasize
    vertical/depth changes because near-horizontal reflector boundaries appear
    primarily as depth-axis jumps shared by the physical channels.
    """
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs()
    vertical = F.pad(vertical, (0, 0, 1, 0))
    horizon = vertical.mean(dim=1, keepdim=True)
    return normalize_map(horizon)


def per_channel_horizon_maps(x: torch.Tensor) -> torch.Tensor:
    """Extract one soft horizon map per physical channel.

    Input shape is [B, C, H, W], output shape is [B, C, H, W]. Unlike
    `soft_horizon_map`, this keeps channel-specific boundary responses so the
    conditional U-Net can distinguish dn/gas/gr/vp/vs structural differences.
    """
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs()
    vertical = F.pad(vertical, (0, 0, 1, 0))
    return normalize_map(vertical)


def combined_horizon_maps(x: torch.Tensor) -> torch.Tensor:
    """Return shared + per-channel soft horizon maps, shape [B, 1 + C, H, W]."""
    return torch.cat([soft_horizon_map(x), per_channel_horizon_maps(x)], dim=1)


def horizon_label_map(x: torch.Tensor, quantile: float = 0.75) -> torch.Tensor:
    """Convert the soft map into a binary label map by per-sample quantile."""
    soft = soft_horizon_map(x)
    flat = soft.flatten(start_dim=1)
    threshold = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
    return (soft >= threshold).to(dtype=x.dtype)


def per_channel_horizon_label_maps(x: torch.Tensor, quantile: float = 0.75) -> torch.Tensor:
    """Convert per-channel soft horizon maps into per-channel binary labels."""
    soft = per_channel_horizon_maps(x)
    flat = soft.flatten(start_dim=2)
    threshold = torch.quantile(flat, quantile, dim=2).view(x.shape[0], x.shape[1], 1, 1)
    return (soft >= threshold).to(dtype=x.dtype)


def combined_horizon_label_maps(x: torch.Tensor, quantile: float = 0.75) -> torch.Tensor:
    """Return shared + per-channel binary horizon labels."""
    return torch.cat(
        [
            horizon_label_map(x, quantile=quantile),
            per_channel_horizon_label_maps(x, quantile=quantile),
        ],
        dim=1,
    )


def horizon_condition(
    x: torch.Tensor,
    latent_size: int | tuple[int, int],
    mode: str = "combined",
    quantile: float = 0.75,
) -> torch.Tensor:
    """Return the latent-resolution horizon condition used by the U-Net."""
    if mode in ("soft", "shared"):
        cond = soft_horizon_map(x)
    elif mode in ("label", "shared_label"):
        cond = horizon_label_map(x, quantile=quantile)
    elif mode in ("per_channel", "per_channel_soft"):
        cond = per_channel_horizon_maps(x)
    elif mode == "per_channel_label":
        cond = per_channel_horizon_label_maps(x, quantile=quantile)
    elif mode in ("combined", "combined_soft"):
        cond = combined_horizon_maps(x)
    elif mode == "combined_label":
        cond = combined_horizon_label_maps(x, quantile=quantile)
    else:
        raise ValueError(f"Unsupported horizon mode: {mode}")
    if isinstance(latent_size, int):
        size = (latent_size, latent_size)
    else:
        size = tuple(latent_size)
    return F.interpolate(cond, size=size, mode="bilinear", align_corners=False)


def save_horizon_labels(
    dataset,
    save_dir: str | Path,
    mode: str = "combined",
    quantile: float = 0.75,
    sample_ids: Iterable[str] | None = None,
) -> None:
    """Materialize extracted horizon maps as .npy files for inspection/reuse."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ids = list(sample_ids) if sample_ids is not None else list(dataset.sample_ids)
    id_to_index = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    for sid in ids:
        x = dataset[id_to_index[sid]].unsqueeze(0)
        if mode in ("soft", "shared"):
            horizon = soft_horizon_map(x)
        elif mode in ("label", "shared_label"):
            horizon = horizon_label_map(x, quantile=quantile)
        elif mode in ("per_channel", "per_channel_soft"):
            horizon = per_channel_horizon_maps(x)
        elif mode == "per_channel_label":
            horizon = per_channel_horizon_label_maps(x, quantile=quantile)
        elif mode in ("combined", "combined_soft"):
            horizon = combined_horizon_maps(x)
        elif mode == "combined_label":
            horizon = combined_horizon_label_maps(x, quantile=quantile)
        else:
            raise ValueError(f"Unsupported horizon mode: {mode}")
        np.save(save_dir / f"{sid}.npy", horizon.squeeze(0).numpy().astype(np.float32))
