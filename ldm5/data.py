"""Five-channel seismic dataset loader.

Loads matched .npy files from five channel folders (dn, gas, gr, vp, vs) and
stacks them into tensors of shape [5, H, W]. Each channel is normalized
independently using dataset-level mean/std statistics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Coerce a raw .npy array to shape [H, W] of float32."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            out = arr[0]
        elif arr.shape[-1] == 1:
            out = arr[..., 0]
        else:
            raise ValueError(f"Ambiguous 3D array shape {arr.shape}; expected single-channel.")
    else:
        raise ValueError(f"Unsupported array ndim={arr.ndim}, shape={arr.shape}.")
    return out.astype(np.float32, copy=False)


def _list_sample_ids(channel_dir: Path) -> List[str]:
    ids = []
    for name in os.listdir(channel_dir):
        if name.endswith(".npy"):
            ids.append(Path(name).stem)
    ids.sort()
    return ids


def discover_sample_ids(data_root: str | Path, channels: Sequence[str]) -> List[str]:
    """Return sample stems that exist in ALL channel folders."""
    data_root = Path(data_root)
    per_channel = []
    for c in channels:
        cdir = data_root / c
        if not cdir.is_dir():
            raise FileNotFoundError(f"Channel folder not found: {cdir}")
        per_channel.append(set(_list_sample_ids(cdir)))
    common = sorted(set.intersection(*per_channel))
    if not common:
        raise RuntimeError("No common sample IDs across channel folders.")
    return common


def compute_channel_stats(
    data_root: str | Path,
    channels: Sequence[str],
    sample_ids: Sequence[str] | None = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute per-channel mean and std over the dataset."""
    data_root = Path(data_root)
    if sample_ids is None:
        sample_ids = discover_sample_ids(data_root, channels)

    means = np.zeros(len(channels), dtype=np.float64)
    sqs = np.zeros(len(channels), dtype=np.float64)
    n = 0

    for sid in sample_ids:
        for ci, c in enumerate(channels):
            arr = _normalize_array(np.load(data_root / c / f"{sid}.npy"))
            means[ci] += arr.mean()
            sqs[ci] += (arr.astype(np.float64) ** 2).mean()
        n += 1

    mean = means / max(n, 1)
    var = sqs / max(n, 1) - mean ** 2
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)

    return {
        "channels": list(channels),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "num_samples": int(n),
    }


def load_or_compute_stats(
    data_root: str | Path,
    channels: Sequence[str],
    stats_path: str | Path,
) -> Dict:
    stats_path = Path(stats_path)
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if list(stats.get("channels", [])) == list(channels):
            return stats
    stats = compute_channel_stats(data_root, channels)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


class FiveChannelSeismicDataset(Dataset):
    """Returns tensors of shape [5, H, W], per-channel z-score normalized."""

    def __init__(
        self,
        data_root: str | Path,
        channels: Sequence[str] = ("dn", "gas", "gr", "vp", "vs"),
        stats: Dict | None = None,
        image_size: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.channels = list(channels)
        self.sample_ids = discover_sample_ids(self.data_root, self.channels)
        self.image_size = image_size

        if stats is None:
            stats = compute_channel_stats(self.data_root, self.channels, self.sample_ids)
        if list(stats["channels"]) != self.channels:
            raise ValueError("Provided stats channels do not match dataset channels.")

        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_stack(self, sid: str) -> torch.Tensor:
        arrs = []
        for c in self.channels:
            a = _normalize_array(np.load(self.data_root / c / f"{sid}.npy"))
            arrs.append(a)
        stacked = np.stack(arrs, axis=0)  # [5, H, W]
        return torch.from_numpy(stacked)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sid = self.sample_ids[idx]
        x = self._load_stack(sid)
        if self.image_size is not None and (x.shape[-1] != self.image_size or x.shape[-2] != self.image_size):
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        x = (x - self.mean) / self.std
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean


def build_dataset(data_cfg, stats_path: str | Path | None = None) -> Tuple[FiveChannelSeismicDataset, Dict]:
    stats_path = stats_path or data_cfg.stats_path
    stats = load_or_compute_stats(data_cfg.data_root, data_cfg.channels, stats_path)
    ds = FiveChannelSeismicDataset(
        data_root=data_cfg.data_root,
        channels=data_cfg.channels,
        stats=stats,
        image_size=data_cfg.image_size,
    )
    return ds, stats
