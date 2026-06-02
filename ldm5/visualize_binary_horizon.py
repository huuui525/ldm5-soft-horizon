"""Generate and visualize strict 0/1 horizon label maps.

This utility is independent of model training. It extracts binary horizon
labels from real five-channel samples and saves both raw arrays and PNG
figures in a separate output directory.

Usage:
    python -m ldm5.visualize_binary_horizon --data_root "标签" --num_samples 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import default_config
from .data import build_dataset
from .horizon import (
    combined_horizon_label_maps,
    horizon_label_map,
    per_channel_horizon_label_maps,
)
from .visualize import grid_figure, imshow_panel


def parse_args() -> argparse.Namespace:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="Generate strict 0/1 binary horizon label maps and visualizations."
    )
    parser.add_argument("--data_root", type=str, default=cfg.data.data_root)
    parser.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    parser.add_argument("--save_dir", type=str, default="outputs_binary_horizon")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label_quantile", type=float, default=0.75)
    parser.add_argument(
        "--mode",
        choices=["label", "per_channel_label", "combined_label"],
        default="combined_label",
        help="label=shared only, per_channel_label=5 channels, combined_label=shared+5 channels.",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=56,
        help="Latent-resolution size for nearest-neighbor downsampled labels.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap for binary maps. Use 'jet' if you want the same style as model figures.",
    )
    return parser.parse_args()


def label_channel_names(mode: str, data_channels) -> list[str]:
    if mode == "label":
        return ["shared"]
    if mode == "per_channel_label":
        return list(data_channels)
    return ["shared", *list(data_channels)]


def extract_binary_labels(x: torch.Tensor, mode: str, quantile: float) -> torch.Tensor:
    if mode == "label":
        return horizon_label_map(x, quantile=quantile)
    if mode == "per_channel_label":
        return per_channel_horizon_label_maps(x, quantile=quantile)
    if mode == "combined_label":
        return combined_horizon_label_maps(x, quantile=quantile)
    raise ValueError(f"Unsupported binary horizon mode: {mode}")


def downsample_binary_nearest(labels: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(labels, size=(size, size), mode="nearest")
    return (resized > 0.5).to(dtype=labels.dtype)


def binary_condition_figure(
    labels: np.ndarray,
    channel_names: list[str],
    row_labels: list[str],
    save_path: Path,
    title: str,
    cmap: str,
) -> None:
    rows, cols = labels.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * rows), squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            imshow_panel(ax, labels[r, c], channel_names[c] if r == 0 else "", cmap)
            ax.images[-1].set_clim(0, 1)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    original_dir = save_dir / f"{args.mode}_original"
    latent_dir = save_dir / f"{args.mode}_latent{args.latent_size}"
    original_dir.mkdir(exist_ok=True)
    latent_dir.mkdir(exist_ok=True)

    cfg = default_config()
    cfg.data.data_root = args.data_root
    dataset, _stats = build_dataset(cfg.data, stats_path=Path(args.output_dir) / "norm_stats.json")
    channels = cfg.data.channels

    idxs = np.random.choice(len(dataset), size=args.num_samples, replace=False)
    real_norm = torch.stack([dataset[int(i)] for i in idxs])
    real_orig = dataset.denormalize(real_norm)

    labels_original = extract_binary_labels(real_norm, args.mode, args.label_quantile)
    labels_latent = downsample_binary_nearest(labels_original, args.latent_size)

    labels_original_np = labels_original.cpu().numpy().astype(np.float32)
    labels_latent_np = labels_latent.cpu().numpy().astype(np.float32)
    channel_names = label_channel_names(args.mode, channels)
    row_labels = [f"#{int(i)}" for i in idxs]

    for offset, sample_index in enumerate(idxs):
        sid = f"{int(sample_index):06d}"
        np.save(original_dir / f"{sid}.npy", labels_original_np[offset])
        np.save(latent_dir / f"{sid}.npy", labels_latent_np[offset])

    np.savez(
        save_dir / f"{args.mode}_binary_horizon_arrays.npz",
        real=real_orig.cpu().numpy(),
        labels_original=labels_original_np,
        labels_latent=labels_latent_np,
        channels=np.array(channels),
        label_channels=np.array(channel_names),
        indices=np.array(idxs),
        mode=np.array(args.mode),
        label_quantile=np.array(args.label_quantile),
        latent_size=np.array(args.latent_size),
    )

    grid_figure(
        real_orig.cpu().numpy(),
        channels=channels,
        row_labels=[f"#{int(i)} real" for i in idxs],
        save_path=save_dir / "real_samples.png",
        suptitle="Real samples used to extract binary horizon labels",
        cmap="jet",
    )
    binary_condition_figure(
        labels_original_np,
        channel_names=channel_names,
        row_labels=row_labels,
        save_path=save_dir / f"{args.mode}_original_binary.png",
        title=f"Strict 0/1 horizon labels at original resolution (q={args.label_quantile})",
        cmap=args.cmap,
    )
    binary_condition_figure(
        labels_latent_np,
        channel_names=channel_names,
        row_labels=row_labels,
        save_path=save_dir / f"{args.mode}_latent{args.latent_size}_binary.png",
        title=f"Strict 0/1 horizon labels at latent resolution (nearest, {args.latent_size}x{args.latent_size})",
        cmap=args.cmap,
    )

    print(f"Saved original labels to {original_dir}")
    print(f"Saved latent labels to {latent_dir}")
    print(f"Saved arrays to {save_dir / f'{args.mode}_binary_horizon_arrays.npz'}")


if __name__ == "__main__":
    main()
