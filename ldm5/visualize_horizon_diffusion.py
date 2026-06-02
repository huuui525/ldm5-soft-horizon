"""Visualize horizon-conditioned latent diffusion samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler

from .conditional_unet import MultiScaleHorizonUNet
from .config import default_config
from .data import build_dataset
from .horizon import horizon_condition
from .visualize import grid_figure, imshow_panel


def parse_args():
    cfg = default_config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=cfg.data.data_root)
    p.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    p.add_argument("--ae_dir", type=str, default="outputs/autoencoder/final")
    p.add_argument("--unet_dir", type=str, default="outputs/horizon_latent_diffusion/final")
    p.add_argument("--save_dir", type=str, default="outputs/horizon_evaluation")
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument(
        "--condition_mode",
        choices=[
            "soft",
            "label",
            "per_channel",
            "per_channel_label",
            "combined",
            "combined_label",
        ],
        default="combined",
    )
    p.add_argument("--label_quantile", type=float, default=0.75)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument(
        "--conditions_only",
        action="store_true",
        help="Only visualize extracted horizon conditions; does not load AE or U-Net.",
    )
    return p.parse_args()


def condition_figure(conditions: np.ndarray, row_labels, save_path: Path, cmap: str) -> None:
    rows, cols = conditions.shape[:2]
    if cols == 6:
        channel_labels = ["shared", "dn", "gas", "gr", "vp", "vs"]
    elif cols == 5:
        channel_labels = ["dn", "gas", "gr", "vp", "vs"]
    else:
        channel_labels = [f"cond {i + 1}" for i in range(cols)]
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * rows), squeeze=False)
    for r in range(rows):
        for c in range(cols):
            imshow_panel(axes[r, c], conditions[r, c], channel_labels[c] if r == 0 else "", cmap)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=9)
    fig.suptitle("Condition horizon maps", fontsize=11)
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

    cfg = default_config()
    cfg.data.data_root = args.data_root
    dataset, _stats = build_dataset(cfg.data, stats_path=Path(args.output_dir) / "norm_stats.json")
    channels = cfg.data.channels

    idxs = np.random.choice(len(dataset), size=args.num_samples, replace=False)
    real_norm = torch.stack([dataset[int(i)] for i in idxs])
    real_orig = dataset.denormalize(real_norm)

    latent_size = 56
    cond = horizon_condition(
        real_norm.to(args.device),
        latent_size=latent_size,
        mode=args.condition_mode,
        quantile=args.label_quantile,
    )

    grid_figure(
        real_orig.cpu().numpy(),
        channels=channels,
        row_labels=[f"#{int(i)} real" for i in idxs],
        save_path=save_dir / "condition_real_samples.png",
        suptitle="Real samples used to extract horizon conditions",
        cmap=args.cmap,
    )
    condition_figure(
        cond.cpu().numpy(),
        row_labels=[f"#{int(i)} cond" for i in idxs],
        save_path=save_dir / "horizon_conditions.png",
        cmap=args.cmap,
    )
    if args.conditions_only:
        np.savez(
            save_dir / "horizon_condition_arrays.npz",
            real=real_orig.cpu().numpy(),
            condition=cond.cpu().numpy(),
            channels=np.array(channels),
            condition_mode=np.array(args.condition_mode),
            indices=np.array(idxs),
        )
        print(f"Saved raw condition arrays to {save_dir / 'horizon_condition_arrays.npz'}")
        return

    ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device).eval()
    unet = MultiScaleHorizonUNet.from_pretrained(args.unet_dir, map_location=args.device).to(args.device).eval()
    if unet.condition_channels != cond.shape[1]:
        raise ValueError(
            f"Checkpoint expects {unet.condition_channels} condition channels, "
            f"but condition_mode={args.condition_mode!r} produced {cond.shape[1]}. "
            "Use a checkpoint trained with the same condition mode."
        )
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

    scheduler.set_timesteps(args.num_inference_steps)
    shape = (
        args.num_samples,
        unet.config.in_channels,
        unet.config.sample_size,
        unet.config.sample_size,
    )
    latents = torch.randn(shape, device=args.device)
    for timestep in scheduler.timesteps:
        pred = unet(latents, timestep, horizon=cond).sample
        latents = scheduler.step(pred, timestep, latents).prev_sample
    gen_norm = ae.decode(latents).sample.cpu()
    gen_orig = dataset.denormalize(gen_norm)

    grid_figure(
        gen_orig.numpy(),
        channels=channels,
        row_labels=[f"#{int(i)} gen" for i in idxs],
        save_path=save_dir / "horizon_conditioned_samples.png",
        suptitle=f"Horizon-conditioned LDM samples (steps={args.num_inference_steps})",
        cmap=args.cmap,
    )

    np.savez(
        save_dir / "horizon_visualize_arrays.npz",
        real=real_orig.cpu().numpy(),
        generated=gen_orig.numpy(),
        condition=cond.cpu().numpy(),
        channels=np.array(channels),
        indices=np.array(idxs),
    )
    print(f"Saved raw arrays to {save_dir / 'horizon_visualize_arrays.npz'}")


if __name__ == "__main__":
    main()
