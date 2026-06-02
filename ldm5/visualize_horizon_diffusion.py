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
    p.add_argument("--condition_mode", choices=["soft", "label"], default="soft")
    p.add_argument("--label_quantile", type=float, default=0.85)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cmap", type=str, default="jet")
    return p.parse_args()


def condition_figure(conditions: np.ndarray, row_labels, save_path: Path) -> None:
    rows = conditions.shape[0]
    fig, axes = plt.subplots(rows, 1, figsize=(2.6, 2.2 * rows), squeeze=False)
    for r in range(rows):
        imshow_panel(axes[r, 0], conditions[r, 0], "horizon" if r == 0 else "", "magma")
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

    ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device).eval()
    unet = MultiScaleHorizonUNet.from_pretrained(args.unet_dir, map_location=args.device).to(args.device).eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

    latent_size = unet.config.sample_size
    cond = horizon_condition(
        real_norm.to(args.device),
        latent_size=latent_size,
        mode=args.condition_mode,
        quantile=args.label_quantile,
    )

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
    )
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
