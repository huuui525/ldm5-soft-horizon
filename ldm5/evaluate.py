"""Evaluate trained AE and LDM checkpoints.

Reports:
- AE: per-channel L1, MSE, PSNR, SSIM on the training set (proxy validation).
- LDM: per-channel mean/std of generated samples vs real data, plus a small
  grid of (real / reconstructed / sampled) visualizations.

Usage:
    python -m ldm5.evaluate --ae_dir outputs/autoencoder/final \
        --unet_dir outputs/latent_diffusion/final --num_samples 16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import default_config
from .data import build_dataset


# ---------- metrics ----------

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


def ssim_2d(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> torch.Tensor:
    """Simple single-scale SSIM (no Gaussian window) per [B,1,H,W]."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu1 = pred.mean(dim=(-1, -2), keepdim=True)
    mu2 = target.mean(dim=(-1, -2), keepdim=True)
    sigma1 = ((pred - mu1) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma2 = ((target - mu2) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma12 = ((pred - mu1) * (target - mu2)).mean(dim=(-1, -2), keepdim=True)
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    return (num / den).mean()


# ---------- main ----------

def parse_args():
    cfg = default_config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=cfg.data.data_root)
    p.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    p.add_argument("--ae_dir", type=str, default="outputs/autoencoder/final")
    p.add_argument("--unet_dir", type=str, default="outputs/latent_diffusion/final")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_eval_batches", type=int, default=24,
                   help="How many batches to use for AE metrics.")
    p.add_argument("--num_samples", type=int, default=16,
                   help="How many synthetic samples to draw from LDM.")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="outputs/evaluation")
    return p.parse_args()


@torch.no_grad()
def evaluate_ae(ae, dataset, channels, mean, std, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sums = {k: torch.zeros(len(channels)) for k in ("l1", "mse", "psnr", "ssim")}
    n_batches = 0
    n_pixels = 0

    for i, batch in enumerate(tqdm(loader, desc="AE eval")):
        if i >= args.num_eval_batches:
            break
        x = batch.to(args.device)
        recon = ae.decode(ae.encode(x).latent_dist.mode()).sample
        # Per channel, both in z-normalized space.
        for c in range(len(channels)):
            p = recon[:, c:c + 1]
            t = x[:, c:c + 1]
            sums["l1"][c] += (p - t).abs().mean().cpu()
            sums["mse"][c] += ((p - t) ** 2).mean().cpu()
            # PSNR using empirical data range of the target channel in this batch.
            dr = (t.max() - t.min()).item()
            sums["psnr"][c] += psnr(p, t, max(dr, 1e-6)).cpu()
            sums["ssim"][c] += ssim_2d(p, t, max(dr, 1e-6)).cpu()
        n_batches += 1

    report = {
        "channels": list(channels),
        "num_batches": n_batches,
        "l1_per_channel": (sums["l1"] / n_batches).tolist(),
        "mse_per_channel": (sums["mse"] / n_batches).tolist(),
        "psnr_db_per_channel": (sums["psnr"] / n_batches).tolist(),
        "ssim_per_channel": (sums["ssim"] / n_batches).tolist(),
    }
    report["l1_mean"] = float(np.mean(report["l1_per_channel"]))
    report["psnr_db_mean"] = float(np.mean(report["psnr_db_per_channel"]))
    report["ssim_mean"] = float(np.mean(report["ssim_per_channel"]))
    return report


@torch.no_grad()
def evaluate_ldm(ae, unet, scheduler, dataset, channels, args):
    # Real statistics in normalized space (per channel mean/std).
    real_means = torch.zeros(len(channels))
    real_stds = torch.zeros(len(channels))
    n = 0
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        for c in range(len(channels)):
            real_means[c] += batch[:, c].mean()
            real_stds[c] += batch[:, c].std()
        n += 1
    real_means /= n
    real_stds /= n

    # Sample latents.
    scheduler.set_timesteps(args.num_inference_steps)
    sample_shape = (args.num_samples, unet.config.in_channels,
                    unet.config.sample_size, unet.config.sample_size)
    latents = torch.randn(sample_shape, device=args.device)
    for t in tqdm(scheduler.timesteps, desc="LDM sample"):
        pred = unet(latents, t).sample
        latents = scheduler.step(pred, t, latents).prev_sample

    images = ae.decode(latents).sample  # [N, 5, H, W] in normalized space.

    gen_means = images.mean(dim=(0, 2, 3)).cpu()
    gen_stds = images.std(dim=(0, 2, 3)).cpu()

    report = {
        "channels": list(channels),
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "real_mean_per_channel": real_means.tolist(),
        "real_std_per_channel": real_stds.tolist(),
        "gen_mean_per_channel": gen_means.tolist(),
        "gen_std_per_channel": gen_stds.tolist(),
        "mean_abs_diff": (gen_means - real_means).abs().mean().item(),
        "std_abs_diff": (gen_stds - real_stds).abs().mean().item(),
    }

    # Save raw normalized samples for inspection.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "ldm_samples_normalized.npy", images.cpu().numpy())
    return report


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    cfg.data.data_root = args.data_root
    dataset, stats = build_dataset(cfg.data, stats_path=Path(args.output_dir) / "norm_stats.json")
    channels = cfg.data.channels

    print(f"Dataset: {len(dataset)} samples, channels={channels}")
    print(f"Device: {args.device}")

    ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device).eval()
    ae_report = evaluate_ae(ae, dataset, channels,
                            stats["mean"], stats["std"], args)

    print("\n=== AutoencoderKL reconstruction ===")
    print(f"  mean L1   : {ae_report['l1_mean']:.4f}")
    print(f"  mean PSNR : {ae_report['psnr_db_mean']:.2f} dB")
    print(f"  mean SSIM : {ae_report['ssim_mean']:.4f}")
    print("  per-channel (l1 / psnr_dB / ssim):")
    for i, c in enumerate(channels):
        print(f"    {c:>4}: l1={ae_report['l1_per_channel'][i]:.4f}  "
              f"psnr={ae_report['psnr_db_per_channel'][i]:6.2f}  "
              f"ssim={ae_report['ssim_per_channel'][i]:.4f}")

    unet = UNet2DModel.from_pretrained(args.unet_dir).to(args.device).eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    ldm_report = evaluate_ldm(ae, unet, scheduler, dataset, channels, args)

    print("\n=== Latent DDPM samples vs real ===")
    print(f"  |mean diff| avg : {ldm_report['mean_abs_diff']:.4f}")
    print(f"  |std  diff| avg : {ldm_report['std_abs_diff']:.4f}")
    print("  per-channel (real_mu/gen_mu | real_sigma/gen_sigma):")
    for i, c in enumerate(channels):
        print(f"    {c:>4}: mu {ldm_report['real_mean_per_channel'][i]:+.3f} / "
              f"{ldm_report['gen_mean_per_channel'][i]:+.3f}    "
              f"sigma {ldm_report['real_std_per_channel'][i]:.3f} / "
              f"{ldm_report['gen_std_per_channel'][i]:.3f}")

    with open(save_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"ae": ae_report, "ldm": ldm_report}, f, indent=2)
    print(f"\nFull report written to {save_dir / 'report.json'}")


if __name__ == "__main__":
    main()
