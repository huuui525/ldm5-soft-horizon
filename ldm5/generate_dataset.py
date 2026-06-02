"""Generate a synthetic five-channel seismic dataset from pure latent noise.

Usage:
    python -m ldm5.generate_dataset --num_samples 32 --num_inference_steps 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DModel,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from .config import default_config
from .data import load_or_compute_stats
from .visualize import grid_figure


def parse_args() -> argparse.Namespace:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="Generate a new five-channel seismic dataset from pure latent noise."
    )
    parser.add_argument("--data_root", type=str, default=cfg.data.data_root)
    parser.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--ae_dir", type=str, default="outputs/autoencoder/final")
    parser.add_argument("--unet_dir", type=str, default="outputs/latent_diffusion/final")
    parser.add_argument("--save_dir", type=str, default="outputs/generated_dataset")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "dpm"],
        help="采样器类型：ddpm=DDPM随机采样；ddim=DDIM确定性/半随机；dpm=DPMSolverMultistep",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="仅在 --scheduler=ddim 时生效；0=确定性 DDIM，1=近似 DDPM 随机性",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--num_preview", type=int, default=8)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--save_normalized", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be positive.")
    if args.num_preview < 0:
        raise ValueError("--num_preview must be zero or positive.")


def latent_spatial_size(unet: UNet2DModel) -> tuple[int, int]:
    sample_size = unet.config.sample_size
    if isinstance(sample_size, int):
        return sample_size, sample_size
    if len(sample_size) != 2:
        raise ValueError(f"Unsupported UNet sample_size: {sample_size}")
    return int(sample_size[0]), int(sample_size[1])


def make_generator(device: str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def build_scheduler(
    name: str,
    num_train_timesteps: int,
    prediction_type: str,
) -> SchedulerMixin:
    if name == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )
    if name == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )
    if name == "dpm":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def generate_dataset_from_noise(
    ae: AutoencoderKL,
    unet: UNet2DModel,
    scheduler: SchedulerMixin,
    scheduler_name: str,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    device: str,
    seed: int,
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    """Return generated samples in normalized five-channel space, shape [N, 5, H, W]."""
    generator = make_generator(device, seed)
    latent_height, latent_width = latent_spatial_size(unet)
    generated_batches: list[torch.Tensor] = []

    sample_ranges = range(0, num_samples, batch_size)
    for batch_start in tqdm(sample_ranges, desc="Generate dataset"):
        current_batch = min(batch_size, num_samples - batch_start)
        scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(
            (
                current_batch,
                unet.config.in_channels,
                latent_height,
                latent_width,
            ),
            device=device,
            generator=generator,
        )
        for timestep in tqdm(scheduler.timesteps, desc=f"{scheduler_name} sample", leave=False):
            predicted_noise = unet(latents, timestep).sample
            if scheduler_name == "ddim":
                latents = scheduler.step(
                    predicted_noise, timestep, latents, eta=ddim_eta, generator=generator
                ).prev_sample
            else:
                latents = scheduler.step(predicted_noise, timestep, latents).prev_sample
        generated = ae.decode(latents).sample.cpu()
        generated_batches.append(generated)

    return torch.cat(generated_batches, dim=0)


def denormalize(samples: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], dtype=samples.dtype).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], dtype=samples.dtype).view(1, -1, 1, 1)
    return samples * std + mean


def sample_id(prefix: str, sample_index: int) -> str:
    return f"{prefix}{sample_index:06d}"


def save_channel_dataset(
    samples_original: np.ndarray,
    samples_normalized: np.ndarray,
    channels: Sequence[str],
    save_dir: Path,
    prefix: str,
    start_index: int,
    save_normalized: bool,
) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = save_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    channel_dirs = {channel_name: save_dir / channel_name for channel_name in channels}
    for channel_dir in channel_dirs.values():
        channel_dir.mkdir(exist_ok=True)

    normalized_dir = save_dir / "normalized_combined"
    if save_normalized:
        normalized_dir.mkdir(exist_ok=True)

    sample_ids: list[str] = []
    for offset, sample_original in enumerate(samples_original):
        current_id = sample_id(prefix, start_index + offset)
        sample_ids.append(current_id)
        np.save(combined_dir / f"{current_id}.npy", sample_original.astype(np.float32, copy=False))
        if save_normalized:
            np.save(
                normalized_dir / f"{current_id}.npy",
                samples_normalized[offset].astype(np.float32, copy=False),
            )
        for channel_index, channel_name in enumerate(channels):
            np.save(
                channel_dirs[channel_name] / f"{current_id}.npy",
                sample_original[channel_index].astype(np.float32, copy=False),
            )

    np.save(save_dir / "samples_original.npy", samples_original.astype(np.float32, copy=False))
    np.save(save_dir / "samples_normalized.npy", samples_normalized.astype(np.float32, copy=False))
    return {
        "sample_ids": sample_ids,
        "combined_dir": str(combined_dir),
        "channel_dirs": {name: str(path) for name, path in channel_dirs.items()},
        "samples_original": str(save_dir / "samples_original.npy"),
        "samples_normalized": str(save_dir / "samples_normalized.npy"),
    }


def save_preview_grid(
    samples_original: np.ndarray,
    channels: Sequence[str],
    save_dir: Path,
    sample_ids: Sequence[str],
    num_preview: int,
    cmap: str,
    scheduler_name: str = "ddpm",
    num_inference_steps: int = 200,
) -> str | None:
    preview_count = min(num_preview, len(samples_original))
    if preview_count == 0:
        return None
    preview_path = save_dir / f"gen_{scheduler_name}_grid.png"
    short = scheduler_name[:3]
    row_labels = [f"{short}{i:02d}" for i in range(preview_count)]
    grid_figure(
        samples_original[:preview_count],
        channels=channels,
        row_labels=row_labels,
        save_path=preview_path,
        suptitle=f"{scheduler_name} ({num_inference_steps} steps)",
        cmap=cmap,
    )
    return str(preview_path)


def write_metadata(
    save_dir: Path,
    args: argparse.Namespace,
    channels: Sequence[str],
    stats: dict,
    dataset_paths: dict,
    preview_path: str | None,
    sample_shape: Sequence[int],
) -> None:
    metadata = {
        "source": "pure_latent_noise",
        "ae_dir": args.ae_dir,
        "unet_dir": args.unet_dir,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "num_train_timesteps": args.num_train_timesteps,
        "prediction_type": args.prediction_type,
        "scheduler": args.scheduler,
        "ddim_eta": args.ddim_eta if args.scheduler == "ddim" else None,
        "seed": args.seed,
        "channels": list(channels),
        "sample_shape": list(sample_shape),
        "stats": stats,
        "paths": dataset_paths,
        "preview_path": preview_path,
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = default_config()
    channels = cfg.data.channels
    stats_path = Path(args.stats_path) if args.stats_path else Path(args.output_dir) / "norm_stats.json"
    stats = load_or_compute_stats(args.data_root, channels, stats_path)

    print(f"Device: {args.device}")
    print(f"Channels: {channels}")
    print(f"Stats: {stats_path}")

    ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device).eval()
    unet = UNet2DModel.from_pretrained(args.unet_dir).to(args.device).eval()
    scheduler = build_scheduler(
        args.scheduler,
        num_train_timesteps=args.num_train_timesteps,
        prediction_type=args.prediction_type,
    )
    print(f"Scheduler: {args.scheduler} (steps={args.num_inference_steps})")

    generated_normalized = generate_dataset_from_noise(
        ae=ae,
        unet=unet,
        scheduler=scheduler,
        scheduler_name=args.scheduler,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        seed=args.seed,
        ddim_eta=args.ddim_eta,
    )
    generated_original = denormalize(generated_normalized, stats)

    save_dir = Path(args.save_dir)
    samples_original = generated_original.numpy()
    samples_normalized = generated_normalized.numpy()
    dataset_paths = save_channel_dataset(
        samples_original=samples_original,
        samples_normalized=samples_normalized,
        channels=channels,
        save_dir=save_dir,
        prefix=args.prefix,
        start_index=args.start_index,
        save_normalized=args.save_normalized,
    )
    preview_path = save_preview_grid(
        samples_original=samples_original,
        channels=channels,
        save_dir=save_dir,
        sample_ids=dataset_paths["sample_ids"],
        num_preview=args.num_preview,
        cmap=args.cmap,
        scheduler_name=args.scheduler,
        num_inference_steps=args.num_inference_steps,
    )
    write_metadata(
        save_dir=save_dir,
        args=args,
        channels=channels,
        stats=stats,
        dataset_paths=dataset_paths,
        preview_path=preview_path,
        sample_shape=samples_original.shape[1:],
    )

    print(f"Generated dataset written to {save_dir}")
    print(f"Combined samples: {save_dir / 'combined'}")
    print(f"Channel folders: {', '.join(channels)}")
    if preview_path is not None:
        print(f"Preview figure: {preview_path}")


if __name__ == "__main__":
    main()
