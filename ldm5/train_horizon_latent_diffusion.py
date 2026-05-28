"""Train a latent DDPM U-Net with multi-scale horizon conditioning.

The AutoencoderKL is loaded and frozen. Horizon labels are extracted from the
real five-channel batch, resized to latent resolution, and injected into each
U-Net downsampling level through small convolutional adapters.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conditional_unet import MultiScaleHorizonUNet
from .config import default_config
from .data import build_dataset
from .horizon import horizon_condition, save_horizon_labels
from .logger import TrainLogger
from .models import build_scheduler


def parse_args():
    cfg = default_config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=cfg.data.data_root)
    p.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    p.add_argument("--ae_dir", type=str, required=True, help="Frozen AutoencoderKL directory.")
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--num_epochs", type=int, default=cfg.train.num_epochs)
    p.add_argument("--lr", type=float, default=cfg.train.learning_rate)
    p.add_argument("--num_workers", type=int, default=cfg.train.num_workers)
    p.add_argument("--save_every", type=int, default=cfg.train.save_every_epochs)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--condition_mode", choices=["soft", "label"], default="soft")
    p.add_argument("--label_quantile", type=float, default=0.85)
    p.add_argument("--condition_scale", type=float, default=1.0)
    p.add_argument("--save_horizon_labels", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run a single batch and exit.")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = default_config()
    cfg.data.data_root = args.data_root
    cfg.train.output_dir = args.output_dir
    use_amp = args.device.startswith("cuda") and not args.no_amp

    out_dir = Path(args.output_dir) / "horizon_latent_diffusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, stats = build_dataset(cfg.data, stats_path=Path(args.output_dir) / "norm_stats.json")
    if args.save_horizon_labels:
        save_horizon_labels(
            dataset,
            out_dir / f"horizon_{args.condition_mode}_labels",
            mode=args.condition_mode,
            quantile=args.label_quantile,
        )
    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        probe = dataset[0].unsqueeze(0).to(args.device)
        z = ae.encode(probe).latent_dist.sample()
        cfg.unet.sample_size = z.shape[-1]
        cfg.unet.in_channels = z.shape[1]
        cfg.unet.out_channels = z.shape[1]
        print(f"Latent shape: {tuple(z.shape)} -> using UNet sample_size={cfg.unet.sample_size}")

    unet = MultiScaleHorizonUNet(
        unet_config=dataclasses.asdict(cfg.unet),
        condition_scale=args.condition_scale,
    ).to(args.device)
    scheduler = build_scheduler(cfg.train)
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = TrainLogger(
        log_root=Path(args.output_dir) / "logs",
        tag="horizon_latent_diffusion",
        field_names=["loss"],
    )
    logger.info(
        f"device={args.device} amp={use_amp} batch_size={args.batch_size} "
        f"epochs={args.num_epochs} lr={args.lr} dataset_size={len(dataset)} "
        f"latent_shape={tuple(z.shape)} condition_mode={args.condition_mode}"
    )

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "unet": dataclasses.asdict(cfg.unet),
                "data": dataclasses.asdict(cfg.data),
                "train": dataclasses.asdict(cfg.train),
                "ae_dir": args.ae_dir,
                "condition_mode": args.condition_mode,
                "label_quantile": args.label_quantile,
                "condition_scale": args.condition_scale,
                "stats": stats,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    global_step = 0
    latent_size = (cfg.unet.sample_size, cfg.unet.sample_size)
    for epoch in range(args.num_epochs):
        unet.train()
        pbar = tqdm(loader, desc=f"Horizon LDM epoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(args.device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latents = ae.encode(batch).latent_dist.sample()
                horizon = horizon_condition(
                    batch,
                    latent_size=latent_size,
                    mode=args.condition_mode,
                    quantile=args.label_quantile,
                )

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = unet(noisy_latents, timesteps, horizon=horizon).sample
                if cfg.train.prediction_type == "epsilon":
                    target = noise
                elif cfg.train.prediction_type == "sample":
                    target = latents
                else:
                    raise ValueError(f"Unsupported prediction_type: {cfg.train.prediction_type}")
                loss = F.mse_loss(pred, target)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            global_step += 1

            metrics = {"loss": loss.item()}
            logger.log_step(
                epoch=epoch + 1,
                step=step + 1,
                global_step=global_step,
                lr=optim.param_groups[0]["lr"],
                metrics=metrics,
            )
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            if args.smoke:
                logger.info("Smoke test OK: one horizon-conditioned latent-DDPM step ran successfully.")
                logger.close()
                return

        logger.end_epoch(epoch + 1)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            ckpt_dir = out_dir / f"epoch_{epoch + 1:04d}"
            unet.save_pretrained(ckpt_dir)
            logger.info(f"Saved horizon-conditioned U-Net checkpoint to {ckpt_dir}")

    unet.save_pretrained(out_dir / "final")
    logger.info(f"Final horizon-conditioned U-Net saved to {out_dir / 'final'}")
    logger.close()


if __name__ == "__main__":
    main()
