"""Stage 1: Pretrain the five-channel `AutoencoderKL`.

Reconstructs five-channel seismic tensors. Uses L1 + optional gradient /
channel-correlation / frequency losses. Saves checkpoints under
`<output_dir>/autoencoder/`.

Usage (Windows PowerShell):
    python -m ldm5.train_autoencoder --data_root "标签/标签" --num_epochs 50
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import default_config
from .data import build_dataset
from .logger import TrainLogger
from .losses import autoencoder_loss
from .models import build_autoencoder


def parse_args():
    cfg = default_config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=cfg.data.data_root)
    p.add_argument("--output_dir", type=str, default=cfg.train.output_dir)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--num_epochs", type=int, default=cfg.train.num_epochs)
    p.add_argument("--lr", type=float, default=cfg.train.learning_rate)
    p.add_argument("--num_workers", type=int, default=cfg.train.num_workers)
    p.add_argument("--save_every", type=int, default=cfg.train.save_every_epochs)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lambda_gradient", type=float, default=cfg.train.lambda_gradient)
    p.add_argument("--lambda_channel_corr", type=float, default=cfg.train.lambda_channel_corr)
    p.add_argument("--lambda_frequency", type=float, default=cfg.train.lambda_frequency)
    p.add_argument("--lambda_horizon", type=float, default=cfg.train.lambda_horizon)
    p.add_argument("--lambda_layer_sync", type=float, default=cfg.train.lambda_layer_sync)
    p.add_argument("--smoke", action="store_true", help="Run a single batch and exit.")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision (saves GPU memory).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = default_config()
    cfg.data.data_root = args.data_root
    cfg.train.output_dir = args.output_dir
    cfg.train.lambda_gradient = args.lambda_gradient
    cfg.train.lambda_channel_corr = args.lambda_channel_corr
    cfg.train.lambda_frequency = args.lambda_frequency
    cfg.train.lambda_horizon = args.lambda_horizon
    cfg.train.lambda_layer_sync = args.lambda_layer_sync
    use_amp = args.device.startswith("cuda") and not args.no_amp

    out_dir = Path(args.output_dir) / "autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, stats = build_dataset(cfg.data, stats_path=Path(args.output_dir) / "norm_stats.json")
    print(f"Dataset size: {len(dataset)}; channels={cfg.data.channels}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = build_autoencoder(cfg.ae).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = TrainLogger(
        log_root=Path(args.output_dir) / "logs",
        tag="autoencoder",
        field_names=["total", "recon", "grad", "chcorr", "freq", "horizon", "sync", "kl", "psnr"],
    )
    logger.info(
        f"device={args.device} amp={use_amp} batch_size={args.batch_size} "
        f"epochs={args.num_epochs} lr={args.lr} dataset_size={len(dataset)}"
    )

    # Save the resolved config for reproducibility.
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ae": dataclasses.asdict(cfg.ae),
                "data": dataclasses.asdict(cfg.data),
                "train": dataclasses.asdict(cfg.train),
                "stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"AE epoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                posterior = model.encode(batch).latent_dist
                z = posterior.sample()
                recon = model.decode(z).sample

                loss, parts = autoencoder_loss(
                    recon,
                    batch,
                    lambda_gradient=cfg.train.lambda_gradient,
                    lambda_channel_corr=cfg.train.lambda_channel_corr,
                    lambda_frequency=cfg.train.lambda_frequency,
                    lambda_horizon=cfg.train.lambda_horizon,
                    lambda_layer_sync=cfg.train.lambda_layer_sync,
                )
                # KL regularization on the latent posterior.
                kl = posterior.kl().mean()
                total = loss + 1e-6 * kl

            optim.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(optim)
            scaler.update()

            global_step += 1

            # PSNR on normalized data (proxy 'accuracy' metric).
            with torch.no_grad():
                mse = torch.nn.functional.mse_loss(recon.float(), batch.float())
                psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))

            metrics = {
                "total": total.item(),
                "recon": parts["recon"].item(),
                "grad": parts["grad"].item() if "grad" in parts else 0.0,
                "chcorr": parts["chcorr"].item() if "chcorr" in parts else 0.0,
                "freq": parts["freq"].item() if "freq" in parts else 0.0,
                "horizon": parts["horizon"].item() if "horizon" in parts else 0.0,
                "sync": parts["sync"].item() if "sync" in parts else 0.0,
                "kl": kl.item(),
                "psnr": psnr.item(),
            }
            logger.log_step(
                epoch=epoch + 1,
                step=step + 1,
                global_step=global_step,
                lr=optim.param_groups[0]["lr"],
                metrics=metrics,
            )
            pbar.set_postfix(
                {k: f"{v:.4f}" for k, v in metrics.items() if k != "kl"}
                | {"kl": f"{metrics['kl']:.2e}"}
            )

            if args.smoke:
                logger.info("Smoke test OK: one autoencoder step ran successfully.")
                logger.close()
                return

        logger.end_epoch(epoch + 1)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            ckpt_dir = out_dir / f"epoch_{epoch + 1:04d}"
            model.save_pretrained(ckpt_dir)
            logger.info(f"Saved autoencoder checkpoint to {ckpt_dir}")

    # Final save with a stable name for downstream loading.
    model.save_pretrained(out_dir / "final")
    logger.info(f"Final autoencoder saved to {out_dir / 'final'}")
    logger.close()


if __name__ == "__main__":
    main()
