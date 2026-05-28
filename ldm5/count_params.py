"""Stand-alone parameter counting module.

Instantiates the autoencoder and latent U-Net using the default config (or a
user-provided JSON config) and reports total / trainable parameter counts per
component. Does NOT require the dataset.

Usage:
    python -m ldm5.count_params
    python -m ldm5.count_params --config path/to/config.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Dict, Tuple

import torch.nn as nn

from .config import AEConfig, FullConfig, UNetConfig, default_config
from .models import build_autoencoder, build_unet


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _fmt(n: int) -> str:
    return f"{n:>14,d}  ({n / 1e6:7.3f} M)"


def report(ae_cfg: AEConfig, unet_cfg: UNetConfig) -> Dict[str, Dict[str, int]]:
    ae = build_autoencoder(ae_cfg)
    unet = build_unet(unet_cfg)

    ae_total, ae_train = count_parameters(ae)
    unet_total, unet_train = count_parameters(unet)
    combined_total = ae_total + unet_total
    combined_train = ae_train + unet_train

    print("=" * 68)
    print("Five-channel Latent Diffusion parameter count")
    print("=" * 68)
    print(f"{'Component':<28}{'Total params':>20}{'Trainable':>20}")
    print("-" * 68)
    print(f"{'AutoencoderKL (5ch)':<28}{_fmt(ae_total):>20}{_fmt(ae_train):>20}")
    print(f"{'Latent U-Net (DDPM)':<28}{_fmt(unet_total):>20}{_fmt(unet_train):>20}")
    print("-" * 68)
    print(f"{'Combined':<28}{_fmt(combined_total):>20}{_fmt(combined_train):>20}")
    print("=" * 68)

    return {
        "autoencoder": {"total": ae_total, "trainable": ae_train},
        "latent_unet": {"total": unet_total, "trainable": unet_train},
        "combined": {"total": combined_total, "trainable": combined_train},
    }


def _load_config(path: str | None) -> FullConfig:
    if path is None:
        return default_config()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = default_config()
    if "ae" in raw:
        cfg.ae = AEConfig(**raw["ae"])
    if "unet" in raw:
        cfg.unet = UNetConfig(**raw["unet"])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters of the LDM5 pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--dump", type=str, default=None, help="Optional path to dump JSON report.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    print("AE config :", dataclasses.asdict(cfg.ae))
    print("UNet config:", dataclasses.asdict(cfg.unet))
    print()
    result = report(cfg.ae, cfg.unet)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved report to {args.dump}")


if __name__ == "__main__":
    main()
