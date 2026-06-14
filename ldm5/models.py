"""Model factories for the five-channel seismic Latent Diffusion pipeline.

We rely on Hugging Face `diffusers`:
  * `AutoencoderKL` as the five-channel latent autoencoder.
  * `UNet2DModel` as the latent-space DDPM noise predictor.
  * `DDPMScheduler` for the forward / reverse diffusion process.
"""

from __future__ import annotations

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

from .config import AEConfig, TrainConfig, UNetConfig


def build_autoencoder(cfg: AEConfig) -> AutoencoderKL:
    """Build a five-channel `AutoencoderKL`.

    `len(block_out_channels) - 1` determines the spatial downsample factor in
    powers of two. With `block_out_channels=(64, 128, 256)` we get factor 4.
    """
    n_blocks = len(cfg.block_out_channels)
    expected_factor = 2 ** (n_blocks - 1)
    if expected_factor != cfg.downsample_factor:
        raise ValueError(
            f"block_out_channels has {n_blocks} entries (factor {expected_factor}), "
            f"but cfg.downsample_factor={cfg.downsample_factor}."
        )

    down_block_types = tuple(["DownEncoderBlock2D"] * n_blocks)
    up_block_types = tuple(["UpDecoderBlock2D"] * n_blocks)

    return AutoencoderKL(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        latent_channels=cfg.latent_channels,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )


def build_unet(cfg: UNetConfig) -> UNet2DModel:
    """Build the latent-space U-Net noise predictor."""
    return UNet2DModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
    )


def build_scheduler(cfg: TrainConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        prediction_type=cfg.prediction_type,
    )
