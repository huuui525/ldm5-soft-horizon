"""Default configuration for five-channel seismic Latent Diffusion pretraining."""

from dataclasses import dataclass, field
from typing import Tuple


# ---------- Data ----------

@dataclass
class DataConfig:
    # Root directory that contains the five channel subfolders.
    data_root: str = "标签"
    channels: Tuple[str, ...] = ("dn", "gas", "gr", "vp", "vs")
    image_size: int = 224          # Native size of the .npy samples.
    # Path where per-channel normalization statistics are cached.
    stats_path: str = "outputs/norm_stats.json"


# ---------- AutoencoderKL ----------

@dataclass
class AEConfig:
    in_channels: int = 5
    out_channels: int = 5
    latent_channels: int = 8       # First-run default chosen in SKILL.md.
    # 3 down-blocks => 2 actual spatial downsamples => factor 4.
    block_out_channels: Tuple[int, ...] = (64, 128, 256)
    layers_per_block: int = 2
    # For 224 input, factor 4 yields 56x56 latents.
    downsample_factor: int = 4


# ---------- Latent U-Net (DDPM) ----------

@dataclass
class UNetConfig:
    sample_size: int = 56          # image_size / downsample_factor.
    in_channels: int = 8           # Equal to latent_channels.
    out_channels: int = 8
    layers_per_block: int = 2
    block_out_channels: Tuple[int, ...] = (64, 128, 256, 256)
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )


# ---------- Training ----------

@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 0           # Windows-friendly default.
    log_every: int = 20
    save_every_epochs: int = 5
    seed: int = 42
    # Auxiliary loss weights for the autoencoder. Set to 0 to disable.
    lambda_gradient: float = 0.1
    lambda_channel_corr: float = 0.05
    lambda_frequency: float = 0.0
    lambda_horizon: float = 0.2
    lambda_layer_sync: float = 0.05
    # Diffusion specific.
    num_train_timesteps: int = 1000
    prediction_type: str = "epsilon"


@dataclass
class FullConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ae: AEConfig = field(default_factory=AEConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def default_config() -> FullConfig:
    return FullConfig()
