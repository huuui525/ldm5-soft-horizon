"""Multi-scale horizon-conditioned wrapper around diffusers UNet2DModel."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class HorizonAdapter(nn.Module):
    """Project a one-channel horizon map into a U-Net feature tensor."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        condition = F.interpolate(condition, size=size, mode="bilinear", align_corners=False)
        return self.net(condition)


class MultiScaleHorizonUNet(nn.Module):
    """UNet2DModel with horizon features injected before each down block.

    The base U-Net still predicts latent diffusion noise. The extra condition
    path injects the same horizon prior at progressively coarser resolutions:
    shallow blocks receive detailed horizons, deeper blocks receive large-scale
    structural boundaries after resizing.
    """

    def __init__(
        self,
        unet_config: dict[str, Any],
        condition_channels: int = 1,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if condition_channels != 1:
            raise ValueError("Only one-channel horizon conditions are supported.")
        self.unet = UNet2DModel(**unet_config)
        self.condition_channels = condition_channels
        self.condition_scale = condition_scale
        block_out = list(unet_config["block_out_channels"])
        inject_channels = [block_out[0]] + block_out[:-1]
        self.horizon_adapters = nn.ModuleList(
            [HorizonAdapter(channels) for channels in inject_channels]
        )

    @property
    def config(self):
        return self.unet.config

    @property
    def dtype(self):
        return self.unet.dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        horizon: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        unet = self.unet

        if unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=unet.dtype)
        emb = unet.time_embedding(t_emb)

        if unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")
            if unet.config.class_embed_type == "timestep":
                class_labels = unet.time_proj(class_labels)
            class_emb = unet.class_embedding(class_labels).to(dtype=unet.dtype)
            emb = emb + class_emb
        elif unet.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        horizon = horizon.to(device=sample.device, dtype=sample.dtype)

        skip_sample = sample
        sample = unet.conv_in(sample)

        down_block_res_samples = (sample,)
        for level, downsample_block in enumerate(unet.down_blocks):
            cond = self.horizon_adapters[level](horizon, sample.shape[-2:]).to(dtype=sample.dtype)
            sample = sample + self.condition_scale * cond

            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb)

        skip_sample = None
        for upsample_block in unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)
        return SimpleNamespace(sample=sample)

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "unet_config": dict(self.unet.config),
            "condition_channels": self.condition_channels,
            "condition_scale": self.condition_scale,
            "_class_name": self.__class__.__name__,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, load_dir: str | Path, map_location: str | torch.device = "cpu"):
        load_dir = Path(load_dir)
        with open(load_dir / "config.json", "r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = cls(
            unet_config=config["unet_config"],
            condition_channels=config.get("condition_channels", 1),
            condition_scale=config.get("condition_scale", 1.0),
        )
        state = torch.load(load_dir / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state)
        return model
