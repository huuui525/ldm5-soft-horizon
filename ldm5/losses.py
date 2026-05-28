"""Auxiliary losses for five-channel seismic autoencoder pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .horizon import normalize_map, soft_horizon_map


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss over five-channel tensors."""
    return F.l1_loss(pred, target)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between horizontal and vertical gradients.

    Encourages preservation of edges, fault boundaries, and reflector
    continuity in seismic images.
    """
    def _grad(x: torch.Tensor):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    pgx, pgy = _grad(pred)
    tgx, tgy = _grad(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def vertical_edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Return per-channel vertical-gradient magnitudes at the input size."""
    edge = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return F.pad(edge, (0, 0, 1, 0))


def horizon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match multi-scale soft horizon maps between reconstruction and target."""
    pred_map = soft_horizon_map(pred)
    target_map = soft_horizon_map(target)
    loss = F.l1_loss(pred_map, target_map)
    for scale in (2, 4):
        loss = loss + F.l1_loss(
            F.avg_pool2d(pred_map, kernel_size=scale, stride=scale),
            F.avg_pool2d(target_map, kernel_size=scale, stride=scale),
        )
    return loss / 3.0


def layer_sync_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match per-channel horizon edge alignment across the five channels."""
    pred_edges = normalize_map(vertical_edge_magnitude(pred))
    target_edges = normalize_map(vertical_edge_magnitude(target))
    return F.l1_loss(pred_edges, target_edges)


def channel_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match per-sample cross-channel correlation matrices.

    Input shape: [B, C, H, W]. The 5x5 correlation matrix is computed by
    flattening spatial dims and using cosine similarity between channel
    feature vectors. The loss is the Frobenius distance between the
    correlation matrices of `pred` and `target`.
    """
    b, c, h, w = pred.shape
    p = pred.reshape(b, c, h * w)
    t = target.reshape(b, c, h * w)
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    p = F.normalize(p, dim=-1, eps=1e-8)
    t = F.normalize(t, dim=-1, eps=1e-8)
    corr_p = torch.bmm(p, p.transpose(1, 2))
    corr_t = torch.bmm(t, t.transpose(1, 2))
    return F.l1_loss(corr_p, corr_t)


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between 2D FFT magnitudes per channel."""
    p = torch.fft.rfft2(pred, norm="ortho").abs()
    t = torch.fft.rfft2(target, norm="ortho").abs()
    return F.l1_loss(p, t)


def autoencoder_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_gradient: float = 0.0,
    lambda_channel_corr: float = 0.0,
    lambda_frequency: float = 0.0,
    lambda_horizon: float = 0.0,
    lambda_layer_sync: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Combined autoencoder loss. Returns (total, components_dict)."""
    rec = reconstruction_loss(pred, target)
    total = rec
    parts = {"recon": rec.detach()}
    if lambda_gradient > 0:
        g = gradient_loss(pred, target)
        total = total + lambda_gradient * g
        parts["grad"] = g.detach()
    if lambda_channel_corr > 0:
        cc = channel_correlation_loss(pred, target)
        total = total + lambda_channel_corr * cc
        parts["chcorr"] = cc.detach()
    if lambda_frequency > 0:
        fq = frequency_loss(pred, target)
        total = total + lambda_frequency * fq
        parts["freq"] = fq.detach()
    if lambda_horizon > 0:
        hz = horizon_loss(pred, target)
        total = total + lambda_horizon * hz
        parts["horizon"] = hz.detach()
    if lambda_layer_sync > 0:
        sync = layer_sync_loss(pred, target)
        total = total + lambda_layer_sync * sync
        parts["sync"] = sync.detach()
    return total, parts
