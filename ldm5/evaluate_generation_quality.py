"""Evaluate generated five-channel seismic image quality.

The script compares paired real/generated arrays saved by visualization scripts
such as `visualize_horizon_diffusion.py`.

Usage:
    python -m ldm5.evaluate_generation_quality \
        --input_npz outputs_six_channel_horizon_label/horizon_evaluation/horizon_visualize_arrays.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated seismic image quality.")
    parser.add_argument(
        "--input_npz",
        type=str,
        default="outputs_six_channel_horizon_label/horizon_evaluation/horizon_visualize_arrays.npz",
        help="NPZ containing real/generated arrays with shape [N,C,H,W].",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs_quality_metrics/<experiment_dir>.",
    )
    parser.add_argument("--label_quantile", type=float, default=0.75)
    parser.add_argument("--edge_top_quantile", type=float, default=0.90)
    parser.add_argument("--blur_peak_fraction", type=float, default=0.5)
    return parser.parse_args()


def as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def standardize_per_sample_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return (x - mean) / (std + eps)


def box_blur2d(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float64)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out += padded[:, :, dy : dy + x.shape[-2], dx : dx + x.shape[-1]]
    return out / float(kernel_size * kernel_size)


def vertical_edges(x: np.ndarray) -> np.ndarray:
    edge = np.abs(x[..., 1:, :] - x[..., :-1, :])
    return np.pad(edge, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")


def shared_horizon(x: np.ndarray) -> np.ndarray:
    return vertical_edges(x).mean(axis=1, keepdims=True)


def normalize_maps(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return (x - mean) / (std + eps)


def binary_by_quantile(x: np.ndarray, q: float) -> np.ndarray:
    flat = x.reshape(*x.shape[:-2], -1)
    thresh = np.quantile(flat, q, axis=-1)[..., None, None]
    return x >= thresh


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv)) + eps
    return float(np.sum(av * bv) / denom)


def corr_matrix(channels: np.ndarray) -> np.ndarray:
    c = channels.reshape(channels.shape[0], -1)
    c = c - c.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
    c = c / norm
    return c @ c.T


def run_lengths(mask_1d: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask_1d:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def blur_width_from_edges(edges: np.ndarray, peak_fraction: float) -> float:
    widths: list[int] = []
    # edges shape [H,W]
    for col in range(edges.shape[1]):
        profile = edges[:, col]
        peak = profile.max()
        if peak <= 1e-12:
            continue
        mask = profile >= peak_fraction * peak
        widths.extend(run_lengths(mask))
    if not widths:
        return 0.0
    return float(np.mean(widths))


def high_frequency_ratio(x: np.ndarray, radius_fraction: float = 0.35) -> float:
    h, w = x.shape
    spectrum = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spectrum) ** 2
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = radius >= radius_fraction * radius.max()
    return float(power[mask].sum() / (power.sum() + 1e-12))


def spectrum_l1(a: np.ndarray, b: np.ndarray) -> float:
    sa = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a))))
    sb = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(b))))
    sa = (sa - sa.mean()) / (sa.std() + 1e-8)
    sb = (sb - sb.mean()) / (sb.std() + 1e-8)
    return float(np.mean(np.abs(sa - sb)))


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    data_range = max(float(a.max() - a.min()), float(b.max() - b.min()), 1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def precision_recall_f1_iou(real_mask: np.ndarray, gen_mask: np.ndarray) -> dict[str, float]:
    tp = np.logical_and(real_mask, gen_mask).sum()
    fp = np.logical_and(~real_mask, gen_mask).sum()
    fn = np.logical_and(real_mask, ~gen_mask).sum()
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    iou = float(tp / max(tp + fp + fn, 1))
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def evaluate(real: np.ndarray, generated: np.ndarray, channels: list[str], args: argparse.Namespace) -> dict:
    real = as_float(real)
    generated = as_float(generated)
    real_z = standardize_per_sample_channel(real)
    gen_z = standardize_per_sample_channel(generated)

    real_blur = box_blur2d(real_z)
    gen_blur = box_blur2d(gen_z)
    real_noise = ((real_z - real_blur) ** 2).mean(axis=(-2, -1)) / ((real_z**2).mean(axis=(-2, -1)) + 1e-12)
    gen_noise = ((gen_z - gen_blur) ** 2).mean(axis=(-2, -1)) / ((gen_z**2).mean(axis=(-2, -1)) + 1e-12)

    real_edges = vertical_edges(real_z)
    gen_edges = vertical_edges(gen_z)
    edge_threshold_real = np.quantile(real_edges.reshape(*real_edges.shape[:2], -1), args.edge_top_quantile, axis=-1)
    edge_threshold_gen = np.quantile(gen_edges.reshape(*gen_edges.shape[:2], -1), args.edge_top_quantile, axis=-1)
    real_strong = np.where(real_edges >= edge_threshold_real[..., None, None], real_edges, np.nan)
    gen_strong = np.where(gen_edges >= edge_threshold_gen[..., None, None], gen_edges, np.nan)
    real_edge_sharpness = np.nanmean(real_strong, axis=(-2, -1))
    gen_edge_sharpness = np.nanmean(gen_strong, axis=(-2, -1))
    real_bg = np.where(real_edges <= np.quantile(real_edges.reshape(*real_edges.shape[:2], -1), 0.50, axis=-1)[..., None, None], real_edges, np.nan)
    gen_bg = np.where(gen_edges <= np.quantile(gen_edges.reshape(*gen_edges.shape[:2], -1), 0.50, axis=-1)[..., None, None], gen_edges, np.nan)
    real_edge_contrast = real_edge_sharpness / (np.nanmean(real_bg, axis=(-2, -1)) + 1e-8)
    gen_edge_contrast = gen_edge_sharpness / (np.nanmean(gen_bg, axis=(-2, -1)) + 1e-8)

    real_h = normalize_maps(shared_horizon(real_z))
    gen_h = normalize_maps(shared_horizon(gen_z))
    real_h_mask = binary_by_quantile(real_h, args.label_quantile)
    gen_h_mask = binary_by_quantile(gen_h, args.label_quantile)

    per_sample: list[dict] = []
    per_channel_rows: list[dict] = []
    for n in range(real.shape[0]):
        channel_corr_real = corr_matrix(real_z[n])
        channel_corr_gen = corr_matrix(gen_z[n])
        sync_real = corr_matrix(real_edges[n])
        sync_gen = corr_matrix(gen_edges[n])
        horizon_scores = precision_recall_f1_iou(real_h_mask[n, 0], gen_h_mask[n, 0])

        sample_row = {
            "sample": n,
            "noise_ratio_diff_mean": float(np.mean(np.abs(gen_noise[n] - real_noise[n]))),
            "edge_sharpness_ratio_mean": float(np.mean(gen_edge_sharpness[n] / (real_edge_sharpness[n] + 1e-8))),
            "edge_contrast_ratio_mean": float(np.mean(gen_edge_contrast[n] / (real_edge_contrast[n] + 1e-8))),
            "blur_width_diff_mean": 0.0,
            "horizon_corr": safe_corr(real_h[n, 0], gen_h[n, 0]),
            "horizon_iou": horizon_scores["iou"],
            "horizon_f1": horizon_scores["f1"],
            "channel_corr_diff": float(np.mean(np.abs(channel_corr_real - channel_corr_gen))),
            "sync_matrix_diff": float(np.mean(np.abs(sync_real - sync_gen))),
            "ssim_mean": float(np.mean([global_ssim(real_z[n, c], gen_z[n, c]) for c in range(real.shape[1])])),
            "spectrum_l1_mean": float(np.mean([spectrum_l1(real_z[n, c], gen_z[n, c]) for c in range(real.shape[1])])),
            "high_freq_ratio_diff_mean": float(
                np.mean(
                    [
                        abs(high_frequency_ratio(gen_z[n, c]) - high_frequency_ratio(real_z[n, c]))
                        for c in range(real.shape[1])
                    ]
                )
            ),
        }
        blur_diffs = []
        for c, name in enumerate(channels):
            real_blur_width = blur_width_from_edges(real_edges[n, c], args.blur_peak_fraction)
            gen_blur_width = blur_width_from_edges(gen_edges[n, c], args.blur_peak_fraction)
            blur_diffs.append(abs(gen_blur_width - real_blur_width))
            per_channel_rows.append(
                {
                    "sample": n,
                    "channel": name,
                    "real_noise_ratio": float(real_noise[n, c]),
                    "gen_noise_ratio": float(gen_noise[n, c]),
                    "noise_ratio_diff": float(abs(gen_noise[n, c] - real_noise[n, c])),
                    "real_edge_sharpness": float(real_edge_sharpness[n, c]),
                    "gen_edge_sharpness": float(gen_edge_sharpness[n, c]),
                    "edge_sharpness_ratio": float(gen_edge_sharpness[n, c] / (real_edge_sharpness[n, c] + 1e-8)),
                    "real_edge_contrast": float(real_edge_contrast[n, c]),
                    "gen_edge_contrast": float(gen_edge_contrast[n, c]),
                    "edge_contrast_ratio": float(gen_edge_contrast[n, c] / (real_edge_contrast[n, c] + 1e-8)),
                    "real_blur_width": real_blur_width,
                    "gen_blur_width": gen_blur_width,
                    "blur_width_diff": float(abs(gen_blur_width - real_blur_width)),
                    "ssim": global_ssim(real_z[n, c], gen_z[n, c]),
                    "spectrum_l1": spectrum_l1(real_z[n, c], gen_z[n, c]),
                    "real_high_freq_ratio": high_frequency_ratio(real_z[n, c]),
                    "gen_high_freq_ratio": high_frequency_ratio(gen_z[n, c]),
                }
            )
        sample_row["blur_width_diff_mean"] = float(np.mean(blur_diffs))
        per_sample.append(sample_row)

    summary = summarize(per_sample, per_channel_rows, channels)
    return {"summary": summary, "per_sample": per_sample, "per_channel": per_channel_rows}


def summarize(per_sample: list[dict], per_channel_rows: list[dict], channels: list[str]) -> dict:
    summary: dict[str, object] = {}
    metric_keys = [k for k in per_sample[0].keys() if k != "sample"]
    for key in metric_keys:
        values = np.array([row[key] for row in per_sample], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}

    by_channel: dict[str, dict[str, float]] = {}
    channel_metric_keys = [k for k in per_channel_rows[0].keys() if k not in ("sample", "channel")]
    for channel in channels:
        rows = [r for r in per_channel_rows if r["channel"] == channel]
        by_channel[channel] = {key: float(np.mean([r[key] for r in rows])) for key in channel_metric_keys}
    summary["by_channel_mean"] = by_channel
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_npz)
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        experiment_name = input_path.parent.parent.name if input_path.parent.name else input_path.stem
        save_dir = Path("outputs_quality_metrics") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(input_path)
    real = arrays["real"]
    generated = arrays["generated"]
    channels = [str(c) for c in arrays["channels"]]
    report = evaluate(real, generated, channels, args)

    with open(save_dir / "quality_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_csv(save_dir / "quality_per_sample.csv", report["per_sample"])
    write_csv(save_dir / "quality_per_channel.csv", report["per_channel"])

    print(f"Saved quality report to {save_dir / 'quality_report.json'}")
    print(f"Saved per-sample CSV to {save_dir / 'quality_per_sample.csv'}")
    print(f"Saved per-channel CSV to {save_dir / 'quality_per_channel.csv'}")
    print("\nKey summary:")
    for key in (
        "noise_ratio_diff_mean",
        "edge_sharpness_ratio_mean",
        "edge_contrast_ratio_mean",
        "blur_width_diff_mean",
        "horizon_corr",
        "horizon_iou",
        "horizon_f1",
        "channel_corr_diff",
        "sync_matrix_diff",
        "ssim_mean",
    ):
        item = report["summary"][key]
        print(f"  {key}: mean={item['mean']:.4f}, std={item['std']:.4f}")


if __name__ == "__main__":
    main()
