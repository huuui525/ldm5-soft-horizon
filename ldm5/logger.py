"""Lightweight training logger.

Writes:
- A human-readable .log file (one line per step + epoch summaries).
- A .csv file with per-step metrics for easy plotting.

Each training run gets a timestamped directory under ``outputs/logs/<tag>/``.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping


class TrainLogger:
    def __init__(self, log_root: str | Path, tag: str, field_names: Iterable[str]) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_root) / tag / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "train.log"
        self.csv_path = self.run_dir / "metrics.csv"
        self.epoch_csv_path = self.run_dir / "epoch_metrics.csv"

        self._step_fields = ["epoch", "step", "global_step", "lr", *field_names]
        self._epoch_fields = ["epoch", "num_steps", *[f"{n}_mean" for n in field_names]]

        self._log_file = open(self.log_path, "w", encoding="utf-8")
        self._csv_file = open(self.csv_path, "w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._step_fields)
        self._csv_writer.writeheader()
        self._epoch_csv_file = open(self.epoch_csv_path, "w", encoding="utf-8", newline="")
        self._epoch_csv_writer = csv.DictWriter(self._epoch_csv_file, fieldnames=self._epoch_fields)
        self._epoch_csv_writer.writeheader()

        self._epoch_accum: Dict[str, float] = {}
        self._epoch_count = 0
        self._start_time = time.time()

        self.info(f"Logger initialized at {self.run_dir}")

    # ---------- public API ----------

    def info(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def log_step(
        self,
        epoch: int,
        step: int,
        global_step: int,
        lr: float,
        metrics: Mapping[str, float],
    ) -> None:
        row = {"epoch": epoch, "step": step, "global_step": global_step, "lr": lr}
        for k, v in metrics.items():
            row[k] = float(v)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        for k, v in metrics.items():
            self._epoch_accum[k] = self._epoch_accum.get(k, 0.0) + float(v)
        self._epoch_count += 1

    def end_epoch(self, epoch: int) -> Dict[str, float]:
        if self._epoch_count == 0:
            means: Dict[str, float] = {}
        else:
            means = {k: v / self._epoch_count for k, v in self._epoch_accum.items()}
        elapsed = time.time() - self._start_time
        parts = " ".join(f"{k}={v:.4f}" for k, v in means.items())
        self.info(f"Epoch {epoch} done | steps={self._epoch_count} | {parts} | elapsed={elapsed:.1f}s")

        row = {"epoch": epoch, "num_steps": self._epoch_count}
        for k, v in means.items():
            row[f"{k}_mean"] = v
        self._epoch_csv_writer.writerow(row)
        self._epoch_csv_file.flush()

        self._epoch_accum.clear()
        self._epoch_count = 0
        return means

    def close(self) -> None:
        self._log_file.close()
        self._csv_file.close()
        self._epoch_csv_file.close()
