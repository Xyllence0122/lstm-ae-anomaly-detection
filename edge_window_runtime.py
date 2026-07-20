# -*- coding: utf-8 -*-
"""Stateful edge runtime for the causal sliding-window LSTM autoencoder."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR


class SlidingWindowAnomalyDetector:
    """Consume one sensor row at a time and emit causal LSTM-AE alarms."""

    def __init__(self, model, mean, std, threshold, window_size, calib,
                  persistence_required=1, persistence_span=1,
                 sensor_names=None, score_mode="last"):
        self.model = model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-9, 1.0, self.std)
        self.calib = np.asarray(calib, dtype=np.float64)
        self.calib = np.where(np.abs(self.calib) < 1e-12, 1.0, self.calib)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.persistence_required = int(persistence_required)
        self.persistence_span = int(persistence_span)
        self.score_mode = str(score_mode)
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not (1 <= self.persistence_required <= self.persistence_span):
            raise ValueError("persistence must satisfy 1 <= required <= span")
        if self.score_mode not in (
                "last", "mean", "max", "delta_mean", "delta_max"):
            raise ValueError(
                "score_mode must be one of: last, mean, max, "
                "delta_mean, delta_max")
        if sensor_names is None:
            sensor_names = [f"sensor_{index}"
                            for index in range(len(self.mean))]
        self.sensor_names = list(sensor_names)
        if not (len(self.mean) == len(self.std) == len(self.calib) ==
                len(self.sensor_names)):
            raise ValueError("normalization, calibration, and sensors must align")
        self.reset()

    @classmethod
    def from_artifacts(cls, checkpoint_path=None, model_path=None):
        checkpoint_path = Path(
            checkpoint_path or OUTPUT_DIR / "sliding_window_lstm_ae.pt")
        model_path = Path(
            model_path or OUTPUT_DIR / "sliding_window_lstm_ae.ts")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        model = torch.jit.load(str(model_path), map_location="cpu")
        return cls(
            model=model,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            threshold=checkpoint["threshold"],
            window_size=checkpoint["window_size"],
            calib=checkpoint["calib"],
            persistence_required=checkpoint["persistence_required"],
            persistence_span=checkpoint["persistence_span"],
            sensor_names=checkpoint["sensor_names"],
            score_mode=checkpoint.get("score_mode", "last"),
        )

    def reset(self):
        """Reset state at a wafer, recipe, or monitored-process boundary."""
        self.samples = deque(maxlen=self.window_size)
        self.recent_scores = deque(maxlen=self.persistence_span)
        self.sample_index = -1
        self.last_timestamp = None

    @torch.no_grad()
    def update(self, sample, timestamp=None):
        values = np.asarray(sample, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(
                f"expected sample shape {self.mean.shape}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("sample contains NaN or infinite values")
        if (timestamp is not None and self.last_timestamp is not None and
                timestamp <= self.last_timestamp):
            raise ValueError("timestamps must be strictly increasing")

        self.sample_index += 1
        self.last_timestamp = timestamp
        normalized = (values - self.mean) / self.std
        self.samples.append(normalized)

        result = {
            "sample_index": self.sample_index,
            "timestamp": timestamp,
            "window_ready": len(self.samples) == self.window_size,
            "alarm_ready": False,
            "raw_score": None,
            "score": None,
            "threshold": self.threshold,
            "alarm": False,
            "per_sensor_score": None,
            "score_mode": self.score_mode,
        }
        if not result["window_ready"]:
            return result

        model_input = torch.from_numpy(
            np.stack(self.samples).astype(np.float32)).unsqueeze(0)
        reconstruction = self.model(model_input)
        squared = ((reconstruction - model_input) ** 2).numpy()[0]
        if self.score_mode == "last":
            errors = squared[-1]
        elif self.score_mode == "mean":
            errors = squared.mean(axis=0)
        elif self.score_mode == "max":
            errors = squared.max(axis=0)
        else:
            reconstruction_values = reconstruction.numpy()[0]
            model_values = model_input.numpy()[0]
            delta_squared = (
                np.diff(reconstruction_values, axis=0) -
                np.diff(model_values, axis=0)
            ) ** 2
            if self.score_mode == "delta_mean":
                errors = delta_squared.mean(axis=0)
            else:
                errors = delta_squared.max(axis=0)
        errors = errors.astype(np.float64)
        calibrated = errors / self.calib
        raw_score = float(calibrated.max())
        self.recent_scores.append(raw_score)
        result["raw_score"] = raw_score
        result["per_sensor_score"] = {
            name: float(value)
            for name, value in zip(self.sensor_names, calibrated)
        }

        if len(self.recent_scores) < self.persistence_span:
            return result
        persistent_score = float(np.partition(
            np.asarray(self.recent_scores, dtype=np.float64),
            -self.persistence_required,
        )[-self.persistence_required])
        result["alarm_ready"] = True
        result["score"] = persistent_score
        result["alarm"] = bool(persistent_score > self.threshold)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Replay a headerless CSV through sliding-window LSTM-AE")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--timestamp-column", type=int,
        help="Optional zero-based numeric timestamp column removed from sensors",
    )
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    detector = SlidingWindowAnomalyDetector.from_artifacts(
        args.checkpoint, args.model)
    rows = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    for row in rows:
        timestamp = None
        values = row
        if args.timestamp_column is not None:
            timestamp = float(row[args.timestamp_column])
            values = np.delete(row, args.timestamp_column)
        result = detector.update(values, timestamp=timestamp)
        if args.show_all or result["alarm"]:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
