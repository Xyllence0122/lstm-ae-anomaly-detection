# -*- coding: utf-8 -*-
"""Stateful runtime for the exported causal LSTM forecaster.

The runtime accepts one multivariate sensor sample at a time. Transport and
device I/O are intentionally outside this module so the same detector can be
connected to MQTT, OPC UA, a DAQ callback, or an offline replay.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR


class StreamingAnomalyDetector:
    """Run one-step prediction, trailing error smoothing, and thresholding."""

    def __init__(self, step_model, mean, std, threshold, window, calib,
                 hidden_size, num_layers=1, sensor_names=None):
        self.model = step_model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-9, 1.0, self.std)
        self.threshold = float(threshold)
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.calib = None if calib is None else np.asarray(calib, dtype=float)
        if self.calib is not None:
            self.calib = np.where(np.abs(self.calib) < 1e-12, 1.0, self.calib)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if sensor_names is None:
            sensor_names = [f"sensor_{i}" for i in range(len(self.mean))]
        self.sensor_names = list(sensor_names)
        if len(self.sensor_names) != len(self.mean):
            raise ValueError("sensor_names must align with normalization arrays")
        self.reset()

    @classmethod
    def from_artifacts(cls, checkpoint_path=None, torchscript_path=None):
        checkpoint_path = Path(
            checkpoint_path or OUTPUT_DIR / "streaming_lstm_forecaster.pt")
        torchscript_path = Path(
            torchscript_path or OUTPUT_DIR / "streaming_lstm_step.ts")
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        model = torch.jit.load(str(torchscript_path), map_location="cpu")
        return cls(
            model,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            threshold=checkpoint["threshold"],
            window=checkpoint["window"],
            calib=checkpoint["calib"] if checkpoint["use_calib"] else None,
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            sensor_names=checkpoint["sensor_names"],
        )

    def reset(self):
        """Reset recurrent state at the start of each wafer/process window."""
        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.cell = torch.zeros_like(self.hidden)
        self.next_prediction = None
        self.errors = deque(maxlen=self.window)
        self.sample_index = -1

    @torch.no_grad()
    def update(self, sample):
        """Consume one sample and return a JSON-serializable detection result."""
        values = np.asarray(sample, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(
                f"expected sample shape {self.mean.shape}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("sample contains NaN or infinite values")

        self.sample_index += 1
        normalized = torch.from_numpy((values - self.mean) / self.std).reshape(
            1, 1, -1)
        if self.next_prediction is not None:
            error = ((self.next_prediction - normalized) ** 2).numpy()[0, 0]
            self.errors.append(error)

        prediction, self.hidden, self.cell = self.model(
            normalized, self.hidden, self.cell)
        self.next_prediction = prediction

        ready = len(self.errors) == self.window
        score = None
        per_sensor = None
        alarm = False
        if ready:
            per_sensor = np.mean(np.stack(self.errors), axis=0)
            if self.calib is not None:
                per_sensor = per_sensor / self.calib
            score = float(per_sensor.max())
            alarm = bool(score > self.threshold)

        return {
            "sample_index": self.sample_index,
            "ready": ready,
            "score": score,
            "threshold": self.threshold,
            "alarm": alarm,
            "per_sensor_score": (
                None if per_sensor is None else {
                    name: float(value)
                    for name, value in zip(self.sensor_names, per_sensor)
                }
            ),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Replay a headerless CSV through the streaming detector")
    parser.add_argument("csv", type=Path,
                        help="Rows are samples; columns follow checkpoint sensor order")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--torchscript", type=Path)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    detector = StreamingAnomalyDetector.from_artifacts(
        args.checkpoint, args.torchscript)
    rows = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    for row in rows:
        result = detector.update(row)
        if args.show_all or result["alarm"]:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
