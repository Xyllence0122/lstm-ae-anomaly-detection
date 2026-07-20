# -*- coding: utf-8 -*-
"""Stateful edge runtime for the causal sliding-window LSTM autoencoder."""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from deployment_manifest import (
    file_sha256,
    load_deployment_manifest,
    sensor_schema_hash,
)


class SlidingWindowAnomalyDetector:
    """Consume one sensor row at a time and emit causal LSTM-AE alarms."""

    def __init__(self, model, mean, std, threshold, window_size, calib,
                  persistence_required=1, persistence_span=1,
                  sensor_names=None, score_mode="last", profile_id=None,
                  expected_schema_hash=None):
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
        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must not contain duplicates")
        self.schema_hash = sensor_schema_hash(self.sensor_names)
        if (expected_schema_hash is not None and
                expected_schema_hash != self.schema_hash):
            raise ValueError(
                f"sensor schema hash mismatch: expected {expected_schema_hash}, "
                f"got {self.schema_hash}")
        self.profile_id = profile_id or "unversioned"
        self.reset()

    @classmethod
    def from_artifacts(cls, checkpoint_path=None, model_path=None,
                       manifest_path=None, profile_id=None):
        """Load a hash-verified deployment profile.

        The default profile is the final normal-only calibration operating
        point recorded in the deployment manifest. Validation experiments must
        call :meth:`from_validation_artifacts` explicitly.
        """
        manifest_path = Path(
            manifest_path or OUTPUT_DIR / "edge_deployment_manifest.json")
        document = load_deployment_manifest(
            manifest_path, verify_artifacts=True, verify_provenance=False,
            verify_runtime=True)
        payload = document["payload"]
        profile_id = profile_id or payload["default_profile"]
        if profile_id not in payload["profiles"]:
            raise ValueError(f"unknown deployment profile: {profile_id}")
        profile = payload["profiles"][profile_id]
        if profile.get("model_id") != "sliding_window_lstm_ae":
            raise ValueError(
                f"profile {profile_id} is not a sliding-window LSTM-AE profile")
        contract_schema_hash = payload["model_contract"]["schema_hash"]
        if profile.get("schema_hash") != contract_schema_hash:
            raise ValueError(
                f"profile {profile_id} does not match the model sensor schema")

        project_dir = manifest_path.resolve().parent.parent
        checkpoint_artifact = payload["artifacts"][
            profile["checkpoint_artifact"]]
        model_artifact = payload["artifacts"][profile["model_artifact"]]
        checkpoint_path = Path(
            checkpoint_path or project_dir / checkpoint_artifact["path"])
        model_path = Path(model_path or project_dir / model_artifact["path"])
        for path, artifact in (
                (checkpoint_path, checkpoint_artifact),
                (model_path, model_artifact)):
            actual = file_sha256(path)
            if actual != artifact["sha256"]:
                raise ValueError(
                    f"artifact hash mismatch for {path}: expected "
                    f"{artifact['sha256']}, got {actual}")
        return cls._load(
            checkpoint_path, model_path, threshold=profile["threshold"],
            profile_id=profile_id,
            expected_schema_hash=contract_schema_hash)

    @classmethod
    def from_validation_artifacts(cls, checkpoint_path=None, model_path=None):
        """Load the original validation operating point from the checkpoint."""
        checkpoint_path = Path(
            checkpoint_path or OUTPUT_DIR / "sliding_window_lstm_ae.pt")
        model_path = Path(
            model_path or OUTPUT_DIR / "sliding_window_lstm_ae.ts")
        return cls._load(
            checkpoint_path, model_path, threshold=None,
            profile_id="validation_operating_point")

    @classmethod
    def _load(cls, checkpoint_path, model_path, threshold, profile_id,
              expected_schema_hash=None):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        model = torch.jit.load(str(model_path), map_location="cpu")
        return cls(
            model=model,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            threshold=(checkpoint["threshold"] if threshold is None
                       else threshold),
            window_size=checkpoint["window_size"],
            calib=checkpoint["calib"],
            persistence_required=checkpoint["persistence_required"],
            persistence_span=checkpoint["persistence_span"],
            sensor_names=checkpoint["sensor_names"],
            score_mode=checkpoint.get("score_mode", "last"),
            profile_id=profile_id,
            expected_schema_hash=expected_schema_hash,
        )

    def reset(self):
        """Reset state at a wafer, recipe, or monitored-process boundary."""
        self.samples = deque(maxlen=self.window_size)
        self.recent_scores = deque(maxlen=self.persistence_span)
        self.sample_index = -1
        self.last_timestamp = None
        self.timestamp_mode = None

    def _validate_timestamp(self, timestamp):
        mode = "absent" if timestamp is None else "present"
        if self.timestamp_mode is not None and mode != self.timestamp_mode:
            raise ValueError(
                "timestamp presence must remain consistent within a stream")
        if timestamp is None:
            return None, mode
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be a finite numeric value") from exc
        if not np.isfinite(value):
            raise ValueError("timestamp must be a finite numeric value")
        if self.last_timestamp is not None and value <= self.last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        return value, mode

    def _validate_columns(self, columns):
        columns = list(columns)
        if len(set(columns)) != len(columns):
            raise ValueError("sensor columns must not contain duplicates")
        if columns != self.sensor_names:
            missing = [name for name in self.sensor_names
                       if name not in columns]
            extra = [name for name in columns
                     if name not in self.sensor_names]
            raise ValueError(
                "sensor schema mismatch: expected ordered columns "
                f"{self.sensor_names}, got {columns}; missing={missing}, "
                f"extra={extra}")
        return columns

    def update_ordered(self, values, columns, timestamp=None):
        """Update from an explicit ordered header and reject schema mismatch."""
        columns = self._validate_columns(columns)
        values = list(values)
        if len(values) != len(columns):
            raise ValueError(
                f"expected {len(columns)} values, got {len(values)}")
        return self.update(dict(zip(columns, values)), timestamp=timestamp)

    @torch.no_grad()
    def update(self, sample, timestamp=None):
        if not isinstance(sample, Mapping):
            raise TypeError(
                "sample must be an ordered sensor-name mapping; use "
                "update_ordered(values, columns) for tabular input")
        columns = self._validate_columns(sample.keys())
        values = np.asarray([sample[name] for name in columns],
                            dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(
                f"expected sample shape {self.mean.shape}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("sample contains NaN or infinite values")
        timestamp, timestamp_mode = self._validate_timestamp(timestamp)

        self.sample_index += 1
        self.timestamp_mode = timestamp_mode
        if timestamp is not None:
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
            "schema_hash": self.schema_hash,
            "profile_id": self.profile_id,
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
        description="Replay a headered CSV through sliding-window LSTM-AE")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile")
    parser.add_argument(
        "--timestamp-column",
        help="Optional timestamp column name; it must be present on every row",
    )
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    detector = SlidingWindowAnomalyDetector.from_artifacts(
        args.checkpoint, args.model, args.manifest, args.profile)
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty and has no sensor header") from exc
        if len(set(header)) != len(header):
            raise ValueError("CSV header contains duplicate columns")
        sensor_columns = [name for name in header
                          if name != args.timestamp_column]
        detector._validate_columns(sensor_columns)
        if (args.timestamp_column is not None and
                args.timestamp_column not in header):
            raise ValueError(
                f"timestamp column {args.timestamp_column!r} is missing")
        timestamp_index = (
            header.index(args.timestamp_column)
            if args.timestamp_column is not None else None)
        sensor_indices = [header.index(name) for name in sensor_columns]
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {line_number} has {len(row)} values; "
                    f"expected {len(header)}")
            timestamp = (
                float(row[timestamp_index])
                if timestamp_index is not None else None)
            values = [float(row[index]) for index in sensor_indices]
            result = detector.update_ordered(
                values, sensor_columns, timestamp=timestamp)
            if args.show_all or result["alarm"]:
                print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
