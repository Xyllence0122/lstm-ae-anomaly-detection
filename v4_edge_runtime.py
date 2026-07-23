# -*- coding: utf-8 -*-
"""Hash-verified V4 streaming runtime for the V3.2 multiscale LSTM-AE."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from config import PROJECT_DIR
from deployment_manifest import (
    file_sha256,
    normalized_text_sha256,
    sensor_schema_hash,
)


DEFAULT_MANIFEST = (
    PROJECT_DIR / "outputs" / "v4" / "deployment_manifest_v4.json"
)


def _sidecar_path(manifest_path):
    return Path(manifest_path).with_suffix(".sha256")


def load_v4_manifest(path=DEFAULT_MANIFEST, verify_provenance=True):
    """Load a V4 deployment manifest and verify all bound artifacts."""
    path = Path(path).resolve()
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"deployment manifest and sidecar are required: {path}, {sidecar}")
    expected_manifest_hash = sidecar.read_text(
        encoding="ascii").strip().split()[0]
    actual_manifest_hash = file_sha256(path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "deployment manifest hash mismatch: "
            f"expected {expected_manifest_hash}, got {actual_manifest_hash}")

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("manifest_version") != 4:
        raise ValueError("unsupported V4 deployment manifest version")
    project_root = path.parents[2]
    for artifact_id, artifact in document["artifacts"].items():
        artifact_path = project_root / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"missing deployment artifact {artifact_id}: {artifact_path}")
        actual = file_sha256(artifact_path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"artifact hash mismatch for {artifact_id}: "
                f"expected {artifact['sha256']}, got {actual}")
    if verify_provenance:
        for source_id, source in document["source_provenance"].items():
            source_path = project_root / source["path"]
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"missing provenance source {source_id}: {source_path}")
            hash_mode = source.get("hash_mode", "sha256")
            if hash_mode == "normalized_text_sha256":
                actual = normalized_text_sha256(source_path)
            elif hash_mode == "sha256":
                actual = file_sha256(source_path)
            else:
                raise ValueError(
                    f"unsupported provenance hash mode: {hash_mode}")
            if actual != source["sha256"]:
                raise ValueError(
                    f"provenance hash mismatch for {source_id}: "
                    f"expected {source['sha256']}, got {actual}")
    return document, path, actual_manifest_hash


class V4MultiscaleDetector:
    """Consume one named sensor row at a time and emit V4 causal alarms."""

    def __init__(self, model, feature_spec, profiles, threshold,
                 timing_contract, model_version, artifact_sha256,
                 manifest_sha256="unversioned"):
        self.model = model.eval()
        self.feature_spec = dict(feature_spec)
        self.raw_sensor_names = list(
            self.feature_spec["raw_sensor_names"])
        self.feature_names = list(self.feature_spec["feature_names"])
        self.score_feature_indices = np.asarray(
            self.feature_spec["score_feature_indices"], dtype=int)
        self.feature_mean = np.asarray(
            self.feature_spec["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(
            self.feature_spec["feature_std"], dtype=np.float32)
        self.feature_std = np.where(
            self.feature_std < 1e-9, 1.0, self.feature_std)
        self.expected_cycle_samples = float(
            self.feature_spec["expected_cycle_samples"])
        self.profiles = [self._validated_profile(index, item)
                         for index, item in enumerate(profiles)]
        self.threshold = float(threshold)
        self.timing_contract = dict(timing_contract)
        self.model_version = str(model_version)
        self.artifact_sha256 = str(artifact_sha256)
        self.manifest_sha256 = str(manifest_sha256)
        self.schema_hash = sensor_schema_hash(self.raw_sensor_names)
        expected_schema_hash = self.feature_spec.get("raw_schema_hash")
        if (expected_schema_hash is not None and
                expected_schema_hash != self.schema_hash):
            raise ValueError(
                "feature specification sensor schema hash mismatch")
        if len(set(self.raw_sensor_names)) != len(self.raw_sensor_names):
            raise ValueError("raw sensor names must be unique")
        if len(self.feature_mean) != len(self.feature_names):
            raise ValueError("feature normalization and names do not align")
        if not np.array_equal(
                self.score_feature_indices,
                np.arange(len(self.raw_sensor_names) * 2)):
            raise ValueError(
                "V4 requires raw and per-sample-difference score features")
        self._validate_timing_contract()
        self.context = None
        self.reset()

    @classmethod
    def from_manifest(cls, path=DEFAULT_MANIFEST, verify_provenance=True):
        document, manifest_path, manifest_hash = load_v4_manifest(
            path, verify_provenance=verify_provenance)
        project_root = manifest_path.parents[2]
        model_record = document["artifacts"]["torchscript_model"]
        model_path = project_root / model_record["path"]
        model = torch.jit.load(str(model_path), map_location="cpu")
        contract = document["model_contract"]
        detector = cls(
            model=model,
            feature_spec=contract["feature_spec"],
            profiles=contract["profiles"],
            threshold=contract["ensemble_threshold"],
            timing_contract=document["timing_contract"],
            model_version=document["model_version"],
            artifact_sha256=model_record["sha256"],
            manifest_sha256=manifest_hash,
        )
        if detector.schema_hash != contract["raw_sensor_schema_hash"]:
            raise ValueError("runtime sensor schema does not match manifest")
        return detector

    def _validated_profile(self, index, profile):
        profile = dict(profile)
        required = {
            "window_size", "score_mode", "persistence_required",
            "persistence_span", "base_scale", "calibration",
        }
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"profile {index} is missing {missing}")
        profile.setdefault("profile_id", f"profile_{index}")
        profile["window_size"] = int(profile["window_size"])
        profile["persistence_required"] = int(
            profile["persistence_required"])
        profile["persistence_span"] = int(profile["persistence_span"])
        profile["base_scale"] = float(profile["base_scale"])
        profile["calibration"] = np.asarray(
            profile["calibration"], dtype=np.float64)
        if profile["score_mode"] not in ("last", "mean", "max"):
            raise ValueError(
                "V4 runtime supports last, mean, and max score modes")
        if profile["window_size"] < 2:
            raise ValueError("profile window_size must be at least 2")
        if not (
            1 <= profile["persistence_required"] <=
            profile["persistence_span"]
        ):
            raise ValueError("invalid profile persistence contract")
        if profile["base_scale"] <= 0:
            raise ValueError("profile base_scale must be positive")
        if len(profile["calibration"]) != len(self.score_feature_indices):
            raise ValueError("profile calibration length is invalid")
        profile["calibration"] = np.where(
            np.abs(profile["calibration"]) < 1e-12,
            1.0,
            profile["calibration"],
        )
        return profile

    def _validate_timing_contract(self):
        if not self.timing_contract.get("timestamp_required", False):
            raise ValueError("V4 requires timestamps for every sensor sample")
        minimum = float(self.timing_contract["minimum_interval_seconds"])
        maximum = float(self.timing_contract["maximum_interval_seconds"])
        nominal = float(self.timing_contract["nominal_interval_seconds"])
        if not (0 < minimum <= nominal <= maximum):
            raise ValueError("invalid V4 sampling interval contract")

    def reset(self):
        """Clear model state at a wafer, recipe, or monitored-step boundary."""
        maximum_window = max(item["window_size"] for item in self.profiles)
        self.feature_buffer = deque(maxlen=maximum_window)
        self.profile_history = {
            item["profile_id"]: deque(
                maxlen=item["persistence_span"])
            for item in self.profiles
        }
        self.previous_raw = None
        self.previous_timestamp = None
        self.sample_index = -1
        self.timeout_latched = False
        self.context = None

    def start_stream(self, wafer_id, recipe_id, equipment_id,
                     stream_instance_id):
        """Start a new explicitly identified process stream."""
        values = {
            "wafer_id": wafer_id,
            "recipe_id": recipe_id,
            "equipment_id": equipment_id,
            "stream_instance_id": stream_instance_id,
        }
        for name, value in values.items():
            if value is None or not str(value).strip():
                raise ValueError(f"{name} is required for a V4 stream")
            values[name] = str(value)
        self.reset()
        self.context = values
        return dict(self.context)

    def _validate_columns(self, columns):
        columns = list(columns)
        if len(set(columns)) != len(columns):
            raise ValueError("sensor columns must not contain duplicates")
        if columns != self.raw_sensor_names:
            missing = [
                name for name in self.raw_sensor_names if name not in columns]
            extra = [
                name for name in columns if name not in self.raw_sensor_names]
            raise ValueError(
                "sensor schema mismatch: expected ordered columns "
                f"{self.raw_sensor_names}, got {columns}; "
                f"missing={missing}, extra={extra}")
        return columns

    def _validate_timestamp(self, timestamp):
        if timestamp is None:
            raise ValueError("timestamp is required for every V4 sample")
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be finite and numeric") from exc
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite and numeric")
        interval = None
        if self.previous_timestamp is not None:
            interval = value - self.previous_timestamp
            if interval <= 0:
                raise ValueError("timestamps must be strictly increasing")
            minimum = float(
                self.timing_contract["minimum_interval_seconds"])
            maximum = float(
                self.timing_contract["maximum_interval_seconds"])
            if not minimum <= interval <= maximum:
                raise ValueError(
                    "sampling interval violates the V4 fixed-cadence "
                    f"contract: {interval:.9g}s not in "
                    f"[{minimum:.9g}, {maximum:.9g}]s")
        return value, interval

    def _make_feature_row(self, raw_values):
        if self.previous_raw is None:
            differences = np.zeros_like(raw_values)
        else:
            differences = raw_values - self.previous_raw
        elapsed_phase = np.asarray([
            max(self.sample_index, 0) /
            max(self.expected_cycle_samples - 1.0, 1.0)
        ], dtype=np.float32)
        unscaled = np.concatenate([
            raw_values, differences, elapsed_phase])
        return ((unscaled - self.feature_mean) /
                self.feature_std).astype(np.float32)

    @staticmethod
    def _score_errors(squared_errors, score_mode):
        if score_mode == "last":
            return squared_errors[-1]
        if score_mode == "mean":
            return squared_errors.mean(axis=0)
        return squared_errors.max(axis=0)

    @torch.no_grad()
    def update_ordered(self, values, columns, timestamp):
        columns = self._validate_columns(columns)
        values = list(values)
        if len(values) != len(columns):
            raise ValueError(
                f"expected {len(columns)} sensor values, got {len(values)}")
        return self.update(
            dict(zip(columns, values)), timestamp=timestamp)

    @torch.no_grad()
    def update(self, sample, timestamp):
        if self.context is None:
            raise RuntimeError("start_stream() must be called before update()")
        if self.timeout_latched:
            raise RuntimeError(
                "sensor timeout is latched; start_stream() is required")
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be an ordered sensor-name mapping")
        columns = self._validate_columns(sample.keys())
        raw_values = np.asarray(
            [sample[name] for name in columns], dtype=np.float32)
        if raw_values.shape != (len(self.raw_sensor_names),):
            raise ValueError("sensor sample shape is invalid")
        if not np.all(np.isfinite(raw_values)):
            raise ValueError("sensor sample contains NaN or infinity")
        timestamp, interval = self._validate_timestamp(timestamp)

        self.sample_index += 1
        feature_row = self._make_feature_row(raw_values)
        self.feature_buffer.append(feature_row)
        self.previous_raw = raw_values.copy()
        self.previous_timestamp = timestamp

        result = {
            **self.context,
            "sample_index": self.sample_index,
            "timestamp": timestamp,
            "sampling_interval_seconds": interval,
            "window_ready": False,
            "alarm_ready": False,
            "score": None,
            "threshold": self.threshold,
            "alarm": False,
            "trigger_profile_id": None,
            "top_evidence": [],
            "profiles": {},
            "raw_sensor_schema_hash": self.schema_hash,
            "model_version": self.model_version,
            "model_artifact_sha256": self.artifact_sha256,
            "deployment_manifest_sha256": self.manifest_sha256,
            "sensor_timeout_latched": self.timeout_latched,
        }

        by_window = {}
        for window_size in sorted({
                item["window_size"] for item in self.profiles}):
            if len(self.feature_buffer) < window_size:
                continue
            model_input = torch.from_numpy(np.stack(
                list(self.feature_buffer)[-window_size:]
            ).astype(np.float32)).unsqueeze(0)
            reconstruction = self.model(model_input)
            squared = (
                (reconstruction - model_input) ** 2
            ).detach().cpu().numpy()[0].astype(np.float64)
            by_window[window_size] = squared

        available = []
        for profile in self.profiles:
            profile_id = profile["profile_id"]
            profile_result = {
                "window_ready": False,
                "alarm_ready": False,
                "raw_score": None,
                "score": None,
                "alarm": False,
                "evidence": [],
            }
            if profile["window_size"] not in by_window:
                result["profiles"][profile_id] = profile_result
                continue
            profile_result["window_ready"] = True
            errors = self._score_errors(
                by_window[profile["window_size"]],
                profile["score_mode"],
            )[self.score_feature_indices]
            calibrated = errors / profile["calibration"]
            raw_score = float(calibrated.max())
            evidence = [
                {
                    "feature": self.feature_names[int(feature_index)],
                    "calibrated_error": float(value),
                }
                for feature_index, value in sorted(
                    zip(self.score_feature_indices, calibrated),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            history = self.profile_history[profile_id]
            history.append({
                "raw_score": raw_score,
                "evidence": evidence,
                "sample_index": self.sample_index,
            })
            profile_result["raw_score"] = raw_score
            if len(history) == profile["persistence_span"]:
                ranked = sorted(
                    history,
                    key=lambda item: item["raw_score"],
                    reverse=True,
                )
                selected = ranked[profile["persistence_required"] - 1]
                normalized_score = (
                    selected["raw_score"] / profile["base_scale"])
                profile_result.update({
                    "alarm_ready": True,
                    "score": float(normalized_score),
                    "alarm": bool(normalized_score > self.threshold),
                    "evidence": selected["evidence"],
                    "evidence_sample_index": selected["sample_index"],
                })
                available.append((
                    float(normalized_score), profile_id,
                    selected["evidence"],
                ))
            result["profiles"][profile_id] = profile_result

        if available:
            available.sort(key=lambda item: item[0], reverse=True)
            score, profile_id, evidence = available[0]
            result.update({
                "window_ready": True,
                "alarm_ready": True,
                "score": score,
                "alarm": bool(score > self.threshold),
                "trigger_profile_id": profile_id,
                "top_evidence": evidence[:3],
            })
        else:
            result["window_ready"] = bool(by_window)
        return result

    def liveness(self, current_timestamp):
        """Report a sensor timeout using the same timestamp domain as samples."""
        try:
            current_timestamp = float(current_timestamp)
        except (TypeError, ValueError) as exc:
            self.timeout_latched = True
            raise ValueError(
                "current_timestamp must be finite and numeric") from exc
        if not math.isfinite(current_timestamp):
            self.timeout_latched = True
            raise ValueError(
                "current_timestamp must be finite and numeric")
        if self.previous_timestamp is None:
            return {
                "healthy": False,
                "reason": "no_sample_received",
                "seconds_since_last_sample": None,
                "timeout_latched": self.timeout_latched,
            }
        elapsed = current_timestamp - self.previous_timestamp
        if elapsed < 0:
            raise ValueError("current_timestamp precedes the last sample")
        limit = float(self.timing_contract["sensor_timeout_seconds"])
        if elapsed > limit:
            self.timeout_latched = True
        return {
            "healthy": not self.timeout_latched,
            "reason": (
                None if not self.timeout_latched
                else "sensor_timeout_latched"),
            "seconds_since_last_sample": elapsed,
            "timeout_seconds": limit,
            "timeout_latched": self.timeout_latched,
        }


class JsonlAlarmRecorder:
    """Persist one rising-edge alarm with pre/post waveform evidence."""

    def __init__(self, path, pre_samples=64, post_samples=16):
        self.path = Path(path)
        self.pre_samples = int(pre_samples)
        self.post_samples = int(post_samples)
        if self.pre_samples < 1 or self.post_samples < 0:
            raise ValueError("invalid event context lengths")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(f"{self.path}.lock")
        try:
            self.lock_fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"event log is already locked: {self.lock_path}") from exc
        os.write(
            self.lock_fd,
            f"pid={os.getpid()}\n".encode("ascii"),
        )
        self.pre_buffer = deque(maxlen=self.pre_samples)
        self.active = []
        self.last_alarm = False
        self.stream_key = None
        self.storage_error_latched = False
        try:
            self.known_event_ids = self._load_known_event_ids()
        except Exception:
            self._release_lock()
            raise

    def _load_known_event_ids(self):
        identifiers = set()
        if not self.path.is_file():
            return identifiers
        for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                event_id = str(item["event_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"invalid event log record at line {line_number}") from exc
            if event_id in identifiers:
                raise RuntimeError(
                    f"duplicate event_id in event log: {event_id}")
            identifiers.add(event_id)
        return identifiers

    def _release_lock(self):
        if getattr(self, "lock_fd", None) is None:
            return
        os.close(self.lock_fd)
        self.lock_fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _sample_record(sample, result):
        return {
            "sample_index": int(result["sample_index"]),
            "timestamp": float(result["timestamp"]),
            "sensors": {
                str(name): float(value) for name, value in sample.items()
            },
            "score": result["score"],
            "alarm": bool(result["alarm"]),
            "trigger_profile_id": result["trigger_profile_id"],
            "top_evidence": result["top_evidence"],
        }

    def _write(self, event):
        event_id = str(event["event_id"])
        if event_id in self.known_event_ids:
            return
        payload = (
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        with self.path.open("ab+") as handle:
            handle.seek(0, os.SEEK_END)
            offset = handle.tell()
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                try:
                    handle.seek(offset)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    raise
        self.known_event_ids.add(event_id)

    @staticmethod
    def _finalized_event(event, status):
        finalized = {
            key: value for key, value in event.items()
            if key not in ("remaining_post_samples", "_pending_status")
        }
        finalized["status"] = status
        return finalized

    def update(self, sample, result):
        if self.storage_error_latched:
            raise RuntimeError(
                "event storage error is latched; "
                "retry_pending_writes() is required")
        stream_key = result["stream_instance_id"]
        if self.stream_key is not None and stream_key != self.stream_key:
            self._flush_active("truncated_on_stream_boundary")
            self.pre_buffer.clear()
            self.last_alarm = False
        self.stream_key = stream_key
        record = self._sample_record(sample, result)
        for item in list(self.active):
            if item.get("_pending_status") is not None:
                continue
            if (
                item["remaining_post_samples"] > 0 and
                record["sample_index"] > item["alarm_sample_index"]
            ):
                item["post_context"].append(record)
                item["remaining_post_samples"] -= 1
            if item["remaining_post_samples"] <= 0:
                item["_pending_status"] = "complete"

        rising_edge = bool(result["alarm"] and not self.last_alarm)
        if rising_edge:
            event_identity = (
                f"v4:{result['stream_instance_id']}:"
                f"{result['sample_index']}:"
                f"{result['deployment_manifest_sha256']}"
            )
            event = {
                "event_id": str(uuid.uuid5(
                    uuid.NAMESPACE_URL, event_identity)),
                "status": "collecting_post_context",
                "wafer_id": result["wafer_id"],
                "recipe_id": result["recipe_id"],
                "equipment_id": result["equipment_id"],
                "stream_instance_id": result["stream_instance_id"],
                "alarm_sample_index": int(result["sample_index"]),
                "alarm_timestamp": float(result["timestamp"]),
                "score": float(result["score"]),
                "threshold": float(result["threshold"]),
                "trigger_profile_id": result["trigger_profile_id"],
                "top_evidence": result["top_evidence"],
                "model_version": result["model_version"],
                "model_artifact_sha256": result[
                    "model_artifact_sha256"],
                "deployment_manifest_sha256": result[
                    "deployment_manifest_sha256"],
                "raw_sensor_schema_hash": result[
                    "raw_sensor_schema_hash"],
                "pre_context": list(self.pre_buffer) + [record],
                "post_context": [],
                "remaining_post_samples": self.post_samples,
            }
            if self.post_samples == 0:
                event["_pending_status"] = "complete"
            self.active.append(event)
        self.pre_buffer.append(record)
        self.last_alarm = bool(result["alarm"])
        self._persist_pending()
        return rising_edge

    def _persist_pending(self):
        for event in list(self.active):
            status = event.get("_pending_status")
            if status is None:
                continue
            try:
                self._write(self._finalized_event(event, status))
            except Exception:
                self.storage_error_latched = True
                raise
            self.active.remove(event)
        self.storage_error_latched = False

    def retry_pending_writes(self):
        """Retry latched event writes before accepting another sample."""
        self._persist_pending()

    def _flush_active(self, status):
        for event in self.active:
            event.setdefault("_pending_status", status)
        self._persist_pending()

    def flush(self):
        """Write incomplete events on shutdown without inventing future data."""
        self._flush_active("truncated_on_shutdown")

    def close(self):
        """Flush pending events and release the single-writer lock."""
        self.flush()
        self._release_lock()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a headered CSV through the V4 edge runtime")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timestamp-column", required=True)
    parser.add_argument("--wafer-id", required=True)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--equipment-id", required=True)
    parser.add_argument(
        "--stream-instance-id", required=True,
        help="Stable process-run ID for idempotent replay after restart")
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    detector = V4MultiscaleDetector.from_manifest(args.manifest)
    detector.start_stream(
        args.wafer_id, args.recipe_id, args.equipment_id,
        args.stream_instance_id)
    recorder = (
        JsonlAlarmRecorder(args.event_log)
        if args.event_log is not None else None
    )
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty and has no header") from exc
        if len(set(header)) != len(header):
            raise ValueError("CSV header contains duplicate columns")
        if args.timestamp_column not in header:
            raise ValueError(
                f"timestamp column {args.timestamp_column!r} is missing")
        sensor_columns = [
            name for name in header if name != args.timestamp_column]
        detector._validate_columns(sensor_columns)
        timestamp_index = header.index(args.timestamp_column)
        sensor_indices = [header.index(name) for name in sensor_columns]
        try:
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV row {line_number} has {len(row)} values; "
                        f"expected {len(header)}")
                timestamp = float(row[timestamp_index])
                values = [float(row[index]) for index in sensor_indices]
                sample = dict(zip(sensor_columns, values))
                result = detector.update(sample, timestamp)
                if recorder is not None:
                    recorder.update(sample, result)
                if args.show_all or result["alarm"]:
                    print(json.dumps(result, ensure_ascii=False))
        finally:
            if recorder is not None:
                recorder.close()


if __name__ == "__main__":
    main()
