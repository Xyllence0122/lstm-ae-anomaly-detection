# -*- coding: utf-8 -*-
"""V5 named-schema multiscale streaming runtime with profile feature groups."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from config import PROJECT_DIR
from deployment_manifest import sensor_schema_hash
from v4_edge_runtime import (
    JsonlAlarmRecorder,
    V4MultiscaleDetector,
    _path_digest,
    _path_size,
    _sidecar_path,
)
from v4_hashing import normalized_text_sha256


DEFAULT_MANIFEST = (
    PROJECT_DIR / "outputs" / "v5" / "deployment_manifest_v5.json"
)


def load_v5_manifest(path=DEFAULT_MANIFEST, verify_provenance=True):
    """Load a V5 manifest and verify every bound artifact and source."""
    path = Path(path).resolve()
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"V5 deployment manifest or sidecar is missing: {path}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) < 3 or fields[1] != path.name:
        raise ValueError("invalid V5 manifest sidecar")
    if fields[2] != "normalized_text_sha256":
        raise ValueError("unsupported V5 manifest sidecar hash mode")
    actual_manifest_hash = normalized_text_sha256(path)
    if fields[0] != actual_manifest_hash:
        raise ValueError(
            "V5 deployment manifest hash mismatch: "
            f"expected {fields[0]}, got {actual_manifest_hash}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("manifest_version") != 5:
        raise ValueError("unsupported V5 deployment manifest version")
    project_root = path.parents[2]
    for artifact_id, record in document["artifacts"].items():
        artifact_path = project_root / record["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"missing V5 artifact {artifact_id}: {artifact_path}")
        actual = _path_digest(artifact_path, record["hash_mode"])
        if actual != record["sha256"]:
            raise ValueError(
                f"V5 artifact hash mismatch for {artifact_id}: "
                f"expected {record['sha256']}, got {actual}")
        if _path_size(artifact_path, record["hash_mode"]) != record["bytes"]:
            raise ValueError(f"V5 artifact size mismatch for {artifact_id}")
    if verify_provenance:
        for source_id, record in document["source_provenance"].items():
            source_path = project_root / record["path"]
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"missing V5 provenance source {source_id}: {source_path}")
            actual = _path_digest(source_path, record["hash_mode"])
            if actual != record["sha256"]:
                raise ValueError(
                    f"V5 provenance hash mismatch for {source_id}: "
                    f"expected {record['sha256']}, got {actual}")
    return document, path, actual_manifest_hash


class V5MultiscaleDetector(V4MultiscaleDetector):
    """V4 safety contracts plus V5 profile-specific feature selection."""

    @classmethod
    def from_manifest(cls, path=DEFAULT_MANIFEST, verify_provenance=True):
        document, manifest_path, manifest_hash = load_v5_manifest(
            path, verify_provenance=verify_provenance)
        project_root = manifest_path.parents[2]
        model_record = document["artifacts"]["torchscript_model"]
        model = torch.jit.load(
            str(project_root / model_record["path"]), map_location="cpu")
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
            raise ValueError("V5 runtime sensor schema does not match manifest")
        return detector

    def _validated_profile(self, index, profile):
        profile = dict(profile)
        required = {
            "window_size", "score_mode", "feature_indices",
            "persistence_required", "persistence_span", "base_scale",
            "calibration",
        }
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"V5 profile {index} is missing {missing}")
        profile.setdefault("profile_id", f"profile_{index}")
        profile["window_size"] = int(profile["window_size"])
        profile["persistence_required"] = int(
            profile["persistence_required"])
        profile["persistence_span"] = int(profile["persistence_span"])
        profile["base_scale"] = float(profile["base_scale"])
        profile["feature_indices"] = np.asarray(
            profile["feature_indices"], dtype=int)
        profile["calibration"] = np.asarray(
            profile["calibration"], dtype=np.float64)
        if profile["score_mode"] not in ("last", "mean", "max"):
            raise ValueError(
                "V5 runtime supports last, mean, and max score modes")
        if profile["window_size"] < 2:
            raise ValueError("V5 profile window_size must be at least 2")
        if not (
            1 <= profile["persistence_required"] <=
            profile["persistence_span"]
        ):
            raise ValueError("invalid V5 profile persistence contract")
        indices = profile["feature_indices"]
        if (
            indices.ndim != 1 or not len(indices) or
            len(np.unique(indices)) != len(indices) or
            np.any(indices < 0) or np.any(indices >= len(self.feature_names))
        ):
            raise ValueError("V5 profile feature indices are invalid")
        if len(profile["calibration"]) != len(indices):
            raise ValueError("V5 profile calibration length is invalid")
        if profile["base_scale"] <= 0:
            raise ValueError("V5 profile base_scale must be positive")
        profile["calibration"] = np.where(
            np.abs(profile["calibration"]) < 1e-12,
            1.0, profile["calibration"])
        return profile

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
            by_window[window_size] = (
                (reconstruction - model_input) ** 2
            ).detach().cpu().numpy()[0].astype(np.float64)

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
            indices = profile["feature_indices"]
            errors = self._score_errors(
                by_window[profile["window_size"]],
                profile["score_mode"])[indices]
            calibrated = errors / profile["calibration"]
            raw_score = float(calibrated.max())
            evidence = [{
                "feature": self.feature_names[int(feature_index)],
                "calibrated_error": float(value),
            } for feature_index, value in sorted(
                zip(indices, calibrated),
                key=lambda item: item[1], reverse=True)]
            history = self.profile_history[profile_id]
            history.append({
                "raw_score": raw_score,
                "evidence": evidence,
                "sample_index": self.sample_index,
            })
            profile_result["raw_score"] = raw_score
            if len(history) == profile["persistence_span"]:
                ranked = sorted(
                    history, key=lambda item: item["raw_score"], reverse=True)
                selected = ranked[profile["persistence_required"] - 1]
                normalized = selected["raw_score"] / profile["base_scale"]
                profile_result.update({
                    "alarm_ready": True,
                    "score": float(normalized),
                    "alarm": bool(normalized > self.threshold),
                    "evidence": selected["evidence"],
                    "evidence_sample_index": selected["sample_index"],
                })
                available.append((
                    float(normalized), profile_id, selected["evidence"]))
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


__all__ = [
    "DEFAULT_MANIFEST", "JsonlAlarmRecorder", "V5MultiscaleDetector",
    "load_v5_manifest",
]
