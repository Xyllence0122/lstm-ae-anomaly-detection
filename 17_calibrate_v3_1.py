# -*- coding: utf-8 -*-
"""Recalibrate the frozen V3 profile on a larger normal-only cohort."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder, sliding_window_error_summaries
from online_evaluation import (
    apply_persistence,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)
from v3_data import generate_set, load_statistics
from v3_features import transform_sequences


V3_DIR = OUTPUT_DIR / "v3"
SOURCE_PATH = V3_DIR / "sliding_window_lstm_ae_v3.pt"
STATS_PATH = V3_DIR / "sensor_stats_v3.json"
OUTPUT_PATH = V3_DIR / "sliding_window_lstm_ae_v3_1.pt"
REPORT_PATH = V3_DIR / "deployment_calibration_v3_1.json"
EXPECTED_SOURCE_SHA256 = (
    "34bbd3ab1ac797ecd2311ef10267e58b991798bfb85602155ac653f2da57e502")
EXPECTED_STATS_SHA256 = (
    "79ba1e51556f6fd1755f645cc92535a35d52984a733cd24a282f22dc16df43e8")
CALIBRATION_SEED = 296001
CALIBRATION_NORMALS = 10000
TARGET_FPR = 0.001


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def verify_inputs():
    actual = {
        "source": file_sha256(SOURCE_PATH),
        "statistics": file_sha256(STATS_PATH),
    }
    expected = {
        "source": EXPECTED_SOURCE_SHA256,
        "statistics": EXPECTED_STATS_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            f"calibration input hash mismatch: expected={expected}, actual={actual}")
    if OUTPUT_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite V3.1 calibration outputs")
    return actual


def normal_event_scores(model, sequences, profiles, score_indices):
    summaries_by_window = {
        window: sliding_window_error_summaries(model, sequences, window)
        for window in sorted({item["window_size"] for item in profiles})
    }
    profile_scores = []
    for profile in profiles:
        errors = [
            values[:, score_indices]
            for values in summaries_by_window[profile["window_size"]][
                profile["score_mode"]]
        ]
        raw = sensor_error_score_curves(errors, profile["calibration"])
        persistent = apply_persistence(
            raw, profile["persistence_required"],
            profile["persistence_span"])
        profile_scores.append(np.asarray([
            curve.max() / profile["base_scale"] for curve in persistent
        ]))
    return np.stack(profile_scores).max(axis=0)


def main():
    input_hashes = verify_inputs()
    artifact = torch.load(
        SOURCE_PATH, map_location="cpu", weights_only=False)
    statistics = load_statistics(STATS_PATH)
    normal = generate_set(
        np.random.default_rng(CALIBRATION_SEED), statistics,
        CALIBRATION_NORMALS, anomaly=0)
    transformed = transform_sequences(normal, artifact["feature_spec"])
    model = SlidingWindowLSTMAutoEncoder(
        len(artifact["feature_spec"]["feature_names"]),
        artifact["hidden_size"], artifact["latent_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    event_scores = normal_event_scores(
        model, transformed, artifact["profiles"],
        artifact["feature_spec"]["score_feature_indices"])
    threshold = threshold_for_target_fpr(event_scores, TARGET_FPR)
    observed_fpr = float(np.mean(event_scores > threshold))

    calibrated = dict(artifact)
    calibrated["threshold"] = threshold
    calibrated["calibration_v3_1"] = {
        "normal_only": True,
        "normal_count": CALIBRATION_NORMALS,
        "seed": CALIBRATION_SEED,
        "target_fpr": TARGET_FPR,
        "observed_fpr": observed_fpr,
        "source_artifact_sha256": input_hashes["source"],
    }
    torch.save(calibrated, OUTPUT_PATH)
    report = {
        "status": "normal_only_threshold_recalibration",
        "model_weights_changed": False,
        "profiles_changed": False,
        "anomaly_labels_accessed": False,
        "normal_count": CALIBRATION_NORMALS,
        "seed": CALIBRATION_SEED,
        "target_fpr": TARGET_FPR,
        "allowed_false_positives": int(np.floor(
            TARGET_FPR * CALIBRATION_NORMALS)),
        "observed_false_positives": int(np.sum(event_scores > threshold)),
        "observed_fpr": observed_fpr,
        "old_threshold": artifact["threshold"],
        "new_threshold": threshold,
        "source_artifact_sha256": input_hashes["source"],
        "statistics_sha256": input_hashes["statistics"],
        "calibrated_artifact_sha256": file_sha256(OUTPUT_PATH),
        "code_sha256": file_sha256(Path(__file__)),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_before_outputs": git_value("status", "--porcelain"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "reason": (
            "The initial 2,000-normal selection cohort produced only two "
            "allowed tail events and did not generalize to the first locked "
            "holdout. V3.1 uses 10,000 new normal-only events and requires a "
            "new artifact, new holdout seeds, and a new report."),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
