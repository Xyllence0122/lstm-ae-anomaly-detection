# -*- coding: utf-8 -*-
"""Normal-only threshold calibration for the frozen V3.2 profile."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder
from online_evaluation import threshold_for_target_fpr
from v3_data import generate_set, load_statistics
from v3_features import transform_sequences


PROJECT_DIR = Path(__file__).resolve().parent
V3_DIR = OUTPUT_DIR / "v3"
SOURCE_PATH = V3_DIR / "experiment_multiscale_final_v3_2" / "candidate.pt"
STATS_PATH = V3_DIR / "sensor_stats_v3_2.json"
OUTPUT_PATH = V3_DIR / "sliding_window_lstm_ae_v3_2.pt"
REPORT_PATH = V3_DIR / "deployment_calibration_v3_2.json"
EXPECTED_SOURCE_SHA256 = (
    "93c229250ec980f461c409ee962b24badc899ea4a303a25d712614f8b0b891bc")
EXPECTED_STATS_SHA256 = (
    "79118aea76f3d3bf8eb41c2b5d62e20ff56ab5adc073e00cc419dbbfb77b2aa6")
CALIBRATION_SEED = 396001
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


def load_v3_1_calibration_helpers():
    spec = importlib.util.spec_from_file_location(
        "v3_1_calibration_helpers", PROJECT_DIR / "17_calibrate_v3_1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for path in (OUTPUT_PATH, REPORT_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V3.2 output: {path}")
    actual_hashes = {
        "source": file_sha256(SOURCE_PATH),
        "statistics": file_sha256(STATS_PATH),
    }
    expected_hashes = {
        "source": EXPECTED_SOURCE_SHA256,
        "statistics": EXPECTED_STATS_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"V3.2 input hash mismatch: {actual_hashes} != {expected_hashes}")

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
    helpers = load_v3_1_calibration_helpers()
    event_scores = helpers.normal_event_scores(
        model, transformed, artifact["profiles"],
        artifact["feature_spec"]["score_feature_indices"])
    threshold = threshold_for_target_fpr(event_scores, TARGET_FPR)
    observed_false_positives = int(np.sum(event_scores > threshold))
    observed_fpr = observed_false_positives / CALIBRATION_NORMALS

    provenance = {
        "version": "3.2",
        "normal_only": True,
        "normal_count": CALIBRATION_NORMALS,
        "seed": CALIBRATION_SEED,
        "target_fpr": TARGET_FPR,
        "observed_fpr": observed_fpr,
        "source_artifact_sha256": actual_hashes["source"],
    }
    calibrated = dict(artifact)
    calibrated["threshold"] = threshold
    calibrated["calibration_provenance"] = provenance
    torch.save(calibrated, OUTPUT_PATH)
    report = {
        "status": "v3_2_normal_only_threshold_calibration",
        "model_weights_changed": False,
        "profiles_changed": False,
        "anomaly_labels_accessed": False,
        "normal_count": CALIBRATION_NORMALS,
        "seed": CALIBRATION_SEED,
        "target_fpr": TARGET_FPR,
        "allowed_false_positives": int(np.floor(
            TARGET_FPR * CALIBRATION_NORMALS)),
        "observed_false_positives": observed_false_positives,
        "observed_fpr": observed_fpr,
        "old_threshold": artifact["threshold"],
        "new_threshold": threshold,
        "source_artifact_sha256": actual_hashes["source"],
        "statistics_sha256": actual_hashes["statistics"],
        "calibrated_artifact_sha256": file_sha256(OUTPUT_PATH),
        "resolved_command": [
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "code_sha256": file_sha256(Path(__file__)),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_before_report": git_value("status", "--porcelain"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
