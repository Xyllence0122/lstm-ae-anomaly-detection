# -*- coding: utf-8 -*-
"""Calibrate the frozen V5 selection candidate on new normal-only events."""
from __future__ import annotations

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
from v4_hashing import normalized_text_sha256
from v5_experiment import file_sha256, normal_ensemble_event_scores


PROJECT_DIR = Path(__file__).resolve().parent
V5_DIR = OUTPUT_DIR / "v5"
SOURCE_PATH = V5_DIR / "experiment_dynamic_grouped_final" / "candidate.pt"
SELECTION_REPORT_PATH = (
    V5_DIR / "experiment_dynamic_grouped_final" / "selection_report.json")
PROTOCOL_PATH = V5_DIR / "selection_protocol_v5.json"
STATS_PATH = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
OUTPUT_PATH = V5_DIR / "sliding_window_lstm_ae_v5.pt"
REPORT_PATH = V5_DIR / "normal_calibration_v5.json"
EXPECTED_SOURCE_SHA256 = (
    "5d064e1c0ab94377aa83f353fbee6729fcdc87f8e597c7fc3051b687910c7f5b")
EXPECTED_SELECTION_REPORT_SHA256 = (
    "f26104072f632fc63458e4223278fd85901627ab088e4e71fc99c0e85187afea")
EXPECTED_PROTOCOL_SHA256 = (
    "ff1498a63c9d804ae7cdb4ea8bd10227e393366dd87887bcdf12a33d911bb309")
EXPECTED_STATS_NORMALIZED_SHA256 = (
    "5e9b3aa0cbec720abb4871c771b11584066fb1fab235767ed92e63557bf2f76b")
CALIBRATION_SEED = 720101
CALIBRATION_NORMALS = 10000
TARGET_FPR = 0.001


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def verify_inputs():
    if OUTPUT_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite V5 calibration outputs")
    actual = {
        "source": file_sha256(SOURCE_PATH),
        "selection_report": file_sha256(SELECTION_REPORT_PATH),
        "protocol": file_sha256(PROTOCOL_PATH),
        "statistics": normalized_text_sha256(STATS_PATH),
    }
    expected = {
        "source": EXPECTED_SOURCE_SHA256,
        "selection_report": EXPECTED_SELECTION_REPORT_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "statistics": EXPECTED_STATS_NORMALIZED_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            f"V5 calibration input mismatch: {actual} != {expected}")
    selection = json.loads(
        SELECTION_REPORT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not selection["release"]["gate"]["passed"]:
        raise RuntimeError("V5 selection gate did not pass")
    reserved = protocol["reserved_unopened_seeds"]["normal_only_calibration"]
    if reserved != CALIBRATION_SEED:
        raise RuntimeError("V5 calibration seed differs from preregistration")
    return actual


def main():
    input_hashes = verify_inputs()
    git_status_before_outputs = git_value("status", "--porcelain")
    if git_status_before_outputs:
        raise RuntimeError(
            "V5 calibration requires a clean preregistered worktree: "
            f"{git_status_before_outputs}")
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
    event_scores = normal_ensemble_event_scores(
        model, transformed, artifact["profiles"])
    threshold = threshold_for_target_fpr(event_scores, TARGET_FPR)
    false_positives = int(np.sum(event_scores > threshold))
    observed_fpr = false_positives / CALIBRATION_NORMALS
    provenance = {
        "normal_only": True,
        "normal_count": CALIBRATION_NORMALS,
        "seed": CALIBRATION_SEED,
        "target_fpr": TARGET_FPR,
        "allowed_false_positives": int(np.floor(
            TARGET_FPR * CALIBRATION_NORMALS)),
        "observed_false_positives": false_positives,
        "observed_fpr": observed_fpr,
        "source_artifact_sha256": input_hashes["source"],
        "selection_report_sha256": input_hashes["selection_report"],
    }
    calibrated = dict(artifact)
    calibrated["threshold"] = threshold
    calibrated["calibration_provenance"] = provenance
    torch.save(calibrated, OUTPUT_PATH)
    report = {
        "status": "v5_normal_only_calibration_locked_profiles_and_weights",
        "model_weights_changed": False,
        "profiles_changed": False,
        "anomaly_labels_accessed": False,
        **provenance,
        "old_selection_threshold": artifact["threshold"],
        "new_calibrated_threshold": threshold,
        "input_hashes": input_hashes,
        "calibrated_artifact_sha256": file_sha256(OUTPUT_PATH),
        "resolved_command": [
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "code_sha256": {
            "calibrator": file_sha256(Path(__file__)),
            "experiment_helpers": file_sha256(
                PROJECT_DIR / "v5_experiment.py"),
        },
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_before_outputs": git_status_before_outputs,
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
