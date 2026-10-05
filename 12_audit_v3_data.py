# -*- coding: utf-8 -*-
"""Audit V2 synthetic observability and failure modes before V3 training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder, sliding_window_errors
from online_evaluation import (
    apply_persistence,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)


V3_DIR = OUTPUT_DIR / "v3"
TARGET_FPR = 0.01
ANOMALY_NAMES = {
    1: "A: fast transient",
    2: "B: oscillation",
    3: "C: drift",
}


def load_data():
    raw = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    return {
        "train": list(raw["X_train"]),
        "val": list(raw["X_val"]),
        "val_anom": list(raw["X_val_anom"]),
        "val_labels": np.asarray(raw["y_val_anom"], dtype=int),
        "val_metadata": [json.loads(str(item)) for item in raw["val_metadata"]],
        "test": list(raw["X_test"]),
        "test_labels": np.asarray(raw["y_test"], dtype=int),
        "sensor_names": [str(item) for item in raw["sensor_names"]],
    }


def difference_scales(train):
    differences = np.concatenate([
        np.diff(np.asarray(sequence, dtype=np.float64), axis=0)
        for sequence in train
    ])
    center = np.median(differences, axis=0)
    absolute = np.abs(differences - center)
    scale = np.quantile(absolute, 0.75, axis=0)
    return center, np.where(scale < 1e-9, 1.0, scale)


def difference_event_scores(sequences, center, scale):
    scores = []
    peak_locations = []
    for sequence in sequences:
        difference = np.diff(np.asarray(sequence, dtype=np.float64), axis=0)
        point_scores = np.max(np.abs(difference - center) / scale, axis=1)
        peak = int(np.argmax(point_scores))
        scores.append(float(point_scores[peak]))
        peak_locations.append(peak + 1)
    return np.asarray(scores), np.asarray(peak_locations)


def recalls(scores, labels, threshold):
    return {
        ANOMALY_NAMES[label]: float(np.mean(scores[labels == label] > threshold))
        for label in ANOMALY_NAMES
    }


def v2_false_positive_audit(data):
    checkpoint = torch.load(
        OUTPUT_DIR / "sliding_window_lstm_ae.pt",
        map_location="cpu", weights_only=False)
    model = SlidingWindowLSTMAutoEncoder(
        len(checkpoint["mean"]), checkpoint["hidden_size"],
        checkpoint["latent_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normal = [
        np.asarray(sequence, dtype=np.float32)
        for sequence, label in zip(data["test"], data["test_labels"])
        if label == 0
    ]
    normalized = [
        (sequence - checkpoint["mean"]) / checkpoint["std"]
        for sequence in normal
    ]
    errors = sliding_window_errors(
        model, normalized, checkpoint["window_size"],
        checkpoint["score_mode"])
    raw_curves = sensor_error_score_curves(errors, checkpoint["calib"])
    curves = apply_persistence(
        raw_curves, checkpoint["persistence_required"],
        checkpoint["persistence_span"])
    manifest = json.loads(
        (OUTPUT_DIR / "edge_deployment_manifest.json").read_text(
            encoding="utf-8"))
    threshold = manifest["payload"]["profiles"][
        "final_calibration_fpr_1pct"]["threshold"]
    first_sample = (
        checkpoint["window_size"] - 1 + checkpoint["persistence_span"] - 1)
    false_positive_indices = []
    alert_progress = []
    peak_progress = []
    for index, (curve, sequence) in enumerate(zip(curves, normal)):
        crossings = np.flatnonzero(curve > threshold)
        if not len(crossings):
            continue
        false_positive_indices.append(index)
        alert_sample = int(first_sample + crossings[0])
        peak_sample = int(first_sample + np.argmax(curve))
        denominator = max(len(sequence) - 1, 1)
        alert_progress.append(alert_sample / denominator)
        peak_progress.append(peak_sample / denominator)
    return {
        "threshold": float(threshold),
        "normal_count": len(normal),
        "false_positive_count": len(false_positive_indices),
        "false_positive_indices": false_positive_indices,
        "first_alert_progress": alert_progress,
        "peak_score_progress": peak_progress,
        "median_first_alert_progress": (
            float(np.median(alert_progress)) if alert_progress else None),
        "median_peak_score_progress": (
            float(np.median(peak_progress)) if peak_progress else None),
    }


def main():
    V3_DIR.mkdir(exist_ok=True)
    data = load_data()
    center, scale = difference_scales(data["train"])
    normal_scores, _ = difference_event_scores(data["val"], center, scale)
    anomaly_scores, anomaly_peaks = difference_event_scores(
        data["val_anom"], center, scale)
    threshold = threshold_for_target_fpr(normal_scores, TARGET_FPR)
    observed_fpr = float(np.mean(normal_scores > threshold))

    type_a = [
        metadata for metadata, label in zip(
            data["val_metadata"], data["val_labels"]) if label == 1
    ]
    type_a_spans = [
        int(item["end_index"] - item["onset_index"] + 1) for item in type_a
    ]
    type_a_peaks = anomaly_peaks[data["val_labels"] == 1]
    report = {
        "audit_scope": (
            "V2 data and model failure analysis performed before V3 design"),
        "data_contract": {
            "train_normal": len(data["train"]),
            "validation_normal": len(data["val"]),
            "validation_anomaly": len(data["val_anom"]),
            "sensor_names": data["sensor_names"],
            "length_range": [
                int(min(map(len, data["train"]))),
                int(max(map(len, data["train"]))),
            ],
        },
        "first_difference_oracle": {
            "purpose": (
                "Observability check only; this is not a selected V3 model."),
            "target_validation_fpr": TARGET_FPR,
            "observed_validation_fpr": observed_fpr,
            "threshold": float(threshold),
            "per_type_recall": recalls(
                anomaly_scores, data["val_labels"], threshold),
            "sensor_difference_center": center.tolist(),
            "sensor_difference_scale_q75": scale.tolist(),
        },
        "type_a_timing": {
            "count": len(type_a),
            "median_anomaly_span_samples": float(np.median(type_a_spans)),
            "median_oracle_peak_sample": float(np.median(type_a_peaks)),
            "fraction_ended_before_v2_window_ready": float(np.mean([
                item["end_index"] < 31 for item in type_a
            ])),
            "v2_window_ready_sample": 31,
        },
        "v2_development_false_positives": v2_false_positive_audit(data),
        "known_generator_limitations": [
            "Normal wafers share one mean profile per sensor.",
            "Residual noise is generated independently per sensor; real cross-sensor covariance is not modeled.",
            "Normal timing variation and recipe/step context are not modeled.",
            "Type A starts at sample zero, so no pre-anomaly context exists.",
            "The same generator family is used across development and holdout cohorts.",
        ],
    }
    output = V3_DIR / "data_audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Audit saved: {output}")


if __name__ == "__main__":
    main()
