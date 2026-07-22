# -*- coding: utf-8 -*-
"""One-time V3 evaluation on new seeds after model/profile selection."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder, sliding_window_error_summaries
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    projected_precision,
    sensor_error_score_curves,
    wilson_interval,
)
from v3_data import generate_set, load_statistics
from v3_features import transform_sequences


V3_DIR = OUTPUT_DIR / "v3"
ARTIFACT_PATH = V3_DIR / "sliding_window_lstm_ae_v3.pt"
STATS_PATH = V3_DIR / "sensor_stats_v3.json"
PROTOCOL_PATH = V3_DIR / "data_protocol_v3.json"
REPORT_PATH = V3_DIR / "final_holdout_v3.json"
EXPECTED_ARTIFACT_SHA256 = (
    "34bbd3ab1ac797ecd2311ef10267e58b991798bfb85602155ac653f2da57e502")
EXPECTED_STATS_SHA256 = (
    "79ba1e51556f6fd1755f645cc92535a35d52984a733cd24a282f22dc16df43e8")
NORMAL_HOLDOUT_SEED = 294001
ANOMALY_HOLDOUT_SEED = 295001
HOLDOUT_NORMALS = 10000
HOLDOUT_PER_ANOMALY = 1000
ANOMALY_NAMES = {
    1: "A: dynamic slew excursion",
    2: "B: oscillation",
    3: "C: drift",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked_inputs():
    actual = {
        "artifact": file_sha256(ARTIFACT_PATH),
        "statistics": file_sha256(STATS_PATH),
    }
    expected = {
        "artifact": EXPECTED_ARTIFACT_SHA256,
        "statistics": EXPECTED_STATS_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            f"locked input hash mismatch: expected={expected}, actual={actual}")
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite one-time holdout report: {REPORT_PATH}")
    return actual


def generate_holdout(statistics):
    normal = generate_set(
        np.random.default_rng(NORMAL_HOLDOUT_SEED), statistics,
        HOLDOUT_NORMALS, anomaly=0)
    anomaly = []
    labels = []
    metadata = []
    rng = np.random.default_rng(ANOMALY_HOLDOUT_SEED)
    for kind in ANOMALY_NAMES:
        sequences, items = generate_set(
            rng, statistics, HOLDOUT_PER_ANOMALY,
            anomaly=kind, with_metadata=True)
        anomaly.extend(sequences)
        labels.extend([kind] * len(sequences))
        metadata.extend(items)
    return normal, anomaly, np.asarray(labels, dtype=int), metadata


def model_from_artifact(artifact):
    model = SlidingWindowLSTMAutoEncoder(
        len(artifact["feature_spec"]["feature_names"]),
        artifact["hidden_size"], artifact["latent_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def profile_curves(model, sequences, profiles, score_indices):
    summaries_by_window = {
        window: sliding_window_error_summaries(model, sequences, window)
        for window in sorted({item["window_size"] for item in profiles})
    }
    output = []
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
        output.append({
            "profile": profile,
            "curves": persistent,
            "first_sample": (
                profile["window_size"] - 1 +
                profile["persistence_span"] - 1),
        })
    return output


def score_events(profile_outputs, labels, metadata, threshold):
    event_scores = []
    alert_indices = []
    pre_onset = []
    for sequence_index, (label, item_metadata) in enumerate(
            zip(labels, metadata)):
        profile_scores = []
        profile_alerts = []
        early = False
        onset = item_metadata.get("onset_index") if label > 0 else None
        for output in profile_outputs:
            profile = output["profile"]
            normalized = (
                output["curves"][sequence_index] /
                profile["base_scale"])
            sample_indices = (
                np.arange(len(normalized)) + output["first_sample"])
            eligible = np.ones(len(normalized), dtype=bool)
            if onset is not None:
                eligible = (
                    sample_indices - profile["persistence_span"] + 1
                ) >= int(onset)
                early = early or bool(np.any(
                    (normalized > threshold) & ~eligible))
            eligible_scores = normalized[eligible]
            profile_scores.append(
                float(eligible_scores.max())
                if len(eligible_scores) else -np.inf)
            crossings = np.flatnonzero(
                (normalized > threshold) & eligible)
            if len(crossings):
                profile_alerts.append(int(sample_indices[crossings[0]]))
        event_scores.append(max(profile_scores))
        alert_indices.append(min(profile_alerts) if profile_alerts else None)
        pre_onset.append(early)
    return (
        np.asarray(event_scores), alert_indices,
        np.asarray(pre_onset, dtype=bool))


def interval_record(successes, total):
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": successes / total,
        "wilson_95pct": [lower, upper],
    }


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def main():
    locked_hashes = verify_locked_inputs()
    statistics = load_statistics(STATS_PATH)
    artifact = torch.load(
        ARTIFACT_PATH, map_location="cpu", weights_only=False)
    normal, anomaly, anomaly_labels, anomaly_metadata = generate_holdout(
        statistics)
    all_sequences = normal + anomaly
    labels = np.concatenate([
        np.zeros(len(normal), dtype=int), anomaly_labels])
    metadata = [{} for _ in normal] + anomaly_metadata
    transformed = transform_sequences(
        all_sequences, artifact["feature_spec"])
    model = model_from_artifact(artifact)
    outputs = profile_curves(
        model, transformed, artifact["profiles"],
        artifact["feature_spec"]["score_feature_indices"])
    event_scores, alerts, pre_onset = score_events(
        outputs, labels, metadata, artifact["threshold"])
    predictions = event_scores > artifact["threshold"]
    metrics = binary_event_metrics(labels, predictions, event_scores)

    normal_mask = labels == 0
    anomaly_mask = labels > 0
    false_positives = int(predictions[normal_mask].sum())
    true_positives = int(predictions[anomaly_mask].sum())
    per_type = {}
    for kind, name in ANOMALY_NAMES.items():
        mask = labels == kind
        per_type[name] = interval_record(predictions[mask].sum(), mask.sum())

    detected_latencies = []
    detected_before_end = 0
    for index in np.flatnonzero(anomaly_mask & predictions):
        onset = metadata[index]["onset_index"]
        if alerts[index] is not None and onset is not None:
            detected_latencies.append(alerts[index] - onset)
            end = metadata[index]["end_index"]
            if end is not None and alerts[index] <= end:
                detected_before_end += 1

    code_paths = [
        Path(__file__), Path("v3_data.py"), Path("v3_features.py"),
        Path("models.py"), Path("online_evaluation.py"),
        Path("14_train_v3_window_lstm_ae.py"),
        Path("15_select_v3_multiscale.py"),
    ]
    report = {
        "status": "locked_final_holdout_no_further_tuning",
        "protocol": {
            "normal_count": HOLDOUT_NORMALS,
            "anomaly_count_per_type": HOLDOUT_PER_ANOMALY,
            "seeds": {
                "normal": NORMAL_HOLDOUT_SEED,
                "anomaly": ANOMALY_HOLDOUT_SEED,
            },
            "threshold_source": (
                "frozen selection-normal profile; no holdout recalibration"),
            "same_family_synthetic_holdout": True,
        },
        "operating_point": {
            "ensemble_threshold": artifact["threshold"],
            "selection_target_fpr": 0.001,
            "profiles": [{
                key: profile[key]
                for key in (
                    "window_size", "score_mode", "persistence_required",
                    "persistence_span", "base_scale")
            } for profile in artifact["profiles"]],
        },
        "confusion_counts": {
            "true_positive": true_positives,
            "false_negative": int(anomaly_mask.sum() - true_positives),
            "false_positive": false_positives,
            "true_negative": int(normal_mask.sum() - false_positives),
        },
        "metrics": {
            **metrics,
            "accuracy": float(np.mean(
                predictions == anomaly_mask)),
            "fpr_interval": interval_record(
                false_positives, normal_mask.sum()),
            "recall_interval": interval_record(
                true_positives, anomaly_mask.sum()),
            "per_type_recall": per_type,
            "projected_precision": {
                "at_1pct_anomaly_prevalence": projected_precision(
                    metrics["recall"], metrics["fpr"], 0.01),
                "at_0_1pct_anomaly_prevalence": projected_precision(
                    metrics["recall"], metrics["fpr"], 0.001),
            },
            "pre_onset_crossing_rate": float(np.mean(
                pre_onset[anomaly_mask])),
            "detected_latency_samples": {
                "median": float(np.median(detected_latencies)),
                "p95": float(np.percentile(detected_latencies, 95)),
            },
            "detected_before_injection_end_rate": (
                detected_before_end / true_positives),
        },
        "provenance": {
            "artifact_sha256": locked_hashes["artifact"],
            "statistics_sha256": locked_hashes["statistics"],
            "data_protocol_sha256": file_sha256(PROTOCOL_PATH),
            "code_sha256": {
                str(path): file_sha256(path) for path in code_paths
            },
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_porcelain": git_value("status", "--porcelain"),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "device": "cpu",
            },
        },
        "limitations": [
            "The locked holdout is generated by the same V3 generator family.",
            "The LAM data cadence is about 1 Hz; sub-second faults are not observable.",
            "Real faulty wafers were not used to tune weights or thresholds.",
            "Raspberry Pi 5 latency and long-duration false-alert rate are not measured yet.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report["confusion_counts"], indent=2))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Locked holdout report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
