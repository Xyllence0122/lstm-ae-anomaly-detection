# -*- coding: utf-8 -*-
"""One-time paired V5/V3.2 evaluation on preregistered locked seeds."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder
from online_evaluation import (
    binary_event_metrics,
    projected_precision,
    wilson_interval,
)
from v3_data import generate_set, load_statistics
from v3_features import transform_sequences
from v4_hashing import normalized_text_sha256
from v5_experiment import (
    ACCEPTANCE_TARGETS,
    ANOMALY_NAMES,
    acceptance_gate,
    file_sha256,
    profile_score_outputs,
    score_profile_outputs,
)


PROJECT_DIR = Path(__file__).resolve().parent
V5_DIR = OUTPUT_DIR / "v5"
V5_ARTIFACT_PATH = V5_DIR / "sliding_window_lstm_ae_v5.pt"
V3_2_ARTIFACT_PATH = OUTPUT_DIR / "v3" / "sliding_window_lstm_ae_v3_2.pt"
CALIBRATION_REPORT_PATH = V5_DIR / "normal_calibration_v5.json"
SELECTION_REPORT_PATH = (
    V5_DIR / "experiment_dynamic_grouped_final" / "selection_report.json")
PROTOCOL_PATH = V5_DIR / "selection_protocol_v5.json"
STATS_PATH = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
REPORT_PATH = V5_DIR / "locked_holdout_v5.json"
EXPECTED_HASHES = {
    "v5_artifact": (
        "a59a6c1ae1e828bce72dbec396806f36344dcaf326e9cc04df0bf560258085f4"),
    "v3_2_artifact": (
        "e3ab0ba9954114bf4b8db5842838be88160ef7b65a308c4c8ffe6c0603a56b5d"),
    "calibration_report": (
        "1cb4752bae3b17fc3dfca2ffa1a7c06a4d5768f57a6c039cf80babb6b454d7f4"),
    "selection_report": (
        "f26104072f632fc63458e4223278fd85901627ab088e4e71fc99c0e85187afea"),
    "protocol": (
        "ff1498a63c9d804ae7cdb4ea8bd10227e393366dd87887bcdf12a33d911bb309"),
    "statistics_normalized": (
        "5e9b3aa0cbec720abb4871c771b11584066fb1fab235767ed92e63557bf2f76b"),
}
NORMAL_HOLDOUT_SEED = 730101
ANOMALY_HOLDOUT_SEED = 730102
HOLDOUT_NORMALS = 10000
HOLDOUT_PER_ANOMALY = 1000


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def interval_record(successes, total):
    successes, total = int(successes), int(total)
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total,
        "wilson_95pct": [lower, upper],
    }


def verify_locked_inputs():
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite one-time V5 holdout: {REPORT_PATH}")
    actual = {
        "v5_artifact": file_sha256(V5_ARTIFACT_PATH),
        "v3_2_artifact": file_sha256(V3_2_ARTIFACT_PATH),
        "calibration_report": file_sha256(CALIBRATION_REPORT_PATH),
        "selection_report": file_sha256(SELECTION_REPORT_PATH),
        "protocol": file_sha256(PROTOCOL_PATH),
        "statistics_normalized": normalized_text_sha256(STATS_PATH),
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(
            f"V5 locked input mismatch: {actual} != {EXPECTED_HASHES}")
    if git_value("status", "--porcelain"):
        raise RuntimeError("V5 locked evaluator requires a clean worktree")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    selection = json.loads(
        SELECTION_REPORT_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(
        CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
    reserved = protocol["reserved_unopened_seeds"]
    expected_seeds = {
        "normal_only_calibration": 720101,
        "locked_holdout_normal": NORMAL_HOLDOUT_SEED,
        "locked_holdout_anomaly": ANOMALY_HOLDOUT_SEED,
    }
    if reserved != expected_seeds:
        raise RuntimeError("V5 reserved seeds differ from preregistration")
    if not selection["release"]["gate"]["passed"]:
        raise RuntimeError("V5 selection gate was not passed")
    if calibration["seed"] != reserved["normal_only_calibration"]:
        raise RuntimeError("V5 calibration used the wrong reserved seed")
    if calibration["anomaly_labels_accessed"]:
        raise RuntimeError("V5 calibration must be normal-only")
    return actual


def generate_holdout(statistics):
    normal = generate_set(
        np.random.default_rng(NORMAL_HOLDOUT_SEED), statistics,
        HOLDOUT_NORMALS, anomaly=0)
    anomalies = []
    labels = []
    metadata = []
    anomaly_rng = np.random.default_rng(ANOMALY_HOLDOUT_SEED)
    for kind in ANOMALY_NAMES:
        sequences, items = generate_set(
            anomaly_rng, statistics, HOLDOUT_PER_ANOMALY,
            anomaly=kind, with_metadata=True)
        anomalies.extend(sequences)
        labels.extend([kind] * len(sequences))
        metadata.extend(items)
    return normal, anomalies, np.asarray(labels, dtype=int), metadata


def model_from_artifact(artifact):
    model = SlidingWindowLSTMAutoEncoder(
        len(artifact["feature_spec"]["feature_names"]),
        artifact["hidden_size"], artifact["latent_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def compatible_profiles(artifact):
    profiles = []
    default_indices = artifact["feature_spec"].get("score_feature_indices")
    for source in artifact["profiles"]:
        profile = dict(source)
        if "feature_indices" not in profile:
            if default_indices is None:
                raise ValueError("artifact profile has no score feature indices")
            profile["feature_indices"] = list(default_indices)
            profile["feature_group"] = "legacy_raw_delta"
        profiles.append(profile)
    return profiles


def evaluate_artifact(artifact, all_sequences, labels, metadata):
    transformed = transform_sequences(
        all_sequences, artifact["feature_spec"])
    model = model_from_artifact(artifact)
    profiles = compatible_profiles(artifact)
    outputs = profile_score_outputs(model, transformed, profiles)
    event_scores, alerts, pre_onset = score_profile_outputs(
        outputs, labels, metadata, artifact["threshold"])
    predictions = event_scores > artifact["threshold"]
    metrics = binary_event_metrics(labels, predictions, event_scores)
    normal_mask = labels == 0
    anomaly_mask = labels > 0
    false_positives = int(predictions[normal_mask].sum())
    true_positives = int(predictions[anomaly_mask].sum())
    per_type = {}
    per_type_rates = {}
    for kind, name in ANOMALY_NAMES.items():
        mask = labels == kind
        record = interval_record(predictions[mask].sum(), mask.sum())
        per_type[name] = record
        per_type_rates[name] = record["rate"]
    detected_latencies = []
    detected_before_end = 0
    for index in np.flatnonzero(anomaly_mask & predictions):
        onset = metadata[index].get("onset_index")
        end = metadata[index].get("end_index")
        if alerts[index] is not None and onset is not None:
            detected_latencies.append(alerts[index] - int(onset))
            if end is not None and alerts[index] <= int(end):
                detected_before_end += 1
    return {
        "operating_point": {
            "threshold": artifact["threshold"],
            "profiles": [{
                key: profile[key]
                for key in (
                    "window_size", "score_mode", "feature_group",
                    "feature_indices", "persistence_required",
                    "persistence_span", "base_scale")
            } for profile in profiles],
        },
        "confusion_counts": {
            "true_positive": true_positives,
            "false_negative": int(anomaly_mask.sum() - true_positives),
            "false_positive": false_positives,
            "true_negative": int(normal_mask.sum() - false_positives),
        },
        "metrics": {
            **metrics,
            "accuracy": float(np.mean(predictions == anomaly_mask)),
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
                "count": len(detected_latencies),
                "median": (float(np.median(detected_latencies))
                           if detected_latencies else None),
                "p95": (float(np.percentile(detected_latencies, 95))
                        if detected_latencies else None),
            },
            "detected_before_injection_end_rate": (
                detected_before_end / true_positives
                if true_positives else None),
        },
        "acceptance_gate": acceptance_gate(metrics, per_type_rates),
        "predictions": predictions,
    }


def comparison_record(v5_result, baseline_result, labels):
    v5_predictions = v5_result["predictions"]
    baseline_predictions = baseline_result["predictions"]
    anomaly_mask = labels > 0
    normal_mask = labels == 0
    per_type_gain = {}
    for kind, name in ANOMALY_NAMES.items():
        mask = labels == kind
        per_type_gain[name] = float(
            np.mean(v5_predictions[mask]) -
            np.mean(baseline_predictions[mask]))
    return {
        "recall_absolute_gain": (
            v5_result["metrics"]["recall"] -
            baseline_result["metrics"]["recall"]),
        "f1_absolute_gain": (
            v5_result["metrics"]["f1"] -
            baseline_result["metrics"]["f1"]),
        "fpr_absolute_change": (
            v5_result["metrics"]["fpr"] -
            baseline_result["metrics"]["fpr"]),
        "per_type_recall_absolute_gain": per_type_gain,
        "paired_anomaly_events": {
            "v5_only_detected": int(np.sum(
                v5_predictions[anomaly_mask] &
                ~baseline_predictions[anomaly_mask])),
            "v3_2_only_detected": int(np.sum(
                ~v5_predictions[anomaly_mask] &
                baseline_predictions[anomaly_mask])),
            "both_detected": int(np.sum(
                v5_predictions[anomaly_mask] &
                baseline_predictions[anomaly_mask])),
            "neither_detected": int(np.sum(
                ~v5_predictions[anomaly_mask] &
                ~baseline_predictions[anomaly_mask])),
        },
        "paired_normal_false_alarms": {
            "v5_only": int(np.sum(
                v5_predictions[normal_mask] &
                ~baseline_predictions[normal_mask])),
            "v3_2_only": int(np.sum(
                ~v5_predictions[normal_mask] &
                baseline_predictions[normal_mask])),
            "both": int(np.sum(
                v5_predictions[normal_mask] &
                baseline_predictions[normal_mask])),
        },
    }


def strip_internal(result):
    return {key: value for key, value in result.items()
            if key != "predictions"}


def main():
    input_hashes = verify_locked_inputs()
    statistics = load_statistics(STATS_PATH)
    normal, anomaly, anomaly_labels, anomaly_metadata = generate_holdout(
        statistics)
    all_sequences = normal + anomaly
    labels = np.concatenate([
        np.zeros(len(normal), dtype=int), anomaly_labels])
    metadata = [{} for _ in normal] + anomaly_metadata
    v5_artifact = torch.load(
        V5_ARTIFACT_PATH, map_location="cpu", weights_only=False)
    baseline_artifact = torch.load(
        V3_2_ARTIFACT_PATH, map_location="cpu", weights_only=False)
    v5_result = evaluate_artifact(
        v5_artifact, all_sequences, labels, metadata)
    baseline_result = evaluate_artifact(
        baseline_artifact, all_sequences, labels, metadata)
    report = {
        "status": "v5_one_time_locked_holdout_no_further_tuning",
        "protocol": {
            "normal_count": HOLDOUT_NORMALS,
            "anomaly_count_per_type": HOLDOUT_PER_ANOMALY,
            "seeds": {
                "normal": NORMAL_HOLDOUT_SEED,
                "anomaly": ANOMALY_HOLDOUT_SEED,
            },
            "same_family_synthetic_holdout": True,
            "paired_frozen_model_comparison": True,
            "no_holdout_threshold_or_weight_tuning": True,
            "acceptance_targets": ACCEPTANCE_TARGETS,
        },
        "v5": strip_internal(v5_result),
        "v3_2_frozen_baseline": strip_internal(baseline_result),
        "paired_comparison": comparison_record(
            v5_result, baseline_result, labels),
        "provenance": {
            "input_hashes": input_hashes,
            "code_sha256": {
                "locked_evaluator": file_sha256(Path(__file__)),
                "experiment_helpers": file_sha256(
                    PROJECT_DIR / "v5_experiment.py"),
                "generator": file_sha256(PROJECT_DIR / "v3_data.py"),
                "features": file_sha256(PROJECT_DIR / "v3_features.py"),
                "models": file_sha256(PROJECT_DIR / "models.py"),
                "evaluation": file_sha256(
                    PROJECT_DIR / "online_evaluation.py"),
            },
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_before_report": git_value("status", "--porcelain"),
            "resolved_command": [
                sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "device": "cpu",
            },
        },
        "limitations": [
            "V5 and V3.2 are evaluated on new seeds but the same synthetic "
            "generator family used for model development.",
            "Real statistics inform synthetic generation; this is not an "
            "independent external real-data validation.",
            "Approximately 1 Hz source cadence cannot support sub-second "
            "fault-observation claims.",
            "No yield loss, equipment damage, or accident outcome labels are "
            "available.",
            "This report evaluates detection, not production safety approval.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "v5": report["v5"],
        "v3_2_frozen_baseline": report["v3_2_frozen_baseline"],
        "paired_comparison": report["paired_comparison"],
        "report_path": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
