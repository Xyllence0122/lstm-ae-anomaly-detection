# -*- coding: utf-8 -*-
"""Select a normal-calibrated multi-scale profile without holdout access."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder, sliding_window_error_summaries
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    calibrate_sensor_errors,
    event_decisions,
    projected_precision,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)
from v3_features import transform_sequences


V3_DIR = OUTPUT_DIR / "v3"
DATA_PATH = V3_DIR / "selection_data_v3.npz"
WINDOW_SIZES = (8, 16, 32, 64)
SCORE_MODES = ("mean", "max", "delta_mean", "delta_max")
PERSISTENCE_OPTIONS = ((1, 1), (2, 3), (3, 5))
DEFAULT_TARGET_VALIDATION_FPR = 0.005
TOP_PER_TYPE = 8
ANOMALY_NAMES = {
    1: "A: per-sample difference excursion",
    2: "B: oscillation",
    3: "C: drift",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(
            V3_DIR / "experiment_observable_contract_quick" / "candidate.pt"))
    parser.add_argument("--experiment", default="multiscale_quick")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument(
        "--target-fpr", type=float, default=DEFAULT_TARGET_VALIDATION_FPR)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1:
        parser.error("--target-fpr must be between zero and one")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection_data(path):
    raw = np.load(path, allow_pickle=True)
    return {
        "normal": list(raw["X_val"]),
        "anomaly": list(raw["X_val_anom"]),
        "labels": np.asarray(raw["y_val_anom"], dtype=int),
        "metadata": [
            json.loads(str(item)) for item in raw["val_metadata"]],
    }


def profile_summary(profile):
    return {
        key: profile[key]
        for key in (
            "window_size", "score_mode", "persistence_required",
            "persistence_span", "base_scale", "validation_fpr",
            "validation_macro_recall", "validation_per_type_recall",
        )
    }


def profile_key(profile):
    return (
        profile["window_size"], profile["score_mode"],
        profile["persistence_required"], profile["persistence_span"],
    )


def calibrate_profile_events(curves, first_sample, labels, metadata,
                             evidence_span, target_fpr, normal_count):
    """Calibrate on normal event scores, then compute real pre-onset flags."""
    _, _, _, event_scores = event_decisions(
        curves, np.inf, first_sample, labels, metadata,
        evidence_span=evidence_span)
    base_scale = threshold_for_target_fpr(
        event_scores[:normal_count], target_fpr)
    predictions, _, pre_onset, event_scores = event_decisions(
        curves, base_scale, first_sample, labels, metadata,
        evidence_span=evidence_span)
    return base_scale, predictions, pre_onset, event_scores


def enumerate_profiles(model, normal, anomaly, labels, metadata,
                       score_indices, target_fpr):
    all_labels = np.concatenate([
        np.zeros(len(normal), dtype=int), labels])
    all_metadata = [{} for _ in normal] + metadata
    profiles = []
    for window_size in WINDOW_SIZES:
        normal_summaries = sliding_window_error_summaries(
            model, normal, window_size)
        anomaly_summaries = sliding_window_error_summaries(
            model, anomaly, window_size)
        for score_mode in SCORE_MODES:
            normal_errors = [
                values[:, score_indices]
                for values in normal_summaries[score_mode]
            ]
            anomaly_errors = [
                values[:, score_indices]
                for values in anomaly_summaries[score_mode]
            ]
            calibration = calibrate_sensor_errors(normal_errors)
            normal_raw = sensor_error_score_curves(
                normal_errors, calibration)
            anomaly_raw = sensor_error_score_curves(
                anomaly_errors, calibration)
            for required, span in PERSISTENCE_OPTIONS:
                curves = apply_persistence(
                    normal_raw + anomaly_raw, required, span)
                first_sample = window_size - 1 + span - 1
                base_scale, predictions, pre_onset, event_scores = (
                    calibrate_profile_events(
                        curves, first_sample, all_labels, all_metadata,
                        span, target_fpr, len(normal)))
                metrics = binary_event_metrics(
                    all_labels, predictions, event_scores)
                per_type = {
                    name: float(np.mean(predictions[all_labels == kind]))
                    for kind, name in ANOMALY_NAMES.items()
                }
                profiles.append({
                    "window_size": window_size,
                    "score_mode": score_mode,
                    "persistence_required": required,
                    "persistence_span": span,
                    "base_scale": float(base_scale),
                    "calibration": calibration,
                    "event_scores": event_scores,
                    "validation_fpr": metrics["fpr"],
                    "validation_macro_recall": float(np.mean(
                        list(per_type.values()))),
                    "validation_per_type_recall": per_type,
                    "validation_pre_onset_rate": float(np.mean(
                        pre_onset[len(normal):])),
                })
    return profiles, all_labels


def select_ensemble(profiles, labels, normal_count, target_fpr):
    shortlists = []
    for name in ANOMALY_NAMES.values():
        ranked = sorted(profiles, key=lambda item: (
            item["validation_per_type_recall"][name],
            item["validation_macro_recall"],
            -item["validation_fpr"],
        ), reverse=True)
        shortlists.append(ranked[:TOP_PER_TYPE])

    best = None
    seen = set()
    for selected in itertools.product(*shortlists):
        unique = []
        keys = []
        for profile in selected:
            key = profile_key(profile)
            if key not in keys:
                unique.append(profile)
                keys.append(key)
        ensemble_key = tuple(sorted(keys))
        if ensemble_key in seen:
            continue
        seen.add(ensemble_key)
        normalized = np.stack([
            profile["event_scores"] / profile["base_scale"]
            for profile in unique
        ])
        event_scores = normalized.max(axis=0)
        threshold = threshold_for_target_fpr(
            event_scores[:normal_count], target_fpr)
        predictions = event_scores > threshold
        metrics = binary_event_metrics(labels, predictions, event_scores)
        per_type = {
            name: float(np.mean(predictions[labels == kind]))
            for kind, name in ANOMALY_NAMES.items()
        }
        candidate = {
            "profiles": unique,
            "threshold": float(threshold),
            "metrics": metrics,
            "per_type": per_type,
            "macro_recall": float(np.mean(list(per_type.values()))),
            "min_type_recall": min(per_type.values()),
        }
        rank = (
            candidate["macro_recall"], candidate["min_type_recall"],
            metrics["recall"], -metrics["fpr"], metrics["f1"],
            -len(unique),
        )
        if best is None or rank > best["rank"]:
            candidate["rank"] = rank
            best = candidate
    return best


def main():
    args = parse_args()
    source_path = Path(args.source)
    data_path = Path(args.data_path)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    current_data_hash = file_sha256(data_path)
    if source["data_sha256"] != current_data_hash:
        raise RuntimeError(
            "source checkpoint and current selection data hashes differ")
    data = load_selection_data(data_path)
    normal = transform_sequences(data["normal"], source["feature_spec"])
    anomaly = transform_sequences(data["anomaly"], source["feature_spec"])
    model = SlidingWindowLSTMAutoEncoder(
        len(source["feature_spec"]["feature_names"]),
        source["hidden_size"], source["latent_size"])
    model.load_state_dict(source["state_dict"])
    model.eval()
    profiles, labels = enumerate_profiles(
        model, normal, anomaly, data["labels"], data["metadata"],
        source["feature_spec"]["score_feature_indices"], args.target_fpr)
    best = select_ensemble(
        profiles, labels, len(normal), args.target_fpr)

    experiment_dir = V3_DIR / f"experiment_{args.experiment}"
    experiment_dir.mkdir(exist_ok=True)
    artifact_path = experiment_dir / "candidate.pt"
    stored_profiles = [{
        **profile_summary(profile),
        "calibration": profile["calibration"],
    } for profile in best["profiles"]]
    torch.save({
        "state_dict": source["state_dict"],
        "feature_spec": source["feature_spec"],
        "profiles": stored_profiles,
        "threshold": best["threshold"],
        "hidden_size": source["hidden_size"],
        "latent_size": source["latent_size"],
        "training_seed": source["seed"],
        "dynamic_weight": source["dynamic_weight"],
        "selection_data_sha256": current_data_hash,
        "source_checkpoint_sha256": file_sha256(source_path),
        "selection_data_path": str(data_path),
        "selection_data_sha256": current_data_hash,
    }, artifact_path)
    metrics = {
        **best["metrics"],
        "macro_recall": best["macro_recall"],
        "minimum_type_recall": best["min_type_recall"],
        "per_type_recall": best["per_type"],
        "projected_precision_at_1pct_prevalence": projected_precision(
            best["metrics"]["recall"], best["metrics"]["fpr"], 0.01),
        "projected_precision_at_0_1pct_prevalence": projected_precision(
            best["metrics"]["recall"], best["metrics"]["fpr"], 0.001),
    }
    report = {
        "protocol": {
            "selection_data_only": True,
            "holdout_access": "none",
            "target_total_validation_fpr": args.target_fpr,
            "profile_search": (
                "top validation profile per anomaly type, followed by joint "
                "normal-only calibration of the maximum normalized score"),
        },
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": file_sha256(source_path),
        "profiles": [profile_summary(item) for item in best["profiles"]],
        "ensemble_threshold": best["threshold"],
        "validation_metrics": metrics,
        "candidate_sha256": file_sha256(artifact_path),
    }
    report_path = experiment_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if args.publish:
        published_path = V3_DIR / "sliding_window_lstm_ae_v3.pt"
        published_path.write_bytes(artifact_path.read_bytes())
        print(f"Published frozen V3 checkpoint: {published_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
