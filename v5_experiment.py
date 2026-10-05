# -*- coding: utf-8 -*-
"""Leakage-safe V5 training, profile selection, and acceptance helpers."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import torch

from models import sliding_window_error_summaries
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    calibrate_sensor_errors,
    event_decisions,
    projected_precision,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)


WINDOW_SIZES = (4, 8, 16, 32, 64)
SCORE_MODES = ("last", "mean", "max", "delta_mean", "delta_max")
PERSISTENCE_OPTIONS = ((1, 1), (2, 3), (3, 5))
TARGET_FPR = 0.001
TOP_PER_TYPE = 10
ANOMALY_NAMES = {
    1: "A: per-sample difference excursion",
    2: "B: oscillation",
    3: "C: drift",
}
ACCEPTANCE_TARGETS = {
    "maximum_fpr": 0.001,
    "minimum_recall": 0.75,
    "minimum_f1": 0.85,
    "minimum_type_a_recall": 0.60,
    "minimum_type_b_recall": 0.70,
    "minimum_type_c_recall": 0.80,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_groups(feature_spec):
    """Return predeclared raw/delta score groups from a V3-style spec."""
    raw_names = list(feature_spec["raw_sensor_names"])
    names = list(feature_spec["feature_names"])
    sensor_count = len(raw_names)
    expected = raw_names + [f"delta::{name}" for name in raw_names]
    if names[:2 * sensor_count] != expected:
        raise ValueError("feature spec does not have contiguous raw/delta fields")
    return {
        "raw": list(range(sensor_count)),
        "delta": list(range(sensor_count, 2 * sensor_count)),
        "raw_delta": list(range(2 * sensor_count)),
    }


def weighted_reconstruction_loss(model, batch, raw_sensor_count,
                                 delta_feature_weight=2.0,
                                 phase_feature_weight=0.25,
                                 temporal_difference_weight=0.5):
    """Reconstruct normal values while emphasizing causal delta features."""
    reconstruction = model(batch)
    n_features = batch.shape[-1]
    if n_features != 2 * raw_sensor_count + 1:
        raise ValueError("V5 expects raw, delta, and one elapsed-phase feature")
    weights = torch.ones(n_features, dtype=batch.dtype, device=batch.device)
    weights[raw_sensor_count:2 * raw_sensor_count] = delta_feature_weight
    weights[-1] = phase_feature_weight
    value_error = (reconstruction - batch) ** 2
    value_loss = (value_error * weights).sum() / (
        value_error.shape[0] * value_error.shape[1] * weights.sum())
    reconstructed_difference = torch.diff(reconstruction, dim=1)
    observed_difference = torch.diff(batch, dim=1)
    difference_error = (reconstructed_difference - observed_difference) ** 2
    temporal_loss = (difference_error * weights).sum() / (
        difference_error.shape[0] * difference_error.shape[1] * weights.sum())
    return value_loss + temporal_difference_weight * temporal_loss


def profile_key(profile):
    return (
        profile["window_size"], profile["score_mode"],
        profile["feature_group"], profile["persistence_required"],
        profile["persistence_span"],
    )


def stored_profile(profile):
    keys = (
        "window_size", "score_mode", "feature_group", "feature_indices",
        "persistence_required", "persistence_span", "base_scale",
        "validation_fpr", "validation_macro_recall",
        "validation_per_type_recall", "calibration",
    )
    result = {key: profile[key] for key in keys}
    result["calibration"] = np.asarray(result["calibration"]).tolist()
    return result


def enumerate_profiles(model, normal, anomaly, anomaly_labels,
                       anomaly_metadata, feature_spec,
                       target_fpr=TARGET_FPR):
    """Enumerate only on selection cohorts; no locked holdout is accepted."""
    groups = feature_groups(feature_spec)
    labels = np.concatenate([
        np.zeros(len(normal), dtype=int), np.asarray(anomaly_labels, dtype=int)
    ])
    metadata = [{} for _ in normal] + list(anomaly_metadata)
    profiles = []
    for window_size in WINDOW_SIZES:
        normal_summaries = sliding_window_error_summaries(
            model, normal, window_size)
        anomaly_summaries = sliding_window_error_summaries(
            model, anomaly, window_size)
        for score_mode in SCORE_MODES:
            for group_name, indices in groups.items():
                normal_errors = [
                    values[:, indices]
                    for values in normal_summaries[score_mode]
                ]
                anomaly_errors = [
                    values[:, indices]
                    for values in anomaly_summaries[score_mode]
                ]
                calibration = calibrate_sensor_errors(normal_errors)
                raw_curves = sensor_error_score_curves(
                    normal_errors + anomaly_errors, calibration)
                for required, span in PERSISTENCE_OPTIONS:
                    curves = apply_persistence(raw_curves, required, span)
                    first_sample = window_size - 1 + span - 1
                    _, _, _, initial_scores = event_decisions(
                        curves, np.inf, first_sample, labels, metadata,
                        evidence_span=span)
                    base_scale = threshold_for_target_fpr(
                        initial_scores[:len(normal)], target_fpr)
                    predictions, _, pre_onset, event_scores = event_decisions(
                        curves, base_scale, first_sample, labels, metadata,
                        evidence_span=span)
                    metrics = binary_event_metrics(
                        labels, predictions, event_scores)
                    per_type = {
                        name: float(np.mean(predictions[labels == kind]))
                        for kind, name in ANOMALY_NAMES.items()
                    }
                    profiles.append({
                        "window_size": window_size,
                        "score_mode": score_mode,
                        "feature_group": group_name,
                        "feature_indices": list(indices),
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
    return profiles, labels


def select_ensemble(profiles, labels, normal_count,
                    target_fpr=TARGET_FPR):
    """Select a maximum-normalized multiscale ensemble on selection data."""
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
        keys = set()
        for profile in selected:
            key = profile_key(profile)
            if key not in keys:
                keys.add(key)
                unique.append(profile)
        ensemble_key = tuple(sorted(keys))
        if ensemble_key in seen:
            continue
        seen.add(ensemble_key)
        event_scores = np.stack([
            profile["event_scores"] / profile["base_scale"]
            for profile in unique
        ]).max(axis=0)
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
            "min_type_recall": float(min(per_type.values())),
            "projected_precision": {
                "at_1pct_prevalence": projected_precision(
                    metrics["recall"], metrics["fpr"], 0.01),
                "at_0_1pct_prevalence": projected_precision(
                    metrics["recall"], metrics["fpr"], 0.001),
            },
        }
        gate = acceptance_gate(metrics, per_type)
        target_ratios = (
            metrics["recall"] / ACCEPTANCE_TARGETS["minimum_recall"],
            metrics["f1"] / ACCEPTANCE_TARGETS["minimum_f1"],
            per_type[ANOMALY_NAMES[1]] /
            ACCEPTANCE_TARGETS["minimum_type_a_recall"],
            per_type[ANOMALY_NAMES[2]] /
            ACCEPTANCE_TARGETS["minimum_type_b_recall"],
            per_type[ANOMALY_NAMES[3]] /
            ACCEPTANCE_TARGETS["minimum_type_c_recall"],
        )
        rank = (
            gate["passed"], sum(gate["checks"].values()),
            min(target_ratios), candidate["macro_recall"],
            candidate["min_type_recall"], metrics["recall"],
            -metrics["fpr"], metrics["f1"], -len(unique),
        )
        if best is None or rank > best["rank"]:
            best = {**candidate, "rank": rank}
    if best is None:
        raise RuntimeError("no V5 ensemble candidate was generated")
    return best


def acceptance_gate(metrics, per_type, targets=ACCEPTANCE_TARGETS):
    checks = {
        "fpr": metrics["fpr"] <= targets["maximum_fpr"],
        "recall": metrics["recall"] >= targets["minimum_recall"],
        "f1": metrics["f1"] >= targets["minimum_f1"],
        "type_a_recall": per_type[ANOMALY_NAMES[1]] >= (
            targets["minimum_type_a_recall"]),
        "type_b_recall": per_type[ANOMALY_NAMES[2]] >= (
            targets["minimum_type_b_recall"]),
        "type_c_recall": per_type[ANOMALY_NAMES[3]] >= (
            targets["minimum_type_c_recall"]),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "targets": dict(targets)}


def profile_score_outputs(model, sequences, profiles):
    """Return normalized causal curves and their first source indices."""
    summaries_by_window = {
        window: sliding_window_error_summaries(model, sequences, window)
        for window in sorted({item["window_size"] for item in profiles})
    }
    outputs = []
    for profile in profiles:
        indices = list(profile["feature_indices"])
        errors = [
            values[:, indices]
            for values in summaries_by_window[profile["window_size"]][
                profile["score_mode"]]
        ]
        raw = sensor_error_score_curves(errors, profile["calibration"])
        persistent = apply_persistence(
            raw, profile["persistence_required"],
            profile["persistence_span"])
        outputs.append({
            "profile": profile,
            "curves": [curve / profile["base_scale"] for curve in persistent],
            "first_sample": (
                profile["window_size"] - 1 +
                profile["persistence_span"] - 1),
        })
    return outputs


def normal_ensemble_event_scores(model, sequences, profiles):
    outputs = profile_score_outputs(model, sequences, profiles)
    return np.stack([
        np.asarray([curve.max() for curve in output["curves"]])
        for output in outputs
    ]).max(axis=0)


def score_profile_outputs(outputs, labels, metadata, threshold):
    """Score events and return earliest causal alert across V5 profiles."""
    labels = np.asarray(labels, dtype=int)
    if len(metadata) != len(labels):
        raise ValueError("metadata and labels must align")
    event_scores = []
    alert_indices = []
    pre_onset = []
    for sequence_index, (label, item_metadata) in enumerate(
            zip(labels, metadata)):
        profile_scores = []
        profile_alerts = []
        early = False
        onset = item_metadata.get("onset_index") if label > 0 else None
        for output in outputs:
            profile = output["profile"]
            curve = np.asarray(
                output["curves"][sequence_index], dtype=np.float64)
            sample_indices = np.arange(len(curve)) + output["first_sample"]
            eligible = np.ones(len(curve), dtype=bool)
            if onset is not None:
                eligible = (
                    sample_indices - profile["persistence_span"] + 1
                ) >= int(onset)
                early = early or bool(np.any(
                    (curve > threshold) & ~eligible))
            eligible_scores = curve[eligible]
            profile_scores.append(
                float(eligible_scores.max())
                if len(eligible_scores) else -np.inf)
            crossings = np.flatnonzero((curve > threshold) & eligible)
            if len(crossings):
                profile_alerts.append(int(sample_indices[crossings[0]]))
        event_scores.append(max(profile_scores))
        alert_indices.append(min(profile_alerts) if profile_alerts else None)
        pre_onset.append(early)
    return (
        np.asarray(event_scores, dtype=np.float64), alert_indices,
        np.asarray(pre_onset, dtype=bool),
    )
