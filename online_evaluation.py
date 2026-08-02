# -*- coding: utf-8 -*-
"""Shared causal event-scoring utilities for online anomaly detectors."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def calibrate_sensor_errors(normal_errors):
    """Return positive per-sensor scale factors from normal point errors."""
    if not normal_errors:
        raise ValueError("normal_errors must not be empty")
    stacked = np.concatenate(normal_errors, axis=0).astype(np.float64)
    calib = stacked.mean(axis=0)
    return np.where(np.abs(calib) < 1e-12, 1.0, calib)


def sensor_error_score_curves(error_list, calib=None):
    """Convert per-sensor errors to one max-calibrated score per sample."""
    denominator = None
    if calib is not None:
        denominator = np.asarray(calib, dtype=np.float64)
        denominator = np.where(np.abs(denominator) < 1e-12, 1.0,
                               denominator)
    curves = []
    for errors in error_list:
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("each error item must have shape (T, F)")
        if denominator is not None:
            values = values / denominator
        curves.append(values.max(axis=1))
    return curves


def persistence_score_curve(scores, required=1, span=1):
    """Score curve equivalent to requiring k-of-n threshold crossings.

    The ``required``-th largest score in each trailing ``span`` is greater than
    a threshold exactly when at least ``required`` samples in that span exceed
    the threshold. Only complete trailing spans are emitted.
    """
    values = np.asarray(scores, dtype=np.float64)
    required, span = int(required), int(span)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if span < 1 or required < 1 or required > span:
        raise ValueError("persistence must satisfy 1 <= required <= span")
    if len(values) < span:
        raise ValueError("persistence span exceeds score-curve length")
    windows = np.lib.stride_tricks.sliding_window_view(values, span)
    return np.partition(windows, -required, axis=1)[:, -required]


def apply_persistence(curves, required=1, span=1):
    return [persistence_score_curve(curve, required, span)
            for curve in curves]


def threshold_for_target_fpr(normal_event_scores, target_fpr):
    """Choose an empirical event threshold with FPR no greater than target.

    Alarms use a strict ``score > threshold`` comparison. The order statistic
    therefore permits at most ``floor(target_fpr * n)`` normal events above
    the returned threshold.
    """
    values = np.sort(np.asarray(normal_event_scores, dtype=np.float64))[::-1]
    if values.ndim != 1 or not len(values):
        raise ValueError("normal_event_scores must be a non-empty vector")
    target_fpr = float(target_fpr)
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must satisfy 0 <= target_fpr < 1")
    allowed_false_positives = int(np.floor(target_fpr * len(values)))
    if allowed_false_positives == 0:
        return float(values[0])
    if allowed_false_positives >= len(values):
        return float(np.nextafter(values[-1], -np.inf))
    return float(values[allowed_false_positives])


def event_decisions(curves, threshold, first_sample_indices, labels,
                    metadata=None, evidence_span=1):
    """Make event decisions while excluding synthetic pre-onset crossings."""
    labels = np.asarray(labels)
    evidence_span = int(evidence_span)
    if evidence_span < 1:
        raise ValueError("evidence_span must be at least 1")
    if len(curves) != len(labels):
        raise ValueError("curves and labels must align")
    if metadata is not None and len(metadata) != len(curves):
        raise ValueError("metadata and curves must align")

    if np.isscalar(first_sample_indices):
        first_sample_indices = [int(first_sample_indices)] * len(curves)
    if len(first_sample_indices) != len(curves):
        raise ValueError("first_sample_indices and curves must align")

    predictions = np.zeros(len(curves), dtype=int)
    alerts = []
    pre_onset = []
    event_scores = []
    for index, (curve, first_index) in enumerate(
            zip(curves, first_sample_indices)):
        curve = np.asarray(curve, dtype=np.float64)
        sample_indices = np.arange(len(curve)) + int(first_index)
        onset = None
        if labels[index] > 0 and metadata is not None:
            onset = metadata[index].get("onset_index")
        eligible = np.ones(len(curve), dtype=bool)
        early = False
        if onset is not None:
            onset = int(onset)
            # A persistence window overlapping the injection boundary still
            # contains pre-onset evidence. Count a true detection only once
            # the complete evidence span lies at or after synthetic onset.
            eligible = (sample_indices - evidence_span + 1) >= onset
            early = bool(np.any(
                (curve > threshold) & (sample_indices < onset)))
        crossings = np.flatnonzero((curve > threshold) & eligible)
        if len(crossings):
            first = int(crossings[0])
            predictions[index] = 1
            alerts.append(int(sample_indices[first]))
        else:
            alerts.append(None)
        pre_onset.append(early)
        eligible_scores = curve[eligible]
        event_scores.append(
            float(eligible_scores.max()) if len(eligible_scores) else -np.inf)
    return (
        predictions,
        alerts,
        np.asarray(pre_onset, dtype=bool),
        np.asarray(event_scores, dtype=np.float64),
    )


def binary_event_metrics(labels, predictions, event_scores=None):
    truth = (np.asarray(labels) > 0).astype(int)
    predictions = np.asarray(predictions).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0)
    normal = truth == 0
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(predictions[normal].mean()) if normal.any() else None,
    }
    if event_scores is not None and len(np.unique(truth)) == 2:
        result["auc"] = float(roc_auc_score(truth, event_scores))
    return result


def projected_precision(recall, fpr, prevalence):
    """Project alert precision for an assumed deployment anomaly prevalence."""
    recall, fpr, prevalence = map(float, (recall, fpr, prevalence))
    numerator = recall * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    return float(numerator / denominator) if denominator > 0 else None


def wilson_interval(successes, total, z=1.959963984540054):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    successes, total = int(successes), int(total)
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total and total >= 1")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * np.sqrt(
            proportion * (1 - proportion) / total +
            z**2 / (4 * total**2)
        ) / denominator
    )
    return [float(center - margin), float(center + margin)]
