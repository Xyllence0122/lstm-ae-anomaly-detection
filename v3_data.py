# -*- coding: utf-8 -*-
"""Chronology-corrected real statistics and semi-physical V3 generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io

from config import (
    DATA_MAT,
    MIN_WAFER_LEN,
    PROCESS_STEPS,
    SELECTED_SENSORS,
    SENSOR_IDX,
    STEP_COL,
)


PROFILE_GRID = 101
SPLIT_SEED = 123
STATS_FRACTION = 0.6
ANOMALY_NAMES = {
    0: "Normal",
    1: "A: per-sample difference excursion",
    2: "B: process oscillation",
    3: "C: gradual drift",
}


def intervention_support(delta, affected, tolerance=1e-12):
    """Return exact pre-quantization bounds of an injected intervention."""
    values = np.asarray(delta, dtype=np.float64)
    by_sensor = []
    all_indices = []
    for feature in affected:
        indices = np.flatnonzero(np.abs(values[:, feature]) > tolerance)
        if not len(indices):
            continue
        all_indices.extend(indices.tolist())
        by_sensor.append({
            "sensor_position": int(feature),
            "first_index": int(indices[0]),
            "last_index": int(indices[-1]),
        })
    return {
        "definition": (
            "indices where the pre-quantization intervention changes the "
            "generated normal baseline by more than 1e-12"),
        "by_sensor": by_sensor,
        "onset_index": min(all_indices) if all_indices else None,
        "end_index": max(all_indices) if all_indices else None,
    }


def load_lam_data(path=DATA_MAT):
    mat = scipy.io.loadmat(path)
    lam = mat["LAMDATA"][0, 0]
    normal = [
        lam["calibration"][index, 0]
        for index in range(lam["calibration"].shape[0])
    ]
    faulty = [
        lam["test"][index, 0]
        for index in range(lam["test"].shape[0])
    ]
    fault_names = [str(name).strip() for name in lam["fault_names"]]
    return normal, faulty, fault_names


def chronological_process_segment(wafer):
    """Filter monitored steps, restore time order, and merge duplicate times."""
    segment = np.asarray(
        wafer[np.isin(wafer[:, STEP_COL], PROCESS_STEPS)], dtype=np.float64)
    if not len(segment):
        raise ValueError("wafer contains no monitored process samples")
    segment = segment[np.argsort(segment[:, 0], kind="stable")]
    times, inverse, counts = np.unique(
        segment[:, 0], return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        merged = np.zeros((len(times), segment.shape[1]), dtype=np.float64)
        np.add.at(merged, inverse, segment)
        merged /= counts[:, None]
        merged[:, 0] = times
        merged[:, STEP_COL] = np.rint(merged[:, STEP_COL])
        segment = merged
    if np.any(np.diff(segment[:, 0]) <= 0):
        raise ValueError("chronology repair did not produce strict timestamps")
    return segment


def real_data_splits():
    normal, faulty, fault_names = load_lam_data()
    normal_segments = [
        chronological_process_segment(wafer)
        for wafer in normal if wafer.shape[0] >= MIN_WAFER_LEN
    ]
    faulty_items = [
        (chronological_process_segment(wafer), name)
        for wafer, name in zip(faulty, fault_names)
        if wafer.shape[0] >= MIN_WAFER_LEN
    ]
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(normal_segments)).tolist()
    n_stats = int(len(normal_segments) * STATS_FRACTION)
    stats_indices = order[:n_stats]
    holdout_indices = order[n_stats:]
    return {
        "stats": [normal_segments[index] for index in stats_indices],
        "holdout": [normal_segments[index] for index in holdout_indices],
        "faulty": [item[0] for item in faulty_items],
        "fault_names": [item[1] for item in faulty_items],
        "stats_indices": stats_indices,
        "holdout_indices": holdout_indices,
    }


def resample(values, length=PROFILE_GRID):
    values = np.asarray(values, dtype=np.float64)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.stack([
        np.interp(target, source, values[:, feature])
        for feature in range(values.shape[1])
    ], axis=1)


def quantization_step(values):
    unique = np.unique(values)
    differences = np.diff(unique)
    differences = differences[differences > 1e-9]
    return float(differences.min()) if len(differences) else 0.0


def _positive_semidefinite(matrix, floor=1e-9):
    matrix = (np.asarray(matrix) + np.asarray(matrix).T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def _transition_fraction(trace, pressure_position):
    stop = max(10, int(np.ceil(0.25 * len(trace))))
    location = int(np.argmax(np.abs(np.diff(
        trace[:stop, pressure_position]))))
    return location / max(len(trace) - 1, 1)


def _align_trace(trace, transition, canonical):
    canonical_progress = np.linspace(0.0, 1.0, PROFILE_GRID)
    source_progress = phase_map(
        canonical_progress, canonical, transition)
    observed_progress = np.linspace(0.0, 1.0, len(trace))
    return np.stack([
        np.interp(source_progress, observed_progress, trace[:, feature])
        for feature in range(trace.shape[1])
    ], axis=1)


def _recipe_partition(traces, pressure_position):
    pressure_means = np.asarray([
        trace[:, pressure_position].mean() for trace in traces
    ])
    order = np.argsort(pressure_means)
    ordered = pressure_means[order]
    gaps = np.diff(ordered)
    largest_index = int(np.argmax(gaps))
    largest_gap = float(gaps[largest_index])
    remaining = np.delete(gaps, largest_index)
    reference_gap = float(np.median(remaining)) if len(remaining) else 0.0
    if largest_gap > max(5.0 * reference_gap, 5.0):
        cut = float((ordered[largest_index] + ordered[largest_index + 1]) / 2)
        labels = (pressure_means > cut).astype(int)
    else:
        cut = None
        labels = np.zeros(len(traces), dtype=int)
    return labels, pressure_means, largest_gap, cut


def _derive_recipe_group(traces, pressure_position, group_id):
    transitions = np.asarray([
        _transition_fraction(trace, pressure_position) for trace in traces
    ])
    canonical = float(np.median(transitions))
    aligned = np.stack([
        _align_trace(trace, transition, canonical)
        for trace, transition in zip(traces, transitions)
    ])
    profile = aligned.mean(axis=0)
    offsets = []
    residuals = []
    lag_values = [[] for _ in range(profile.shape[1])]
    grid = np.linspace(0.0, 1.0, PROFILE_GRID)
    for trace, transition in zip(traces, transitions):
        progress = np.linspace(0.0, 1.0, len(trace))
        mapped = phase_map(progress, transition, canonical)
        expected = np.stack([
            np.interp(mapped, grid, profile[:, feature])
            for feature in range(profile.shape[1])
        ], axis=1)
        offset = (trace - expected).mean(axis=0)
        residual = trace - expected - offset
        offsets.append(offset)
        residuals.append(residual)
        for feature in range(profile.shape[1]):
            values = residual[:, feature]
            if values.std() > 1e-9:
                lag_values[feature].append(float(np.corrcoef(
                    values[:-1], values[1:])[0, 1]))
    residual_covariance = _positive_semidefinite(
        np.cov(np.concatenate(residuals), rowvar=False))
    if len(offsets) > 1:
        offset_covariance = _positive_semidefinite(
            np.cov(np.asarray(offsets), rowvar=False))
    else:
        offset_covariance = np.eye(profile.shape[1]) * 1e-9
    lag1 = np.clip(np.asarray([
        float(np.mean(values)) if values else 0.0
        for values in lag_values
    ]), 0.0, 0.95)
    early_slew = np.asarray([
        np.max(np.abs(np.diff(
            trace[:max(10, int(np.ceil(0.25 * len(trace))))], axis=0)), axis=0)
        for trace in traces
    ])
    return {
        "id": int(group_id),
        "count": len(traces),
        "profile": profile.tolist(),
        "residual_covariance": residual_covariance.tolist(),
        "offset_covariance": offset_covariance.tolist(),
        "lag1": lag1.tolist(),
        "canonical_transition_fraction": canonical,
        "normal_transition_fractions": transitions.tolist(),
        "early_slew_abs_p99": np.percentile(
            early_slew, 99, axis=0).tolist(),
        "pressure_mean_range": [
            float(min(trace[:, pressure_position].mean() for trace in traces)),
            float(max(trace[:, pressure_position].mean() for trace in traces)),
        ],
    }


def derive_statistics(stats_wafers):
    sensor_names = [SELECTED_SENSORS[index][0] for index in SENSOR_IDX]
    traces = [wafer[:, SENSOR_IDX] for wafer in stats_wafers]
    pressure_position = sensor_names.index("Pressure")
    labels, pressure_means, largest_gap, recipe_cut = _recipe_partition(
        traces, pressure_position)
    groups = []
    for group_id in np.unique(labels):
        group_traces = [
            trace for trace, label in zip(traces, labels)
            if label == group_id
        ]
        group = _derive_recipe_group(
            group_traces, pressure_position, group_id)
        group["weight"] = len(group_traces) / len(traces)
        groups.append(group)

    all_values = np.concatenate(traces, axis=0)
    quantization = np.asarray([
        quantization_step(all_values[:, feature])
        for feature in range(len(sensor_names))
    ])
    positive_intervals = np.concatenate([
        np.diff(wafer[:, 0]) for wafer in stats_wafers
    ])
    positive_intervals = positive_intervals[positive_intervals > 0]

    return {
        "version": "3.1",
        "chronology_contract": (
            "monitored samples sorted by Time; duplicate timestamps averaged"),
        "sensor_names": sensor_names,
        "sensor_indices": list(SENSOR_IDX),
        "length_range": [
            int(min(map(len, traces))), int(max(map(len, traces)))],
        "recipe_partition": {
            "method": "largest gap in stats-only wafer mean Pressure",
            "largest_gap": largest_gap,
            "cut": recipe_cut,
            "pressure_means": pressure_means.tolist(),
        },
        "recipe_groups": groups,
        "quantization_step": quantization.tolist(),
        "anomaly_protocol": {
            "fast_transition_speed_factor": [3.0, 6.0],
            "dynamic_slew_p99_ratio": [1.25, 1.75],
            "oscillation_residual_sigma": [4.0, 7.0],
            "drift_endpoint_residual_sigma": [4.0, 6.0],
            "interpretation": (
                "synthetic medium-to-strong faults; magnitudes are fixed "
                "before model selection and must be reported explicitly"),
        },
        "sampling": {
            "median_interval": float(np.median(positive_intervals)),
            "p05_interval": float(np.percentile(positive_intervals, 5)),
            "p95_interval": float(np.percentile(positive_intervals, 95)),
        },
        "normal_stats_count": len(stats_wafers),
    }


def save_statistics(path, statistics):
    Path(path).write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def load_statistics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def phase_map(progress, transition, canonical):
    progress = np.asarray(progress, dtype=np.float64)
    transition = float(np.clip(transition, 1e-3, 0.95))
    canonical = float(np.clip(canonical, 1e-3, 0.95))
    before = progress <= transition
    mapped = np.empty_like(progress)
    mapped[before] = progress[before] * canonical / transition
    mapped[~before] = canonical + (
        (progress[~before] - transition) *
        (1.0 - canonical) / (1.0 - transition))
    return np.clip(mapped, 0.0, 1.0)


def multivariate_ar_noise(rng, length, covariance, lag1):
    covariance = _positive_semidefinite(covariance)
    lag1 = np.asarray(lag1, dtype=np.float64)
    innovation_covariance = covariance * (1.0 - np.outer(lag1, lag1))
    innovation_covariance = _positive_semidefinite(innovation_covariance)
    values = np.zeros((length, len(lag1)), dtype=np.float64)
    values[0] = rng.multivariate_normal(
        np.zeros(len(lag1)), covariance, check_valid="ignore")
    innovations = rng.multivariate_normal(
        np.zeros(len(lag1)), innovation_covariance,
        size=max(length - 1, 0), check_valid="ignore")
    for index in range(1, length):
        values[index] = lag1 * values[index - 1] + innovations[index - 1]
    return values


def generate_wafer(rng, statistics, anomaly=0, return_metadata=False):
    group_index = int(rng.choice(
        len(statistics["recipe_groups"]),
        p=[group["weight"] for group in statistics["recipe_groups"]]))
    group = statistics["recipe_groups"][group_index]
    profile = np.asarray(group["profile"], dtype=np.float64)
    covariance = np.asarray(group["residual_covariance"], dtype=np.float64)
    offset_covariance = np.asarray(
        group["offset_covariance"], dtype=np.float64)
    lag1 = np.asarray(group["lag1"], dtype=np.float64)
    quantization = np.asarray(
        statistics["quantization_step"], dtype=np.float64)
    transient_amplitude = profile[PROFILE_GRID // 5:].mean(axis=0) - profile[0]
    anomaly_protocol = statistics["anomaly_protocol"]
    length = int(rng.integers(
        statistics["length_range"][0], statistics["length_range"][1] + 1))
    progress = np.linspace(0.0, 1.0, length)
    canonical = group["canonical_transition_fraction"]
    transition = float(rng.choice(
        group["normal_transition_fractions"]))
    mapped = phase_map(progress, transition, canonical)
    grid = np.linspace(0.0, 1.0, len(profile))
    base = np.stack([
        np.interp(mapped, grid, profile[:, feature])
        for feature in range(profile.shape[1])
    ], axis=1)
    base += rng.multivariate_normal(
        np.zeros(profile.shape[1]), offset_covariance, check_valid="ignore")
    base += multivariate_ar_noise(rng, length, covariance, lag1)

    if anomaly == 0:
        affected = []
    else:
        pool = list(range(profile.shape[1]))
        count = min(int(rng.integers(1, 3)), len(pool))
        affected = sorted(
            int(index) for index in rng.choice(
                pool, size=count, replace=False))

    onset_indices = []
    end_indices = []
    speed_factor = None
    fast_transition = None
    anomaly_strength_sigma = None
    anomaly_slew_ratio = None
    support = None
    if anomaly == 1:
        baseline_before_intervention = base.copy()
        speed_factor = float(rng.uniform(
            *anomaly_protocol["fast_transition_speed_factor"]))
        anomaly_slew_ratio = float(rng.uniform(
            *anomaly_protocol["dynamic_slew_p99_ratio"]))
        fast_transition = max(transition / speed_factor, 1.0 / length)
        fast_mapped = phase_map(progress, fast_transition, canonical)
        slew_reference = np.asarray(
            group["early_slew_abs_p99"], dtype=np.float64)
        for feature in affected:
            normal_profile = np.interp(mapped, grid, profile[:, feature])
            fast_profile = np.interp(fast_mapped, grid, profile[:, feature])
            base[:, feature] += fast_profile - normal_profile
            start = max(int(round(fast_transition * (length - 1))), 0)
            ring_length = max(int(round(0.08 * length)), 5)
            ring_index = np.arange(length) - start
            active = (ring_index >= 0) & (ring_index < ring_length)
            amplitude = abs(transient_amplitude[feature]) * rng.uniform(0.08, 0.15)
            base[active, feature] += (
                amplitude * np.exp(-ring_index[active] / max(ring_length / 3, 1)) *
                np.sin(2 * np.pi * ring_index[active] / max(ring_length / 2, 3)))
            pulse = min(max(start, 1), length - 2)
            target_slew = anomaly_slew_ratio * slew_reference[feature]
            if quantization[feature] > 0:
                target_slew = (
                    np.ceil(target_slew / quantization[feature]) *
                    quantization[feature])
            direction = np.sign(
                profile[min(PROFILE_GRID // 5, len(profile) - 1), feature] -
                profile[0, feature])
            if direction == 0:
                direction = 1.0
            base[pulse, feature] = (
                base[pulse - 1, feature] + direction * target_slew)
        support = intervention_support(
            base - baseline_before_intervention, affected)
        if support["onset_index"] is None:
            raise RuntimeError("Type A intervention has empty support")
        onset_indices.append(support["onset_index"])
        end_indices.append(support["end_index"])

    if anomaly == 2:
        anomaly_strength_sigma = float(rng.uniform(
            *anomaly_protocol["oscillation_residual_sigma"]))
        for feature in affected:
            sigma = np.sqrt(covariance[feature, feature])
            amplitude = anomaly_strength_sigma * sigma
            period = rng.uniform(6.0, 15.0)
            center = rng.uniform(0.35, 0.70) * length
            half = rng.uniform(0.12, 0.20) * length
            low = max(int(center - half), 0)
            high = min(int(center + half), length - 8)
            envelope = np.zeros(length)
            envelope[low:high] = np.hanning(high - low)
            steps = np.arange(length, dtype=np.float64)
            base[:, feature] += (
                amplitude * envelope * np.sin(
                    2 * np.pi * steps / period + rng.uniform(0, 2 * np.pi)))
            onset_indices.append(low)
            end_indices.append(max(high - 1, low))

    if anomaly == 3:
        anomaly_strength_sigma = float(rng.uniform(
            *anomaly_protocol["drift_endpoint_residual_sigma"]))
        for feature in affected:
            sigma = np.sqrt(covariance[feature, feature])
            start = int(rng.uniform(0.20, 0.35) * length)
            endpoint = (
                rng.choice([-1.0, 1.0]) * anomaly_strength_sigma * sigma)
            drift = np.zeros(length)
            drift[start:] = np.linspace(0.0, endpoint, length - start)
            base[:, feature] += drift
            onset_indices.append(start)
            end_indices.append(length - 1)

    for feature, step in enumerate(quantization):
        if step > 0:
            base[:, feature] = np.round(base[:, feature] / step) * step

    metadata = {
        "generator_version": 3,
        "anomaly_type": int(anomaly),
        "anomaly_name": ANOMALY_NAMES[int(anomaly)],
        "affected_sensor_positions": affected,
        "onset_index": min(onset_indices) if onset_indices else None,
        "end_index": max(end_indices) if end_indices else None,
        "sequence_length": length,
        "recipe_group_id": group["id"],
        "normal_transition_fraction": transition,
        "anomaly_transition_fraction": fast_transition,
        "speed_factor": speed_factor,
        "anomaly_slew_ratio": anomaly_slew_ratio,
        "anomaly_strength_sigma": anomaly_strength_sigma,
        "intervention_support": support,
    }
    return (base.astype(np.float32), metadata) if return_metadata else base.astype(np.float32)


def generate_set(rng, statistics, count, anomaly=0, with_metadata=False):
    generated = [
        generate_wafer(
            rng, statistics, anomaly=anomaly,
            return_metadata=with_metadata)
        for _ in range(count)
    ]
    if not with_metadata:
        return generated
    sequences, metadata = zip(*generated)
    return list(sequences), list(metadata)


def object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def metadata_array(values):
    return np.asarray([
        json.dumps(value, ensure_ascii=False) for value in values
    ])
