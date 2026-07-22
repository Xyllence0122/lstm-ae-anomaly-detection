# -*- coding: utf-8 -*-
"""Causal raw, per-sample-difference, and sample-progress V3 features."""
from __future__ import annotations

import numpy as np


def _unscaled_features(sequence, expected_cycle_samples):
    values = np.asarray(sequence, dtype=np.float32)
    differences = np.zeros_like(values)
    differences[1:] = values[1:] - values[:-1]
    elapsed_phase = (
        np.arange(len(values), dtype=np.float32) /
        max(float(expected_cycle_samples) - 1.0, 1.0)
    )[:, None]
    return np.concatenate([values, differences, elapsed_phase], axis=1)


def fit_feature_spec(sequences, sensor_names):
    expected_cycle_samples = float(np.median([len(item) for item in sequences]))
    unscaled = [
        _unscaled_features(item, expected_cycle_samples)
        for item in sequences
    ]
    stacked = np.concatenate(unscaled, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    feature_names = (
        list(sensor_names) +
        [f"delta::{name}" for name in sensor_names] +
        ["elapsed_phase"]
    )
    return {
        "version": 1,
        "raw_sensor_names": list(sensor_names),
        "feature_names": feature_names,
        "expected_cycle_samples": expected_cycle_samples,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "score_feature_indices": list(range(2 * len(sensor_names))),
        "causality": (
            "features at t use raw values through t, x[t]-x[t-1], and "
            "elapsed sample count since reset; actual cycle length is unused"),
        "timing_contract": (
            "no timestamp, delta-t, or resampling is used; delta features are "
            "per-sample differences and elapsed_phase is sample progress"),
    }


def transform_sequence(sequence, feature_spec):
    unscaled = _unscaled_features(
        sequence, feature_spec["expected_cycle_samples"])
    mean = np.asarray(feature_spec["feature_mean"], dtype=np.float32)
    std = np.asarray(feature_spec["feature_std"], dtype=np.float32)
    return ((unscaled - mean) / std).astype(np.float32)


def transform_sequences(sequences, feature_spec):
    return [transform_sequence(item, feature_spec) for item in sequences]


def sample_training_windows(sequences, window_sizes, samples_per_size,
                            seed):
    if samples_per_size < 1:
        raise ValueError("samples_per_size must be positive")
    rng = np.random.default_rng(seed)
    windows = []
    for sequence in sequences:
        for size in window_sizes:
            if len(sequence) < size:
                raise ValueError(
                    f"window {size} exceeds sequence length {len(sequence)}")
            starts = np.arange(len(sequence) - size + 1)
            mandatory = np.asarray([0, len(sequence) - size], dtype=int)
            random_count = max(samples_per_size - len(np.unique(mandatory)), 0)
            selected = rng.choice(
                starts, size=random_count,
                replace=random_count > len(starts)) if random_count else np.empty(
                    0, dtype=int)
            selected = np.unique(np.concatenate([mandatory, selected]))
            windows.extend([
                np.asarray(sequence[start:start + size], dtype=np.float32)
                for start in selected
            ])
    return windows
