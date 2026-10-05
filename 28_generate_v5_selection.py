# -*- coding: utf-8 -*-
"""Generate new V5 selection cohorts without creating locked holdout data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import OUTPUT_DIR
from v3_data import (
    generate_set,
    load_statistics,
    metadata_array,
    object_array,
)
from v4_hashing import normalized_text_sha256
from v5_experiment import file_sha256


V5_DIR = OUTPUT_DIR / "v5"
STATS_PATH = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
DATA_PATH = V5_DIR / "selection_data_v5.npz"
PROTOCOL_PATH = V5_DIR / "selection_protocol_v5.json"
EXPECTED_STATS_NORMALIZED_SHA256 = (
    "5e9b3aa0cbec720abb4871c771b11584066fb1fab235767ed92e63557bf2f76b")
N_TRAIN_NORMAL = 1500
N_VALIDATION_NORMAL = 3000
N_VALIDATION_PER_ANOMALY = 400
TRAIN_SEED = 710101
VALIDATION_NORMAL_SEED = 710102
VALIDATION_ANOMALY_SEED = 710103
RESERVED_CALIBRATION_SEED = 720101
RESERVED_HOLDOUT_NORMAL_SEED = 730101
RESERVED_HOLDOUT_ANOMALY_SEED = 730102


def main():
    V5_DIR.mkdir(exist_ok=True)
    for path in (DATA_PATH, PROTOCOL_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V5 selection: {path}")
    actual_stats_hash = normalized_text_sha256(STATS_PATH)
    if actual_stats_hash != EXPECTED_STATS_NORMALIZED_SHA256:
        raise RuntimeError(
            f"statistics hash mismatch: {actual_stats_hash}")
    statistics = load_statistics(STATS_PATH)
    train = generate_set(
        np.random.default_rng(TRAIN_SEED), statistics,
        N_TRAIN_NORMAL, anomaly=0)
    validation = generate_set(
        np.random.default_rng(VALIDATION_NORMAL_SEED), statistics,
        N_VALIDATION_NORMAL, anomaly=0)
    anomaly_rng = np.random.default_rng(VALIDATION_ANOMALY_SEED)
    anomalies = []
    labels = []
    metadata = []
    for anomaly_type in (1, 2, 3):
        sequences, sequence_metadata = generate_set(
            anomaly_rng, statistics, N_VALIDATION_PER_ANOMALY,
            anomaly=anomaly_type, with_metadata=True)
        anomalies.extend(sequences)
        labels.extend([anomaly_type] * len(sequences))
        metadata.extend(sequence_metadata)
    np.savez_compressed(
        DATA_PATH,
        X_train=object_array(train),
        X_val=object_array(validation),
        X_val_anom=object_array(anomalies),
        y_val_anom=np.asarray(labels, dtype=np.int64),
        val_metadata=metadata_array(metadata),
        sensor_names=np.asarray(statistics["sensor_names"]),
    )
    protocol = {
        "version": 5,
        "status": "selection_only_no_locked_holdout_generated",
        "same_generator_family_as_v3_2": True,
        "counts": {
            "train_normal": N_TRAIN_NORMAL,
            "validation_normal": N_VALIDATION_NORMAL,
            "validation_per_anomaly": N_VALIDATION_PER_ANOMALY,
        },
        "selection_seeds": {
            "train_normal": TRAIN_SEED,
            "validation_normal": VALIDATION_NORMAL_SEED,
            "validation_anomaly": VALIDATION_ANOMALY_SEED,
        },
        "reserved_unopened_seeds": {
            "normal_only_calibration": RESERVED_CALIBRATION_SEED,
            "locked_holdout_normal": RESERVED_HOLDOUT_NORMAL_SEED,
            "locked_holdout_anomaly": RESERVED_HOLDOUT_ANOMALY_SEED,
        },
        "statistics": {
            "path": str(STATS_PATH),
            "hash_mode": "normalized_text_sha256",
            "sha256": actual_stats_hash,
        },
        "data_sha256": file_sha256(DATA_PATH),
        "methodology": (
            "Only these cohorts may be used for V5 model, checkpoint, profile, "
            "and hyperparameter selection. Reserved calibration is normal-only. "
            "Reserved locked holdout seeds must remain unopened until the "
            "selection acceptance gate passes."),
        "limitations": [
            "All synthetic cohorts use the same generator family as V3.2.",
            "Real statistics inform the generator; this is not independent "
            "external validation.",
        ],
    }
    PROTOCOL_PATH.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
