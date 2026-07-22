# -*- coding: utf-8 -*-
"""Generate V3.2 selection data with corrected Type A support metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from config import DATA_MAT, OUTPUT_DIR
from v3_data import (
    derive_statistics,
    generate_set,
    metadata_array,
    object_array,
    real_data_splits,
    save_statistics,
)


V3_DIR = OUTPUT_DIR / "v3"
OLD_DATA_PATH = V3_DIR / "selection_data_v3.npz"
STATS_PATH = V3_DIR / "sensor_stats_v3_2.json"
DATA_PATH = V3_DIR / "selection_data_v3_2.npz"
PROTOCOL_PATH = V3_DIR / "data_protocol_v3_2.json"
N_TRAIN_NORMAL = 1000
N_VALIDATION_NORMAL = 2000
N_VALIDATION_PER_ANOMALY = 300
TRAIN_SEED = 31001
VALIDATION_NORMAL_SEED = 32001
VALIDATION_ANOMALY_SEED = 33001


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequences_equal(first, second):
    return len(first) == len(second) and all(
        np.array_equal(left, right)
        for left, right in zip(first, second)
    )


def main():
    for path in (STATS_PATH, DATA_PATH, PROTOCOL_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V3.2 data: {path}")
    splits = real_data_splits()
    statistics = derive_statistics(splits["stats"])
    statistics.update({
        "split_seed": 123,
        "stats_indices": splits["stats_indices"],
        "holdout_indices": splits["holdout_indices"],
        "source_mat_sha256": file_sha256(DATA_MAT),
        "v3_2_metadata_revision": (
            "Type A onset/end use exact pre-quantization intervention support"),
    })
    save_statistics(STATS_PATH, statistics)

    train = generate_set(
        np.random.default_rng(TRAIN_SEED), statistics,
        N_TRAIN_NORMAL, anomaly=0)
    validation = generate_set(
        np.random.default_rng(VALIDATION_NORMAL_SEED), statistics,
        N_VALIDATION_NORMAL, anomaly=0)
    anomaly_rng = np.random.default_rng(VALIDATION_ANOMALY_SEED)
    validation_anomaly = []
    validation_labels = []
    validation_metadata = []
    for anomaly_type in (1, 2, 3):
        sequences, metadata = generate_set(
            anomaly_rng, statistics, N_VALIDATION_PER_ANOMALY,
            anomaly=anomaly_type, with_metadata=True)
        validation_anomaly.extend(sequences)
        validation_labels.extend([anomaly_type] * len(sequences))
        validation_metadata.extend(metadata)

    old = np.load(OLD_DATA_PATH, allow_pickle=True)
    values_equal = {
        "train": sequences_equal(train, old["X_train"]),
        "validation_normal": sequences_equal(validation, old["X_val"]),
        "validation_anomaly": sequences_equal(
            validation_anomaly, old["X_val_anom"]),
        "labels": bool(np.array_equal(
            validation_labels, old["y_val_anom"])),
    }
    old_metadata = [json.loads(str(item)) for item in old["val_metadata"]]
    changed_type_a_ends = sum(
        before.get("end_index") != after.get("end_index")
        for before, after in zip(old_metadata, validation_metadata)
    )
    if not all(values_equal.values()) or changed_type_a_ends != 300:
        raise RuntimeError(
            "V3.2 must preserve all sequence values and change 300 Type A ends")

    np.savez_compressed(
        DATA_PATH,
        X_train=object_array(train),
        X_val=object_array(validation),
        X_val_anom=object_array(validation_anomaly),
        y_val_anom=np.asarray(validation_labels, dtype=np.int64),
        val_metadata=metadata_array(validation_metadata),
        sensor_names=np.asarray(statistics["sensor_names"]),
    )
    protocol = {
        "version": "3.2",
        "selection_data_only": True,
        "lineage": {
            "source_data_path": str(OLD_DATA_PATH),
            "source_data_sha256": file_sha256(OLD_DATA_PATH),
            "sequence_values_equal_to_v3": values_equal,
            "changed_type_a_end_metadata_count": changed_type_a_ends,
        },
        "counts": {
            "train_normal": len(train),
            "validation_normal": len(validation),
            "validation_per_anomaly": N_VALIDATION_PER_ANOMALY,
        },
        "seeds": {
            "train_normal": TRAIN_SEED,
            "validation_normal": VALIDATION_NORMAL_SEED,
            "validation_anomaly": VALIDATION_ANOMALY_SEED,
        },
        "timing_feature_contract": (
            "no timestamp, delta-t, or fixed-cadence resampling; x[t]-x[t-1] "
            "and sample-index progress only"),
        "data_sha256": file_sha256(DATA_PATH),
        "statistics_sha256": file_sha256(STATS_PATH),
        "holdout_policy": "No V3.2 holdout data are generated here.",
    }
    PROTOCOL_PATH.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
