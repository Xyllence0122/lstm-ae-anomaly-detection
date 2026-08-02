# -*- coding: utf-8 -*-
"""Historical-reuse audit on real wafers already evaluated during V2."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from config import DATA_MAT, OUTPUT_DIR, SENSOR_IDX
from online_evaluation import binary_event_metrics, wilson_interval
from v3_data import real_data_splits
from v3_features import transform_sequences


PROJECT_DIR = Path(__file__).resolve().parent
V3_DIR = OUTPUT_DIR / "v3"
ARTIFACT_PATH = V3_DIR / "sliding_window_lstm_ae_v3_1.pt"
REPORT_PATH = V3_DIR / "historical_real_data_reuse_audit_v3_1.json"
V2_STATS_PATH = OUTPUT_DIR / "sensor_stats.json"
V2_REAL_REPORT_PATH = OUTPUT_DIR / "real_validation.json"
EXPECTED_ARTIFACT_SHA256 = (
    "5225deea1e1af34fc5fcab987af03d0aefaebae0c42433b71be4868dac6468b2")
EXTRA_PROVENANCE_PATHS = []


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def interval_record(successes, total):
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": successes / total,
        "wilson_95pct": [lower, upper],
    }


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "v3_locked_evaluator",
        PROJECT_DIR / "16_evaluate_v3_locked_holdout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if file_sha256(ARTIFACT_PATH) != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("V3.1 artifact hash mismatch")
    if REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite historical reuse audit")
    splits = real_data_splits()
    normal = [
        np.asarray(wafer[:, SENSOR_IDX], dtype=np.float32)
        for wafer in splits["holdout"]
    ]
    faulty = [
        np.asarray(wafer[:, SENSOR_IDX], dtype=np.float32)
        for wafer in splits["faulty"]
    ]
    artifact = torch.load(
        ARTIFACT_PATH, map_location="cpu", weights_only=False)
    sequences = normal + faulty
    transformed = transform_sequences(sequences, artifact["feature_spec"])
    evaluator = load_evaluator()
    model = evaluator.model_from_artifact(artifact)
    outputs = evaluator.profile_curves(
        model, transformed, artifact["profiles"],
        artifact["feature_spec"]["score_feature_indices"])
    labels = np.concatenate([
        np.zeros(len(normal), dtype=int),
        np.ones(len(faulty), dtype=int),
    ])
    metadata = [{} for _ in sequences]
    scores, alerts, _ = evaluator.score_events(
        outputs, labels, metadata, artifact["threshold"])
    predictions = scores > artifact["threshold"]
    metrics = binary_event_metrics(labels, predictions, scores)
    normal_predictions = predictions[:len(normal)]
    faulty_predictions = predictions[len(normal):]

    by_fault_name = defaultdict(list)
    for name, detected in zip(splits["fault_names"], faulty_predictions):
        by_fault_name[name].append(bool(detected))
    fault_summary = {
        name: interval_record(sum(values), len(values))
        for name, values in sorted(by_fault_name.items())
    }
    report = {
        "status": "historically_reused_real_data_audit_no_v3_tuning",
        "interpretation": (
            "These wafers were excluded from V3 weights and threshold fitting, "
            "but the same 43 normal and 20 faulty wafers were already evaluated "
            "during V2. This is a historical reuse audit, not unseen or "
            "independent external validation."),
        "data": {
            "normal_holdout_count": len(normal),
            "faulty_count": len(faulty),
            "normal_split_indices": splits["holdout_indices"],
            "source_mat_sha256": file_sha256(DATA_MAT),
            "chronology_repair": (
                "sort monitored step 4/5 samples by Time and average duplicates"),
            "historical_reuse": {
                "same_split_as_v2": True,
                "previously_evaluated_during_v2": True,
                "v2_statistics_path": str(V2_STATS_PATH),
                "v2_statistics_sha256": file_sha256(V2_STATS_PATH),
                "v2_real_report_path": str(V2_REAL_REPORT_PATH),
                "v2_real_report_sha256": file_sha256(V2_REAL_REPORT_PATH),
            },
        },
        "counts": {
            "normal_false_positives": int(normal_predictions.sum()),
            "normal_true_negatives": int((~normal_predictions).sum()),
            "faulty_detected": int(faulty_predictions.sum()),
            "faulty_missed": int((~faulty_predictions).sum()),
        },
        "metrics": {
            **metrics,
            "normal_fpr_interval": interval_record(
                normal_predictions.sum(), len(normal_predictions)),
            "faulty_detection_interval": interval_record(
                faulty_predictions.sum(), len(faulty_predictions)),
            "per_fault_name": fault_summary,
            "alert_sample_indices": alerts[len(normal):],
        },
        "provenance": {
            "artifact_sha256": file_sha256(ARTIFACT_PATH),
            "code_sha256": {
                str(Path(__file__)): file_sha256(Path(__file__)),
                "16_evaluate_v3_locked_holdout.py": file_sha256(
                    "16_evaluate_v3_locked_holdout.py"),
                "v3_data.py": file_sha256("v3_data.py"),
                "v3_features.py": file_sha256("v3_features.py"),
                **{
                    str(path): file_sha256(path)
                    for path in EXTRA_PROVENANCE_PATHS
                },
            },
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_porcelain": git_value("status", "--porcelain"),
        },
        "limitations": [
            "The same real wafers and split were evaluated during V2, so this is not a fresh external validation set.",
            "Only 43 V3-weight/threshold-excluded real normal wafers are available, so the FPR confidence interval is wide.",
            "Only 20 real faulty wafers are available and reliable fault onset labels are absent.",
            "The historically reused real dataset is not a representative prevalence-weighted production trial.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
