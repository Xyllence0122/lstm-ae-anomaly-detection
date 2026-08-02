# -*- coding: utf-8 -*-
"""Create a non-destructive erratum for immutable V3/V3.1 reports."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from config import OUTPUT_DIR


V3_DIR = OUTPUT_DIR / "v3"
OUTPUT_PATH = V3_DIR / "report_errata_v3_family.json"
REPORT_PATHS = {
    "v3_initial_holdout": V3_DIR / "final_holdout_v3.json",
    "v3_1_holdout": V3_DIR / "final_holdout_v3_1.json",
    "real_reuse_report": V3_DIR / "external_real_data_v3_1.json",
    "v3_1_calibration": V3_DIR / "deployment_calibration_v3_1.json",
    "v2_real_report": OUTPUT_DIR / "real_validation.json",
    "v2_statistics": OUTPUT_DIR / "sensor_stats.json",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite erratum: {OUTPUT_PATH}")
    v3 = load_json(REPORT_PATHS["v3_initial_holdout"])
    v3_1 = load_json(REPORT_PATHS["v3_1_holdout"])
    calibration = load_json(REPORT_PATHS["v3_1_calibration"])
    v3_stats = load_json(V3_DIR / "sensor_stats_v3.json")
    v2_stats = load_json(REPORT_PATHS["v2_statistics"])
    same_split = (
        v3_stats["stats_indices"] == v2_stats["stats_idx"] and
        v3_stats["holdout_indices"] == v2_stats["holdout_idx"])
    if not same_split:
        raise RuntimeError("expected V2/V3 historical split reuse was not found")

    report = {
        "version": "V3 family erratum 1",
        "policy": (
            "Original reports remain byte-for-byte immutable. Corrections below "
            "supersede only the identified claims or metrics."),
        "source_sha256": {
            name: file_sha256(path) for name, path in REPORT_PATHS.items()
        },
        "corrections": [
            {
                "source": str(REPORT_PATHS["v3_initial_holdout"]),
                "field": "status",
                "original": v3["status"],
                "corrected": (
                    "initial locked V3 holdout; its FPR result motivated the "
                    "separately versioned V3.1 normal-only calibration protocol"),
                "impact": (
                    "The report remains a locked evaluation of its frozen V3 "
                    "artifact, but it is not a no-further-tuning statement for "
                    "the entire V3 family."),
            },
            {
                "sources": [
                    str(REPORT_PATHS["v3_initial_holdout"]),
                    str(REPORT_PATHS["v3_1_holdout"]),
                ],
                "field": "metrics.detected_before_injection_end_rate",
                "action": "withdraw",
                "reason": (
                    "Type A end_index used transition+0.08 although the phase-"
                    "warp intervention remained nonzero much later. The original "
                    "99% values have no valid injection-support interpretation."),
                "replacement_definition": (
                    "V3.2 computes onset/end from exact indices where the pre-"
                    "quantization intervention changes the generated normal "
                    "baseline by more than 1e-12."),
                "other_metrics_changed": False,
            },
            {
                "source": str(REPORT_PATHS["v3_1_holdout"]),
                "field": "protocol.threshold_source",
                "original": v3_1["protocol"]["threshold_source"],
                "corrected": {
                    "type": "post_selection_normal_only_calibration",
                    "normal_count": calibration["normal_count"],
                    "seed": calibration["seed"],
                    "target_fpr": calibration["target_fpr"],
                    "threshold": calibration["new_threshold"],
                    "calibration_report_sha256": file_sha256(
                        REPORT_PATHS["v3_1_calibration"]),
                    "calibration_code_sha256": file_sha256(
                        "17_calibrate_v3_1.py"),
                },
                "model_or_threshold_changed": False,
            },
            {
                "source": str(REPORT_PATHS["real_reuse_report"]),
                "fields": ["status", "interpretation", "limitations"],
                "action": "replace external/untouched wording",
                "corrected": (
                    "The 43 normal and 20 faulty wafers were excluded from V3 "
                    "weights and threshold fitting, but the identical split and "
                    "wafers had already been evaluated in V2. Treat this as a "
                    "historical reuse audit, not unseen independent validation."),
                "evidence": {
                    "v2_v3_split_indices_equal": same_split,
                    "v2_real_report": str(REPORT_PATHS["v2_real_report"]),
                },
            },
            {
                "scope": "V3 feature terminology",
                "action": "replace rate/time wording",
                "corrected": (
                    "Features are x[t]-x[t-1] per-sample differences and sample-"
                    "index progress. No timestamp, delta-t, or fixed-cadence "
                    "resampling is implemented."),
            },
            {
                "source": "outputs/v3/experiment_final_training/selection_report.json",
                "action": "provenance limitation",
                "corrected": (
                    "The historical report does not independently preserve the "
                    "40-epoch CLI, batch size, or samples-per-size. V3.2 must "
                    "store resolved argv/config and completed epochs per seed."),
            },
            {
                "scope": "edge deployment",
                "corrected": (
                    "V3/V3.1 have no TorchScript export, streaming runtime, or "
                    "Raspberry Pi 5 benchmark and are not edge-deployment ready."),
            },
        ],
        "provenance": {
            "generator_code_sha256": file_sha256(Path(__file__)),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_porcelain": git_value("status", "--porcelain"),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
