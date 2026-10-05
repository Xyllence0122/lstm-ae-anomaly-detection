# -*- coding: utf-8 -*-
"""Create the immutable V2 deployment/profile and provenance manifest.

This script only adds audit metadata around already-frozen reports. It never
recomputes scores, thresholds, predictions, or holdout metrics.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR, PROJECT_DIR
from deployment_manifest import (
    canonical_json_bytes,
    content_id,
    file_sha256,
    load_deployment_manifest,
    normalized_text_sha256,
    sensor_schema_hash,
    sha256_bytes,
)


EVALUATION_SOURCE_COMMIT = "389a3ce8e9fed49ce1fde668303ef1da378df766"
MANIFEST_PATH = OUTPUT_DIR / "edge_deployment_manifest.json"
SOURCE_PATHS = (
    "02_generate_synthetic.py",
    "07_streaming_early_warning.py",
    "08_train_sliding_window_lstm_ae.py",
    "09_compare_online_models.py",
    "10_evaluate_locked_holdout.py",
    "config.py",
    "edge_runtime.py",
    "edge_window_runtime.py",
    "models.py",
    "online_evaluation.py",
    "outputs/sensor_stats.json",
    "requirements.txt",
)
ARTIFACT_PATHS = {
    "sliding_checkpoint": "outputs/sliding_window_lstm_ae.pt",
    "sliding_torchscript": "outputs/sliding_window_lstm_ae.ts",
    "forecaster_checkpoint": "outputs/streaming_lstm_forecaster.pt",
    "forecaster_torchscript": "outputs/streaming_lstm_step.ts",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def git_blob(path):
    result = subprocess.run(
        ["git", "show", f"{EVALUATION_SOURCE_COMMIT}:{path}"],
        cwd=PROJECT_DIR, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"cannot freeze {EVALUATION_SOURCE_COMMIT}:{path}: {detail}")
    return {
        "path": path,
        "sha256": sha256_bytes(result.stdout),
    }


def json_digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def profile(value):
    value = dict(value)
    value["profile_hash"] = content_id(value)
    return value


def checkpoint_contract(checkpoint):
    keys = (
        "seed", "hidden_size", "latent_size", "window_size", "score_mode",
        "persistence_required", "persistence_span",
        "validation_target_fpr", "threshold",
    )
    return {
        key: (checkpoint[key].item()
              if isinstance(checkpoint[key], np.generic)
              else checkpoint[key])
        for key in keys
    }


def result_binding(item, index, manifest_id, *, report_scope):
    model = item["model"]
    artifact_id = (
        "sliding_checkpoint" if model == "Sliding-Window LSTM-AE"
        else "forecaster_checkpoint")
    binding = {
        "result_index": index,
        "model": model,
        "target_fpr": item["target_fpr"],
        "threshold": item["threshold"],
        "checkpoint_artifact": artifact_id,
        "checkpoint_sha256": file_sha256(
            PROJECT_DIR / ARTIFACT_PATHS[artifact_id]),
        "calibration_scope": report_scope,
        "manifest_id": manifest_id,
    }
    if report_scope == "final_holdout_normal_only_calibration" and \
            model == "Sliding-Window LSTM-AE":
        suffix = {0.01: "1pct", 0.005: "0_5pct", 0.001: "0_1pct"}[
            item["target_fpr"]]
        binding["profile_id"] = f"final_calibration_fpr_{suffix}"
    return binding


def add_report_bindings(manifest_id, reports):
    final_report = reports["final"]
    final_report["protocol"]["provenance_manifest"] = {
        "path": "outputs/edge_deployment_manifest.json",
        "manifest_id": manifest_id,
    }
    final_report["result_bindings"] = [
        result_binding(
            item, index, manifest_id,
            report_scope="final_holdout_normal_only_calibration")
        for index, item in enumerate(final_report["results"])
    ]
    write_json(OUTPUT_DIR / "final_holdout_evaluation.json", final_report)

    prevalence = reports["prevalence"]
    prevalence["result_bindings"] = [
        result_binding(
            item, index, manifest_id,
            report_scope="development_normal_only_calibration")
        for index, item in enumerate(prevalence["online_results"])
    ]
    write_json(OUTPUT_DIR / "prevalence_evaluation.json", prevalence)

    sliding = reports["sliding"]
    sliding["deployment_binding"] = {
        "manifest_id": manifest_id,
        "profile_id": "validation_operating_point",
        "checkpoint_artifact": "sliding_checkpoint",
        "checkpoint_sha256": file_sha256(
            PROJECT_DIR / ARTIFACT_PATHS["sliding_checkpoint"]),
        "threshold": sliding["release"]["threshold"],
    }
    write_json(OUTPUT_DIR / "sliding_window_metrics.json", sliding)

    real = reports["real"]
    real["result_bindings"] = [{
        "result_key": "sliding_window_lstm_ae",
        "manifest_id": manifest_id,
        "checkpoint_artifact": "sliding_checkpoint",
        "checkpoint_sha256": file_sha256(
            PROJECT_DIR / ARTIFACT_PATHS["sliding_checkpoint"]),
        "operating_point": "real_normal_recalibration_sanity_check",
        "deployment_profile": False,
        "threshold": real["sliding_window_lstm_ae"]["threshold"],
    }]
    write_json(OUTPUT_DIR / "real_validation.json", real)


def strip_audit_metadata(reports):
    """Make rebuilding independent of metadata from an earlier manifest."""
    reports["final"].pop("result_bindings", None)
    reports["final"]["protocol"].pop("provenance_manifest", None)
    reports["prevalence"].pop("result_bindings", None)
    reports["sliding"].pop("deployment_binding", None)
    reports["real"].pop("result_bindings", None)


def build():
    if MANIFEST_PATH.exists():
        document = load_deployment_manifest(
            MANIFEST_PATH, PROJECT_DIR, verify_artifacts=True,
            verify_provenance=True)
        print(f"Verified existing manifest: {document['manifest_id']}")
        return

    report_paths = {
        "final": OUTPUT_DIR / "final_holdout_evaluation.json",
        "prevalence": OUTPUT_DIR / "prevalence_evaluation.json",
        "sliding": OUTPUT_DIR / "sliding_window_metrics.json",
        "real": OUTPUT_DIR / "real_validation.json",
    }
    reports = {name: read_json(path) for name, path in report_paths.items()}
    strip_audit_metadata(reports)
    checkpoint = torch.load(
        PROJECT_DIR / ARTIFACT_PATHS["sliding_checkpoint"],
        map_location="cpu", weights_only=False)
    sensor_names = list(checkpoint["sensor_names"])
    schema_hash = sensor_schema_hash(sensor_names)

    final_sliding = [
        item for item in reports["final"]["results"]
        if item["model"] == "Sliding-Window LSTM-AE"]
    final_profiles = {}
    for item in final_sliding:
        suffix = {0.01: "1pct", 0.005: "0_5pct", 0.001: "0_1pct"}[
            item["target_fpr"]]
        final_profiles[f"final_calibration_fpr_{suffix}"] = profile({
            "model_id": "sliding_window_lstm_ae",
            "checkpoint_artifact": "sliding_checkpoint",
            "model_artifact": "sliding_torchscript",
            "schema_hash": schema_hash,
            "threshold": item["threshold"],
            "target_fpr": item["target_fpr"],
            "calibration_data": "final synthetic normal calibration cohort",
            "calibration_normal_count": reports["final"]["protocol"][
                "calibration_normal_count"],
            "calibration_seed": reports["final"]["protocol"]["random_seeds"][
                "calibration"],
            "test_labels_used_for_threshold": False,
        })

    profiles = {
        "validation_operating_point": profile({
            "model_id": "sliding_window_lstm_ae",
            "checkpoint_artifact": "sliding_checkpoint",
            "model_artifact": "sliding_torchscript",
            "schema_hash": schema_hash,
            "threshold": reports["sliding"]["release"]["threshold"],
            "target_fpr": reports["sliding"]["release"][
                "validation_target_fpr"],
            "calibration_data": "Step 3/V2 validation operating point",
            "test_labels_used_for_threshold": False,
            "selection_note": (
                "Labeled synthetic validation anomalies were used for model "
                "and score-configuration selection; the numeric threshold "
                "was computed from validation normals only."),
        }),
        **final_profiles,
    }

    report_entries = [
        {
            "path": "outputs/final_holdout_evaluation.json",
            "json_pointer": "/results",
            "sha256": json_digest(reports["final"]["results"]),
        },
        {
            "path": "outputs/prevalence_evaluation.json",
            "json_pointer": "/online_results",
            "sha256": json_digest(reports["prevalence"]["online_results"]),
        },
        {
            "path": "outputs/sliding_window_metrics.json",
            "json_pointer": "/release",
            "sha256": json_digest(reports["sliding"]["release"]),
        },
        {
            "path": "outputs/sliding_window_metrics.json",
            "json_pointer": "/artifact_parity",
            "sha256": json_digest(reports["sliding"]["artifact_parity"]),
        },
        {
            "path": "outputs/real_validation.json",
            "json_pointer": "/sliding_window_lstm_ae",
            "sha256": json_digest(reports["real"]["sliding_window_lstm_ae"]),
        },
    ]
    artifacts = {
        artifact_id: {
            "path": relative_path,
            "sha256": file_sha256(PROJECT_DIR / relative_path),
        }
        for artifact_id, relative_path in ARTIFACT_PATHS.items()
    }
    payload = {
        "default_profile": "final_calibration_fpr_1pct",
        "model_contract": {
            "model_id": "sliding_window_lstm_ae",
            "sensor_names": sensor_names,
            "schema_hash": schema_hash,
            "input_contract": "ordered named mapping or explicit columns header",
            "checkpoint_configuration": checkpoint_contract(checkpoint),
        },
        "profiles": profiles,
        "artifacts": artifacts,
        "reports": report_entries,
        "provenance": {
            "evaluation_source_commit": EVALUATION_SOURCE_COMMIT,
            "git_blobs": [git_blob(path) for path in SOURCE_PATHS],
            "deployment_runtime_files": [
                {
                    "path": path,
                    "hash_mode": "normalized_lf_text_sha256",
                    "sha256": normalized_text_sha256(PROJECT_DIR / path),
                }
                for path in (
                    "deployment_manifest.py", "edge_window_runtime.py")
            ],
            "environment": {
                "python": platform.python_version(),
                "distributions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "matplotlib", "numpy", "scikit-learn", "scipy",
                        "torch")
                },
            },
            "final_holdout_protocol": reports["final"]["protocol"],
            "development_protocol": reports["prevalence"]["protocol"],
        },
    }
    document = {
        "manifest_version": 1,
        "manifest_id": content_id(payload),
        "payload": payload,
    }
    write_json(MANIFEST_PATH, document)
    MANIFEST_PATH.with_suffix(".sha256").write_text(
        file_sha256(MANIFEST_PATH) + "  " + MANIFEST_PATH.name + "\n",
        encoding="ascii")
    add_report_bindings(document["manifest_id"], reports)
    load_deployment_manifest(
        MANIFEST_PATH, PROJECT_DIR, verify_artifacts=True,
        verify_provenance=True)
    print(f"Created and verified manifest: {document['manifest_id']}")


if __name__ == "__main__":
    build()
