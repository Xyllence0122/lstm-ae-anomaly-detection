# -*- coding: utf-8 -*-
"""Build the immutable V4 runtime package from the locked V3.2 artifact."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR, PROJECT_DIR
from deployment_manifest import (
    file_sha256,
    normalized_text_sha256,
    sensor_schema_hash,
)
from models import (
    SlidingWindowLSTMAutoEncoder,
    sliding_window_error_summaries,
)
from online_evaluation import apply_persistence, sensor_error_score_curves
from v3_features import transform_sequence
from v4_edge_runtime import V4MultiscaleDetector, load_v4_manifest


V3_DIR = OUTPUT_DIR / "v3"
V4_DIR = OUTPUT_DIR / "v4"
SOURCE_ARTIFACT = V3_DIR / "sliding_window_lstm_ae_v3_2.pt"
SOURCE_FINAL_REPORT = V3_DIR / "final_holdout_v3_2.json"
SOURCE_STATS = V3_DIR / "sensor_stats_v3_2.json"
SOURCE_CALIBRATION = V3_DIR / "deployment_calibration_v3_2.json"
PARITY_DATA = V3_DIR / "selection_data_v3_2.npz"
TORCHSCRIPT_PATH = V4_DIR / "sliding_window_lstm_ae_v4.ts"
PARITY_REPORT_PATH = V4_DIR / "runtime_parity_v4.json"
ENVIRONMENT_PATH = V4_DIR / "deployment_environment_v4.json"
MANIFEST_PATH = V4_DIR / "deployment_manifest_v4.json"
MANIFEST_SIDECAR = MANIFEST_PATH.with_suffix(".sha256")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing V4 package during controlled development")
    return parser.parse_args()


def relative(path):
    return Path(path).resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            json_ready(value), ensure_ascii=False, indent=2,
            sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_record():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=PROJECT_DIR,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "status_porcelain": status}


def load_locked_source():
    report = json.loads(SOURCE_FINAL_REPORT.read_text(encoding="utf-8"))
    calibration = json.loads(
        SOURCE_CALIBRATION.read_text(encoding="utf-8"))
    expected_artifact_hash = report["provenance"]["artifact_sha256"]
    actual_artifact_hash = file_sha256(SOURCE_ARTIFACT)
    if actual_artifact_hash != expected_artifact_hash:
        raise RuntimeError(
            "V3.2 source artifact does not match its locked report")
    expected_stats_hash = report["provenance"]["statistics_sha256"]
    if file_sha256(SOURCE_STATS) != expected_stats_hash:
        raise RuntimeError("V3.2 statistics do not match the locked report")
    expected_calibration_hash = report["protocol"]["threshold_source"][
        "calibration_report"]["sha256"]
    if file_sha256(SOURCE_CALIBRATION) != expected_calibration_hash:
        raise RuntimeError(
            "V3.2 calibration report does not match the locked report")
    artifact = torch.load(
        SOURCE_ARTIFACT, map_location="cpu", weights_only=False)
    threshold = float(report["operating_point"]["ensemble_threshold"])
    if not (
        float(artifact["threshold"]) == threshold ==
        float(calibration["new_threshold"])
    ):
        raise RuntimeError(
            "V3.2 artifact, calibration, and locked threshold differ")
    if calibration["calibrated_artifact_sha256"] != actual_artifact_hash:
        raise RuntimeError(
            "calibration report does not bind the V3.2 artifact")
    if calibration["statistics_sha256"] != expected_stats_hash:
        raise RuntimeError(
            "calibration report does not bind the V3.2 statistics")
    report_profiles = report["operating_point"]["profiles"]
    artifact_profile_contract = [{
        key: profile[key] for key in (
            "window_size", "score_mode", "persistence_required",
            "persistence_span", "base_scale")
    } for profile in artifact["profiles"]]
    if artifact_profile_contract != report_profiles:
        raise RuntimeError(
            "V3.2 artifact profiles differ from the locked report")
    return artifact, report


def model_from_artifact(artifact):
    model = SlidingWindowLSTMAutoEncoder(
        len(artifact["feature_spec"]["feature_names"]),
        artifact["hidden_size"],
        artifact["latent_size"],
    )
    model.load_state_dict(artifact["state_dict"])
    return model.eval()


def export_torchscript(model, feature_count):
    example = torch.zeros((1, 64, feature_count), dtype=torch.float32)
    traced = torch.jit.trace(model, example, strict=True)
    traced.save(str(TORCHSCRIPT_PATH))
    loaded = torch.jit.load(str(TORCHSCRIPT_PATH), map_location="cpu")
    maximum_difference = 0.0
    generator = torch.Generator().manual_seed(426001)
    for window_size in (8, 64):
        sample = torch.randn(
            (4, window_size, feature_count), generator=generator)
        with torch.no_grad():
            expected = model(sample)
            actual = loaded(sample)
        maximum_difference = max(
            maximum_difference,
            float(torch.max(torch.abs(expected - actual)).item()),
        )
    return loaded, maximum_difference


def profile_contracts(artifact):
    output = []
    for index, profile in enumerate(artifact["profiles"], start=1):
        output.append({
            "profile_id": (
                f"p{index}_w{profile['window_size']}_"
                f"{profile['score_mode']}_"
                f"{profile['persistence_required']}of"
                f"{profile['persistence_span']}"
            ),
            **json_ready(profile),
        })
    return output


def timing_contract(statistics):
    nominal = float(statistics["sampling"]["median_interval"])
    p05 = float(statistics["sampling"]["p05_interval"])
    p95 = float(statistics["sampling"]["p95_interval"])
    margin = nominal * 0.05
    return {
        "timestamp_required": True,
        "strictly_increasing": True,
        "mode": "fixed_cadence_fail_closed",
        "nominal_interval_seconds": nominal,
        "minimum_interval_seconds": max(p05 - margin, 1e-6),
        "maximum_interval_seconds": p95 + margin,
        "sensor_timeout_seconds": nominal * 3.0,
        "training_interval_p05_seconds": p05,
        "training_interval_p95_seconds": p95,
        "interval_margin_seconds": margin,
        "interval_rule": "training p05/p95 extended by 5% of median",
        "subsecond_claim": (
            "not supported; the source data cadence is approximately 1 Hz"),
        "interpretation": (
            "per-sample differences are valid only inside this enforced "
            "cadence band; samples outside the band are rejected and must "
            "not be silently scored"),
    }


def parity_sequences():
    raw = np.load(PARITY_DATA, allow_pickle=True)
    normal = list(raw["X_val"])
    anomaly = list(raw["X_val_anom"])
    labels = np.asarray(raw["y_val_anom"], dtype=int)
    selected = [
        (f"normal_{index}", sequence)
        for index, sequence in enumerate(normal[:10])
    ]
    for kind in (1, 2, 3):
        indices = np.flatnonzero(labels == kind)
        for local, index in enumerate(indices[:10]):
            selected.append((f"anomaly_{kind}_{local}", anomaly[index]))
    return selected


def offline_profile_curves(model, sequence, feature_spec, profiles):
    transformed = transform_sequence(sequence, feature_spec)
    summaries = {
        window: sliding_window_error_summaries(
            model, [transformed], window)
        for window in sorted({item["window_size"] for item in profiles})
    }
    curves = {}
    first_samples = {}
    for profile in profiles:
        errors = summaries[profile["window_size"]][
            profile["score_mode"]][0]
        errors = errors[:, feature_spec["score_feature_indices"]]
        raw = sensor_error_score_curves(
            [errors], profile["calibration"])[0]
        persistent = apply_persistence(
            [raw],
            profile["persistence_required"],
            profile["persistence_span"],
        )[0]
        curves[profile["profile_id"]] = (
            persistent / profile["base_scale"])
        first_samples[profile["profile_id"]] = (
            profile["window_size"] - 1 +
            profile["persistence_span"] - 1)
    return curves, first_samples


def runtime_parity(model, scripted, artifact, profiles, timing):
    maximum_profile_score_difference = 0.0
    alarm_mismatches = 0
    compared_scores = 0
    sequence_results = []
    nominal = timing["nominal_interval_seconds"]
    for sequence_id, sequence in parity_sequences():
        curves, first_samples = offline_profile_curves(
            model, sequence, artifact["feature_spec"], profiles)
        detector = V4MultiscaleDetector(
            scripted,
            artifact["feature_spec"],
            profiles,
            artifact["threshold"],
            timing,
            model_version="v4-parity",
            artifact_sha256=file_sha256(TORCHSCRIPT_PATH),
        )
        detector.start_stream(
            sequence_id, "parity", "offline", f"parity:{sequence_id}")
        first_runtime_alarm = None
        first_offline_alarm = None
        for sample_index, row in enumerate(sequence):
            sample = dict(zip(
                artifact["feature_spec"]["raw_sensor_names"], row))
            result = detector.update(
                sample, timestamp=sample_index * nominal)
            reference_available = []
            for profile in profiles:
                profile_id = profile["profile_id"]
                curve_index = sample_index - first_samples[profile_id]
                expected_score = None
                if 0 <= curve_index < len(curves[profile_id]):
                    expected_score = float(curves[profile_id][curve_index])
                    reference_available.append(expected_score)
                actual_score = result["profiles"][profile_id]["score"]
                if expected_score is None:
                    if actual_score is not None:
                        raise AssertionError(
                            f"{profile_id} became ready too early")
                    continue
                if actual_score is None:
                    raise AssertionError(
                        f"{profile_id} was not ready at {sample_index}")
                maximum_profile_score_difference = max(
                    maximum_profile_score_difference,
                    abs(expected_score - actual_score),
                )
                compared_scores += 1
            expected_alarm = bool(
                reference_available and
                max(reference_available) > artifact["threshold"])
            if expected_alarm != result["alarm"]:
                alarm_mismatches += 1
            if result["alarm"] and first_runtime_alarm is None:
                first_runtime_alarm = sample_index
            if expected_alarm and first_offline_alarm is None:
                first_offline_alarm = sample_index
        sequence_results.append({
            "sequence_id": sequence_id,
            "length": len(sequence),
            "first_offline_alarm_sample": first_offline_alarm,
            "first_runtime_alarm_sample": first_runtime_alarm,
            "first_alarm_match": first_offline_alarm == first_runtime_alarm,
        })
    return {
        "status": "pass" if alarm_mismatches == 0 else "fail",
        "sequence_count": len(sequence_results),
        "compared_profile_scores": compared_scores,
        "maximum_absolute_profile_score_difference": (
            maximum_profile_score_difference),
        "alarm_decision_mismatches": alarm_mismatches,
        "all_first_alarm_indices_match": all(
            item["first_alarm_match"] for item in sequence_results),
        "sequences": sequence_results,
        "parity_data": {
            "path": relative(PARITY_DATA),
            "sha256": file_sha256(PARITY_DATA),
            "selection_data_only": True,
        },
    }


def artifact_record(path, role):
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "bytes": Path(path).stat().st_size,
        "role": role,
    }


def source_record(path, role):
    return {
        "path": relative(path),
        "sha256": normalized_text_sha256(path),
        "hash_mode": "normalized_text_sha256",
        "bytes": Path(path).stat().st_size,
        "role": role,
    }


def environment_record():
    packages = {}
    for name in (
            "numpy", "torch", "scipy", "scikit-learn", "matplotlib",
            "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "runtime_minimum_direct_dependencies": ["numpy", "torch"],
        "builder_additional_dependencies": ["scipy", "scikit-learn"],
        "interpretation": (
            "This records the exact successful build environment; Pi wheels "
            "may differ and must be recorded in the Pi benchmark report."),
    }


def main():
    args = parse_args()
    V4_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists() and not args.force:
        document, _, manifest_hash = load_v4_manifest(MANIFEST_PATH)
        print(json.dumps({
            "status": "existing_verified_v4_package",
            "manifest_sha256": manifest_hash,
            "model_version": document["model_version"],
        }, indent=2))
        return

    source_git = git_record()
    if source_git["status_porcelain"]:
        raise RuntimeError(
            "V4 deployment build requires a clean source worktree; "
            f"found {source_git['status_porcelain']}")
    write_json(ENVIRONMENT_PATH, environment_record())
    artifact, final_report = load_locked_source()
    statistics = json.loads(SOURCE_STATS.read_text(encoding="utf-8"))
    model = model_from_artifact(artifact)
    scripted, eager_script_max_abs = export_torchscript(
        model, len(artifact["feature_spec"]["feature_names"]))
    profiles = profile_contracts(artifact)
    timing = timing_contract(statistics)
    parity = runtime_parity(
        model, scripted, artifact, profiles, timing)
    parity.update({
        "eager_torchscript_maximum_absolute_output_difference": (
            eager_script_max_abs),
        "source_model_sha256": file_sha256(SOURCE_ARTIFACT),
        "torchscript_model_sha256": file_sha256(TORCHSCRIPT_PATH),
        "ensemble_threshold": float(artifact["threshold"]),
    })
    if (
        parity["status"] != "pass" or
        not parity["all_first_alarm_indices_match"] or
        eager_script_max_abs > 1e-5 or
        parity["maximum_absolute_profile_score_difference"] > 1e-5
    ):
        raise RuntimeError(f"V4 runtime parity failed: {parity}")
    write_json(PARITY_REPORT_PATH, parity)

    feature_spec = json_ready(artifact["feature_spec"])
    feature_spec["raw_schema_hash"] = sensor_schema_hash(
        feature_spec["raw_sensor_names"])
    manifest = {
        "manifest_version": 4,
        "model_version": "v4-runtime-v3.2-weights",
        "status": "paper_edge_runtime_prototype_not_production_release",
        "lineage": {
            "weights": "locked V3.2 weights, unchanged",
            "threshold": "locked V3.2 normal-only calibration, unchanged",
            "holdout_reuse_for_tuning": False,
            "source_final_report_status": final_report["status"],
        },
        "model_contract": {
            "architecture": "Sliding-Window LSTM Autoencoder",
            "raw_sensor_schema_hash": feature_spec["raw_schema_hash"],
            "feature_spec": feature_spec,
            "profiles": profiles,
            "ensemble_rule": (
                "maximum available normalized persistent profile score"),
            "ensemble_threshold": float(artifact["threshold"]),
            "alarm_comparison": "strict score > threshold",
            "stream_boundary": (
                "start_stream/reset required for every wafer, recipe, or "
                "monitored process segment"),
        },
        "timing_contract": timing,
        "event_contract": {
            "required_context": [
                "wafer_id", "recipe_id", "equipment_id",
                "stream_instance_id"],
            "default_pre_alarm_samples": 64,
            "default_post_alarm_samples": 16,
            "storage_format": "append-only JSON Lines",
            "saved_provenance": [
                "model version", "model hash", "manifest hash",
                "sensor schema hash", "threshold", "profile evidence",
            ],
        },
        "artifacts": {
            "source_checkpoint": artifact_record(
                SOURCE_ARTIFACT, "locked V3.2 PyTorch checkpoint"),
            "torchscript_model": artifact_record(
                TORCHSCRIPT_PATH, "V4 CPU streaming inference model"),
            "locked_final_report": artifact_record(
                SOURCE_FINAL_REPORT, "V3.2 locked holdout report"),
            "source_statistics": artifact_record(
                SOURCE_STATS, "V3.2 source statistics"),
            "normal_calibration": artifact_record(
                SOURCE_CALIBRATION, "V3.2 normal-only calibration"),
            "runtime_parity": artifact_record(
                PARITY_REPORT_PATH, "offline versus streaming parity"),
            "build_environment": artifact_record(
                ENVIRONMENT_PATH, "exact successful build environment"),
        },
        "source_provenance": {
            "runtime": source_record(
                PROJECT_DIR / "v4_edge_runtime.py", "runtime source"),
            "builder": source_record(
                Path(__file__), "deployment builder source"),
            "benchmark": source_record(
                PROJECT_DIR / "27_benchmark_v4_runtime.py",
                "benchmark source"),
            "runtime_tests": source_record(
                PROJECT_DIR / "tests" / "test_v4_runtime.py",
                "V4 runtime tests"),
            "feature_transform": source_record(
                PROJECT_DIR / "v3_features.py", "feature source"),
            "model": source_record(
                PROJECT_DIR / "models.py", "model source"),
            "online_scoring": source_record(
                PROJECT_DIR / "online_evaluation.py", "score source"),
            "locked_evaluator": source_record(
                PROJECT_DIR / "24_evaluate_v3_2_locked_holdout.py",
                "locked evaluator entry point"),
            "configuration": source_record(
                PROJECT_DIR / "config.py", "project path and sensor config"),
            "hash_verifier": source_record(
                PROJECT_DIR / "deployment_manifest.py",
                "hash and schema verifier source"),
            "requirements": source_record(
                PROJECT_DIR / "requirements.txt",
                "project minimum dependency declarations"),
            "benchmark_generator": source_record(
                PROJECT_DIR / "v3_data.py",
                "synthetic benchmark workload generator"),
        },
        "build": {
            "command_argv": [sys.executable, *sys.argv],
            "git": source_git,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "V4 reuses V3.2 weights and does not improve locked recall.",
            "The approximately 1 Hz source data cannot validate subsecond faults.",
            "Raspberry Pi 5 latency, temperature, power, and long-duration "
            "stability require measurement on the physical device.",
            "The historical 43 normal and 20 faulty wafers are not an "
            "independent unseen external validation set.",
            "The runtime accepts CSV replay; a production equipment protocol "
            "adapter remains site-specific.",
            "Runtime parity uses selection data and shared mathematical "
            "components; it validates implementation consistency, not model "
            "accuracy or independent external validity.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    manifest_hash = file_sha256(MANIFEST_PATH)
    MANIFEST_SIDECAR.write_text(
        f"{manifest_hash}  {MANIFEST_PATH.name}\n", encoding="ascii")

    verified, _, verified_hash = load_v4_manifest(MANIFEST_PATH)
    detector = V4MultiscaleDetector.from_manifest(MANIFEST_PATH)
    detector.start_stream("smoke", "smoke", "smoke", "smoke-run")
    print(json.dumps({
        "status": "built_and_verified",
        "manifest_sha256": verified_hash,
        "model_version": verified["model_version"],
        "torchscript_bytes": TORCHSCRIPT_PATH.stat().st_size,
        "parity": parity,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
