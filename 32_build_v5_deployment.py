# -*- coding: utf-8 -*-
"""Build and verify the immutable V5 TorchScript edge deployment package."""
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
from deployment_manifest import file_sha256, sensor_schema_hash
from models import SlidingWindowLSTMAutoEncoder
from v3_features import transform_sequence
from v4_hashing import normalized_text_bytes, normalized_text_sha256
from v5_edge_runtime import V5MultiscaleDetector, load_v5_manifest
from v5_experiment import profile_score_outputs


V5_DIR = OUTPUT_DIR / "v5"
SOURCE_ARTIFACT = V5_DIR / "sliding_window_lstm_ae_v5.pt"
SOURCE_FINAL_REPORT = V5_DIR / "locked_holdout_v5.json"
SOURCE_CALIBRATION = V5_DIR / "normal_calibration_v5.json"
SOURCE_SELECTION_REPORT = (
    V5_DIR / "experiment_dynamic_grouped_final" / "selection_report.json")
SOURCE_PROTOCOL = V5_DIR / "selection_protocol_v5.json"
SOURCE_SELECTION_DATA = V5_DIR / "selection_data_v5.npz"
SOURCE_STATS = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
TORCHSCRIPT_PATH = V5_DIR / "sliding_window_lstm_ae_v5.ts"
PARITY_REPORT_PATH = V5_DIR / "runtime_parity_v5.json"
ENVIRONMENT_PATH = V5_DIR / "deployment_environment_v5.json"
MANIFEST_PATH = V5_DIR / "deployment_manifest_v5.json"
MANIFEST_SIDECAR = MANIFEST_PATH.with_suffix(".sha256")
EXPECTED_HASHES = {
    "source_artifact": (
        "a59a6c1ae1e828bce72dbec396806f36344dcaf326e9cc04df0bf560258085f4"),
    "final_report": (
        "7c323d2213b4a486eb0c743d0c0a643a5226d5795e422fef0eae5a8b7bb6d0e8"),
    "calibration": (
        "1cb4752bae3b17fc3dfca2ffa1a7c06a4d5768f57a6c039cf80babb6b454d7f4"),
    "selection_report": (
        "f26104072f632fc63458e4223278fd85901627ab088e4e71fc99c0e85187afea"),
    "protocol": (
        "ff1498a63c9d804ae7cdb4ea8bd10227e393366dd87887bcdf12a33d911bb309"),
    "selection_data": (
        "bd5ca5be9090bbb474b17ca09330734a91a31255fc77e8b64ff09380987d621b"),
    "statistics_normalized": (
        "5e9b3aa0cbec720abb4871c771b11584066fb1fab235767ed92e63557bf2f76b"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path, document):
    payload = json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(PROJECT_DIR.resolve()).as_posix()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def git_value(*args):
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True,
        text=True, encoding="utf-8")
    return result.stdout.strip()


def git_record():
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "status_porcelain": git_value("status", "--porcelain").splitlines(),
    }


def verify_locked_source():
    actual = {
        "source_artifact": file_sha256(SOURCE_ARTIFACT),
        "final_report": file_sha256(SOURCE_FINAL_REPORT),
        "calibration": file_sha256(SOURCE_CALIBRATION),
        "selection_report": file_sha256(SOURCE_SELECTION_REPORT),
        "protocol": file_sha256(SOURCE_PROTOCOL),
        "selection_data": file_sha256(SOURCE_SELECTION_DATA),
        "statistics_normalized": normalized_text_sha256(SOURCE_STATS),
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(
            f"V5 deployment source mismatch: {actual} != {EXPECTED_HASHES}")
    artifact = torch.load(
        SOURCE_ARTIFACT, map_location="cpu", weights_only=False)
    final_report = json.loads(
        SOURCE_FINAL_REPORT.read_text(encoding="utf-8"))
    calibration = json.loads(
        SOURCE_CALIBRATION.read_text(encoding="utf-8"))
    selection = json.loads(
        SOURCE_SELECTION_REPORT.read_text(encoding="utf-8"))
    if not final_report["v5"]["acceptance_gate"]["passed"]:
        raise RuntimeError("locked V5 result did not pass its gate")
    if not selection["release"]["gate"]["passed"]:
        raise RuntimeError("selection V5 result did not pass its gate")
    threshold = float(final_report["v5"]["operating_point"]["threshold"])
    if not (
        float(artifact["threshold"]) == threshold ==
        float(calibration["new_calibrated_threshold"])
    ):
        raise RuntimeError("V5 artifact, calibration, and report thresholds differ")
    if calibration["calibrated_artifact_sha256"] != actual["source_artifact"]:
        raise RuntimeError("V5 calibration does not bind the source artifact")
    report_profiles = final_report["v5"]["operating_point"]["profiles"]
    artifact_profiles = [{
        key: profile[key] for key in (
            "window_size", "score_mode", "feature_group", "feature_indices",
            "persistence_required", "persistence_span", "base_scale")
    } for profile in artifact["profiles"]]
    if artifact_profiles != report_profiles:
        raise RuntimeError("V5 artifact profiles differ from locked report")
    return artifact, final_report, actual


def model_from_artifact(artifact):
    model = SlidingWindowLSTMAutoEncoder(
        len(artifact["feature_spec"]["feature_names"]),
        artifact["hidden_size"], artifact["latent_size"])
    model.load_state_dict(artifact["state_dict"])
    return model.eval()


def export_torchscript(model, feature_count):
    traced = torch.jit.trace(
        model, torch.zeros((1, 64, feature_count), dtype=torch.float32),
        strict=True)
    traced.save(str(TORCHSCRIPT_PATH))
    scripted = torch.jit.load(str(TORCHSCRIPT_PATH), map_location="cpu")
    maximum_difference = 0.0
    generator = torch.Generator().manual_seed(740101)
    for window_size in (4, 16, 64):
        sample = torch.randn(
            (4, window_size, feature_count), generator=generator)
        with torch.no_grad():
            expected = model(sample)
            actual = scripted(sample)
        maximum_difference = max(
            maximum_difference,
            float(torch.max(torch.abs(expected - actual)).item()))
    return scripted, maximum_difference


def profile_contracts(artifact):
    output = []
    for index, profile in enumerate(artifact["profiles"], start=1):
        output.append({
            "profile_id": (
                f"p{index}_w{profile['window_size']}_"
                f"{profile['score_mode']}_"
                f"{profile['feature_group']}_"
                f"{profile['persistence_required']}of"
                f"{profile['persistence_span']}"),
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
        "subsecond_claim": "not supported; source cadence is about 1 Hz",
    }


def parity_sequences():
    raw = np.load(SOURCE_SELECTION_DATA, allow_pickle=True)
    normal = list(raw["X_val"])
    anomaly = list(raw["X_val_anom"])
    labels = np.asarray(raw["y_val_anom"], dtype=int)
    selected = [
        (f"normal_{index}", sequence)
        for index, sequence in enumerate(normal[:10])
    ]
    for kind in ANOMALY_NAMES:
        indices = np.flatnonzero(labels == kind)
        selected.extend([
            (f"anomaly_{kind}_{local}", anomaly[index])
            for local, index in enumerate(indices[:10])
        ])
    return selected


ANOMALY_NAMES = (1, 2, 3)


def runtime_parity(model, scripted, artifact, profiles, timing):
    maximum_difference = 0.0
    compared_scores = 0
    alarm_mismatches = 0
    first_alarm_matches = []
    sequence_records = []
    nominal = timing["nominal_interval_seconds"]
    names = artifact["feature_spec"]["raw_sensor_names"]
    for sequence_id, sequence in parity_sequences():
        transformed = transform_sequence(sequence, artifact["feature_spec"])
        offline = profile_score_outputs(model, [transformed], profiles)
        detector = V5MultiscaleDetector(
            scripted, artifact["feature_spec"], profiles,
            artifact["threshold"], timing, "v5-parity",
            file_sha256(TORCHSCRIPT_PATH))
        detector.start_stream(
            sequence_id, "parity", "offline", f"parity:{sequence_id}")
        first_runtime_alarm = None
        first_offline_alarm = None
        for sample_index, row in enumerate(sequence):
            result = detector.update(
                dict(zip(names, row)), sample_index * nominal)
            available = []
            for output in offline:
                profile_id = output["profile"]["profile_id"]
                curve_index = sample_index - output["first_sample"]
                actual = result["profiles"][profile_id]["score"]
                if curve_index < 0:
                    if actual is not None:
                        raise AssertionError(
                            f"{profile_id} became ready too early")
                    continue
                expected = float(output["curves"][0][curve_index])
                if actual is None:
                    raise AssertionError(f"{profile_id} runtime score missing")
                maximum_difference = max(
                    maximum_difference, abs(expected - actual))
                compared_scores += 1
                available.append(expected)
            expected_alarm = bool(
                available and max(available) > artifact["threshold"])
            if expected_alarm != result["alarm"]:
                alarm_mismatches += 1
            if expected_alarm and first_offline_alarm is None:
                first_offline_alarm = sample_index
            if result["alarm"] and first_runtime_alarm is None:
                first_runtime_alarm = sample_index
        first_match = first_runtime_alarm == first_offline_alarm
        first_alarm_matches.append(first_match)
        sequence_records.append({
            "sequence_id": sequence_id,
            "first_offline_alarm": first_offline_alarm,
            "first_runtime_alarm": first_runtime_alarm,
            "match": first_match,
        })
    passed = (
        maximum_difference <= 1e-5 and alarm_mismatches == 0 and
        all(first_alarm_matches))
    return {
        "status": "pass" if passed else "fail",
        "sequence_count": len(sequence_records),
        "compared_profile_scores": compared_scores,
        "maximum_absolute_profile_score_difference": maximum_difference,
        "alarm_decision_mismatches": alarm_mismatches,
        "all_first_alarm_indices_match": all(first_alarm_matches),
        "sequences": sequence_records,
    }


def artifact_record(path, role, hash_mode=None):
    path = Path(path)
    if hash_mode is None:
        hash_mode = (
            "normalized_text_sha256"
            if path.suffix.lower() == ".json" else "sha256")
    if hash_mode == "normalized_text_sha256":
        digest = normalized_text_sha256(path)
        size = len(normalized_text_bytes(path))
    elif hash_mode == "sha256":
        digest = file_sha256(path)
        size = path.stat().st_size
    else:
        raise ValueError(f"unsupported hash mode: {hash_mode}")
    return {
        "path": relative(path), "sha256": digest,
        "hash_mode": hash_mode, "bytes": size, "role": role,
    }


def source_record(path, role):
    return {
        "path": relative(path),
        "sha256": normalized_text_sha256(path),
        "hash_mode": "normalized_text_sha256",
        "bytes": len(normalized_text_bytes(path)),
        "role": role,
    }


def environment_record():
    packages = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine()},
        "packages": packages,
        "runtime_minimum_direct_dependencies": ["numpy", "torch"],
    }


def main():
    args = parse_args()
    V5_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists() and not args.force:
        document, _, manifest_hash = load_v5_manifest(MANIFEST_PATH)
        print(json.dumps({
            "status": "existing_verified_v5_package",
            "manifest_sha256": manifest_hash,
            "model_version": document["model_version"],
        }, indent=2))
        return
    source_git = git_record()
    if source_git["status_porcelain"]:
        raise RuntimeError(
            "V5 deployment build requires a clean source worktree: "
            f"{source_git['status_porcelain']}")
    artifact, final_report, locked_hashes = verify_locked_source()
    statistics = json.loads(SOURCE_STATS.read_text(encoding="utf-8"))
    model = model_from_artifact(artifact)
    scripted, eager_script_difference = export_torchscript(
        model, len(artifact["feature_spec"]["feature_names"]))
    profiles = profile_contracts(artifact)
    timing = timing_contract(statistics)
    parity = runtime_parity(model, scripted, artifact, profiles, timing)
    parity.update({
        "eager_torchscript_maximum_absolute_output_difference": (
            eager_script_difference),
        "source_model_sha256": locked_hashes["source_artifact"],
        "torchscript_model_sha256": file_sha256(TORCHSCRIPT_PATH),
        "ensemble_threshold": float(artifact["threshold"]),
    })
    if (
        parity["status"] != "pass" or eager_script_difference > 1e-5 or
        parity["maximum_absolute_profile_score_difference"] > 1e-5
    ):
        raise RuntimeError(f"V5 runtime parity failed: {parity}")
    write_json(PARITY_REPORT_PATH, parity)
    write_json(ENVIRONMENT_PATH, environment_record())
    feature_spec = json_ready(artifact["feature_spec"])
    feature_spec["raw_schema_hash"] = sensor_schema_hash(
        feature_spec["raw_sensor_names"])
    manifest = {
        "manifest_version": 5,
        "model_version": "v5-runtime-locked-v5-weights",
        "status": "paper_edge_runtime_prototype_not_production_release",
        "lineage": {
            "weights": "locked V5 weights, unchanged",
            "threshold": "locked V5 normal-only calibration, unchanged",
            "profiles": "locked V5 profiles, unchanged",
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
            "stream_boundary": "start_stream/reset required per process stream",
        },
        "timing_contract": timing,
        "artifacts": {
            "source_checkpoint": artifact_record(
                SOURCE_ARTIFACT, "locked V5 PyTorch checkpoint"),
            "torchscript_model": artifact_record(
                TORCHSCRIPT_PATH, "V5 CPU streaming model"),
            "locked_final_report": artifact_record(
                SOURCE_FINAL_REPORT, "one-time V5 locked holdout report"),
            "normal_calibration": artifact_record(
                SOURCE_CALIBRATION, "V5 normal-only calibration"),
            "selection_report": artifact_record(
                SOURCE_SELECTION_REPORT, "V5 three-seed selection report"),
            "selection_protocol": artifact_record(
                SOURCE_PROTOCOL, "V5 preregistered selection protocol"),
            "selection_data": artifact_record(
                SOURCE_SELECTION_DATA, "V5 selection cohorts"),
            "source_statistics": artifact_record(
                SOURCE_STATS, "V3.2 real-informed source statistics"),
            "runtime_parity": artifact_record(
                PARITY_REPORT_PATH, "V5 offline/TorchScript parity"),
            "build_environment": artifact_record(
                ENVIRONMENT_PATH, "exact V5 build environment"),
        },
        "source_provenance": {
            "runtime": source_record(
                PROJECT_DIR / "v5_edge_runtime.py", "V5 runtime source"),
            "base_runtime": source_record(
                PROJECT_DIR / "v4_edge_runtime.py", "shared safety runtime"),
            "builder": source_record(Path(__file__), "V5 builder source"),
            "runtime_tests": source_record(
                PROJECT_DIR / "tests" / "test_v5_runtime.py",
                "V5 runtime tests"),
            "features": source_record(
                PROJECT_DIR / "v3_features.py", "causal feature transform"),
            "model": source_record(PROJECT_DIR / "models.py", "model source"),
            "scoring": source_record(
                PROJECT_DIR / "v5_experiment.py", "V5 score source"),
            "locked_evaluator": source_record(
                PROJECT_DIR / "31_evaluate_v5_locked_holdout.py",
                "one-time locked evaluator"),
            "calibrator": source_record(
                PROJECT_DIR / "30_calibrate_v5.py", "normal calibrator"),
            "trainer": source_record(
                PROJECT_DIR / "29_train_select_v5.py", "V5 trainer"),
            "generator": source_record(
                PROJECT_DIR / "28_generate_v5_selection.py",
                "V5 selection generator"),
            "configuration": source_record(
                PROJECT_DIR / "config.py", "project configuration"),
            "hashing": source_record(
                PROJECT_DIR / "v4_hashing.py", "cross-platform hashing"),
        },
        "build": {
            "command_argv": [sys.executable, *sys.argv],
            "git": source_git,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "V5 locked holdout uses the same synthetic generator family.",
            "V5 has not yet been benchmarked on the physical Raspberry Pi 5; "
            "V4 Pi measurements cannot be relabeled as V5 performance.",
            "Approximately 1 Hz data do not validate subsecond faults.",
            "No independent new real faulty-wafer validation is available.",
            "Equipment transport, external alarms, power, and long-duration "
            "stability remain deployment-site experiments.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    manifest_hash = normalized_text_sha256(MANIFEST_PATH)
    MANIFEST_SIDECAR.write_bytes(
        f"{manifest_hash}  {MANIFEST_PATH.name}  "
        "normalized_text_sha256\n".encode("utf-8"))
    load_v5_manifest(MANIFEST_PATH)
    print(json.dumps({
        "status": "built_verified_v5_package",
        "manifest_sha256": manifest_hash,
        "model_version": manifest["model_version"],
        "parity": parity,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
