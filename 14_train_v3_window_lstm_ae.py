# -*- coding: utf-8 -*-
"""Train and select V3 on chronology-corrected selection data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import DEVICE, SlidingWindowLSTMAutoEncoder, _buckets_by_length
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    calibrate_sensor_errors,
    event_decisions,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)
from models import sliding_window_error_summaries
from v3_features import (
    fit_feature_spec,
    sample_training_windows,
    transform_sequences,
)


V3_DIR = OUTPUT_DIR / "v3"
DATA_PATH = V3_DIR / "selection_data_v3.npz"
WINDOW_SIZES = (8, 16, 32, 64)
PERSISTENCE_OPTIONS = ((1, 1), (2, 3), (3, 5))
SCORE_MODES = ("mean", "max", "delta_mean", "delta_max")
TARGET_VALIDATION_FPR = 0.005
HIDDEN_SIZE = 64
LATENT_SIZE = 16
ANOMALY_NAMES = {
    1: "A: dynamic slew excursion",
    2: "B: oscillation",
    3: "C: drift",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="34001")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-size", type=int, default=4)
    parser.add_argument("--dynamic-weight", type=float, default=0.5)
    parser.add_argument("--experiment", default="dynamic")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.seeds = [
        int(value.strip()) for value in args.seeds.split(",")
        if value.strip()
    ]
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if args.epochs < 10:
        parser.error("--epochs must be at least 10")
    if args.dynamic_weight < 0:
        parser.error("--dynamic-weight must be nonnegative")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data():
    raw = np.load(DATA_PATH, allow_pickle=True)
    return {
        "train": list(raw["X_train"]),
        "validation": list(raw["X_val"]),
        "validation_anomaly": list(raw["X_val_anom"]),
        "labels": np.asarray(raw["y_val_anom"], dtype=int),
        "metadata": [json.loads(str(item)) for item in raw["val_metadata"]],
        "sensor_names": [str(item) for item in raw["sensor_names"]],
    }


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def make_buckets(windows):
    return {
        length: torch.as_tensor(
            np.stack([windows[index] for index in indices]),
            dtype=torch.float32).to(DEVICE)
        for length, indices in _buckets_by_length(windows).items()
    }


def reconstruction_loss(model, batch, dynamic_weight):
    reconstructed = model(batch)
    value_loss = torch.mean((reconstructed - batch) ** 2)
    delta_loss = torch.mean((
        torch.diff(reconstructed, dim=1) - torch.diff(batch, dim=1)
    ) ** 2)
    return value_loss + dynamic_weight * delta_loss


def train_model(model, train_windows, validation_windows, seed, epochs,
                batch_size, dynamic_weight):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_buckets = make_buckets(train_windows)
    validation_buckets = make_buckets(validation_windows)
    checkpoints = []
    history = {"train": [], "validation": []}
    checkpoint_interval = 10
    for epoch in range(1, epochs + 1):
        model.train()
        batches = []
        for tensor in train_buckets.values():
            order = rng.permutation(len(tensor))
            batches.extend([
                tensor[order[start:start + batch_size]]
                for start in range(0, len(order), batch_size)
            ])
        rng.shuffle(batches)
        losses = []
        for batch in batches:
            optimizer.zero_grad()
            loss = reconstruction_loss(model, batch, dynamic_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = np.mean([
                float(reconstruction_loss(
                    model, tensor, dynamic_weight).item())
                for tensor in validation_buckets.values()
            ])
        history["train"].append(float(np.mean(losses)))
        history["validation"].append(float(validation_loss))
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            checkpoints.append((epoch, cpu_state(model)))
            print(
                f"seed={seed} epoch={epoch}/{epochs} "
                f"train={history['train'][-1]:.5f} "
                f"validation={validation_loss:.5f}")
    return history, checkpoints


def score_summaries(model, sequences, window_size, score_indices):
    summaries = sliding_window_error_summaries(
        model, sequences, window_size)
    return {
        mode: [values[:, score_indices] for values in summaries[mode]]
        for mode in SCORE_MODES
    }


def select_configuration(model, checkpoints, validation, anomalies, labels,
                         metadata, score_indices):
    all_labels = np.concatenate([
        np.zeros(len(validation), dtype=int), labels])
    best = None
    candidates = []
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        model.to(DEVICE)
        for window_size in WINDOW_SIZES:
            normal_summaries = score_summaries(
                model, validation, window_size, score_indices)
            anomaly_summaries = score_summaries(
                model, anomalies, window_size, score_indices)
            for score_mode in SCORE_MODES:
                normal_errors = normal_summaries[score_mode]
                anomaly_errors = anomaly_summaries[score_mode]
                calibration = calibrate_sensor_errors(normal_errors)
                normal_raw = sensor_error_score_curves(
                    normal_errors, calibration)
                anomaly_raw = sensor_error_score_curves(
                    anomaly_errors, calibration)
                for required, span in PERSISTENCE_OPTIONS:
                    normal_curves = apply_persistence(
                        normal_raw, required, span)
                    anomaly_curves = apply_persistence(
                        anomaly_raw, required, span)
                    normal_event_scores = np.asarray([
                        curve.max() for curve in normal_curves])
                    threshold = threshold_for_target_fpr(
                        normal_event_scores, TARGET_VALIDATION_FPR)
                    curves = normal_curves + anomaly_curves
                    first_sample = window_size - 1 + span - 1
                    predictions, _, pre_onset, event_scores = event_decisions(
                        curves, threshold, first_sample, all_labels,
                        ([{} for _ in validation] + metadata),
                        evidence_span=span)
                    metrics = binary_event_metrics(
                        all_labels, predictions, event_scores)
                    per_type = {
                        ANOMALY_NAMES[kind]: float(np.mean(
                            predictions[all_labels == kind]))
                        for kind in ANOMALY_NAMES
                    }
                    macro_recall = float(np.mean(list(per_type.values())))
                    candidate = {
                        "epoch": epoch,
                        "window_size": window_size,
                        "score_mode": score_mode,
                        "persistence_required": required,
                        "persistence_span": span,
                        "threshold": float(threshold),
                        "validation_precision": metrics["precision"],
                        "validation_recall": metrics["recall"],
                        "validation_f1": metrics["f1"],
                        "validation_fpr": metrics["fpr"],
                        "validation_macro_recall": macro_recall,
                        "validation_min_type_recall": min(per_type.values()),
                        "validation_per_type_recall": per_type,
                        "validation_pre_onset_rate": float(np.mean(
                            pre_onset[all_labels > 0])),
                    }
                    rank = (
                        candidate["validation_macro_recall"],
                        candidate["validation_min_type_recall"],
                        candidate["validation_recall"],
                        -candidate["validation_fpr"],
                        candidate["validation_f1"],
                        -candidate["window_size"],
                    )
                    candidates.append(candidate)
                    if best is None or rank > best["rank"]:
                        best = {
                            **candidate,
                            "calibration": calibration.copy(),
                            "state_dict": cpu_state(model),
                            "rank": rank,
                        }
    diagnostics = {
        "best_by_window": {
            str(window_size): serialize_candidate(max(
                (item for item in candidates
                 if item["window_size"] == window_size),
                key=lambda item: (
                    item["validation_macro_recall"],
                    item["validation_min_type_recall"],
                    -item["validation_fpr"],
                )))
            for window_size in WINDOW_SIZES
        },
        "best_by_anomaly_type": {
            name: serialize_candidate(max(
                candidates,
                key=lambda item: (
                    item["validation_per_type_recall"][name],
                    item["validation_macro_recall"],
                    -item["validation_fpr"],
                )))
            for name in ANOMALY_NAMES.values()
        },
    }
    return best, diagnostics


def serialize_candidate(candidate):
    return {
        key: value for key, value in candidate.items()
        if key not in ("calibration", "state_dict", "rank")
    }


def main():
    args = parse_args()
    V3_DIR.mkdir(exist_ok=True)
    data = load_data()
    feature_spec = fit_feature_spec(data["train"], data["sensor_names"])
    train = transform_sequences(data["train"], feature_spec)
    validation = transform_sequences(data["validation"], feature_spec)
    anomalies = transform_sequences(
        data["validation_anomaly"], feature_spec)
    validation_windows = sample_training_windows(
        validation[:500], WINDOW_SIZES, 2, seed=35001)
    results = []
    start_time = time.time()
    for seed in args.seeds:
        train_windows = sample_training_windows(
            train, WINDOW_SIZES, args.samples_per_size, seed=seed)
        torch.manual_seed(seed)
        model = SlidingWindowLSTMAutoEncoder(
            len(feature_spec["feature_names"]), HIDDEN_SIZE, LATENT_SIZE)
        history, checkpoints = train_model(
            model, train_windows, validation_windows, seed, args.epochs,
            args.batch_size, args.dynamic_weight)
        best, diagnostics = select_configuration(
            model, checkpoints, validation, anomalies, data["labels"],
            data["metadata"], feature_spec["score_feature_indices"])
        result = {
            "seed": seed,
            "history": history,
            "selection": serialize_candidate(best),
            "state_dict": best["state_dict"],
            "calibration": best["calibration"],
            "diagnostics": diagnostics,
        }
        results.append(result)
        print(
            f"seed={seed} selected W={best['window_size']} "
            f"mode={best['score_mode']} "
            f"macro_recall={best['validation_macro_recall']:.3f} "
            f"FPR={best['validation_fpr']:.3%} "
            f"types={best['validation_per_type_recall']}")

    release = max(results, key=lambda item: (
        item["selection"]["validation_macro_recall"],
        item["selection"]["validation_min_type_recall"],
        -item["selection"]["validation_fpr"],
    ))
    experiment_dir = V3_DIR / f"experiment_{args.experiment}"
    experiment_dir.mkdir(exist_ok=True)
    artifact_path = experiment_dir / "candidate.pt"
    torch.save({
        "state_dict": release["state_dict"],
        "feature_spec": feature_spec,
        "calibration": release["calibration"],
        "selection": release["selection"],
        "seed": release["seed"],
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "dynamic_weight": args.dynamic_weight,
        "data_sha256": file_sha256(DATA_PATH),
    }, artifact_path)
    report = {
        "protocol": {
            "selection_data_only": True,
            "target_validation_fpr": TARGET_VALIDATION_FPR,
            "window_training": True,
            "dynamic_reconstruction_loss_weight": args.dynamic_weight,
            "selection_objective": (
                "macro anomaly recall, then minimum type recall, under the "
                "normal-only validation FPR operating point"),
            "holdout_access": "none",
        },
        "feature_spec": feature_spec,
        "seeds": [
            {
                "seed": item["seed"],
                "selection": item["selection"],
                "final_train_loss": item["history"]["train"][-1],
                "final_validation_loss": item["history"]["validation"][-1],
                "diagnostics": item["diagnostics"],
            }
            for item in results
        ],
        "release_seed": release["seed"],
        "release_selection": release["selection"],
        "candidate_sha256": file_sha256(artifact_path),
        "elapsed_seconds": time.time() - start_time,
    }
    report_path = experiment_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if args.publish:
        published_path = V3_DIR / "sliding_window_lstm_ae_v3.pt"
        published_path.write_bytes(artifact_path.read_bytes())
        print(f"Published frozen V3 checkpoint: {published_path}")
    print(json.dumps(report["release_selection"], ensure_ascii=False, indent=2))
    print(f"Selection report saved: {report_path}")


if __name__ == "__main__":
    main()
