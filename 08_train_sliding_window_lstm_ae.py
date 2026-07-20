# -*- coding: utf-8 -*-
"""Step 8: causal sliding-window LSTM-AE for edge early warning.

The autoencoder first learns complete normal process trajectories in Step 3.
At deployment it reconstructs only a causal trailing window. Window length,
error reduction, and persistence are selected on a separate validation set
under an empirical event-FPR constraint. The fixed test set is evaluated once
per seed after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from config import COLORS, FIGURE_DIR, OUTPUT_DIR, set_plot_style
import matplotlib.pyplot as plt
from edge_window_runtime import SlidingWindowAnomalyDetector
from models import (DEVICE, SlidingWindowLSTMAutoEncoder,
                    sliding_window_error_summaries, sliding_window_errors)
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    calibrate_sensor_errors,
    event_decisions,
    sensor_error_score_curves,
    threshold_for_target_fpr,
)


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
HIDDEN_SIZE = 64
LATENT_SIZE = 16
WINDOW_SIZES = (8, 16, 32, 64)
SCORE_MODES = ("last", "mean", "max", "delta_mean", "delta_max")
PERSISTENCE_OPTIONS = ((1, 1), (2, 3), (3, 5))
TARGET_VALIDATION_FPR = 0.01
PIPELINE_VERSION = 7
ANOMALY_LABELS = {
    1: "A: 暫態響應速度異常",
    2: "B: 過程震盪",
    3: "C: 緩慢漂移",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated random seeds",
    )
    parser.add_argument("--benchmark-runs", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(item.strip()) for item in args.seeds.split(",")
                  if item.strip()]
    if not args.seeds:
        parser.error("--seeds must contain at least one integer")
    if args.benchmark_runs < 10:
        parser.error("--benchmark-runs must be at least 10")
    return args


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_signature(data_path, seed, source_signature):
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "data_sha256": file_sha256(data_path),
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "window_sizes": WINDOW_SIZES,
        "score_modes": SCORE_MODES,
        "persistence_options": PERSISTENCE_OPTIONS,
        "validation_target_fpr": TARGET_VALIDATION_FPR,
        "seed": int(seed),
        "source_step3_signature": source_signature,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def load_data():
    path = OUTPUT_DIR / "synthetic_data.npz"
    raw = np.load(path, allow_pickle=True)
    if "test_metadata" not in raw.files:
        raise RuntimeError("Run 02_generate_synthetic.py to add onset metadata")
    return {
        "path": path,
        "train": list(raw["X_train"]),
        "val": list(raw["X_val"]),
        "val_anom": list(raw["X_val_anom"]),
        "y_val_anom": np.asarray(raw["y_val_anom"]),
        "val_metadata": [json.loads(str(item))
                         for item in raw["val_metadata"]],
        "test": list(raw["X_test"]),
        "y_test": np.asarray(raw["y_test"]),
        "test_metadata": [json.loads(str(item))
                          for item in raw["test_metadata"]],
        "sensor_names": [str(item) for item in raw["sensor_names"]],
    }


def normalize(items, mean, std):
    return [(np.asarray(item, dtype=np.float32) - mean) / std
            for item in items]


def score_curves(model, normal_sequences, anomaly_sequences, window_size,
                 persistence_required, persistence_span, score_mode,
                 calib=None):
    normal_errors = sliding_window_errors(
        model, normal_sequences, window_size, score_mode)
    anomaly_errors = sliding_window_errors(
        model, anomaly_sequences, window_size, score_mode)
    if calib is None:
        calib = calibrate_sensor_errors(normal_errors)
    normal_raw = sensor_error_score_curves(normal_errors, calib)
    anomaly_raw = sensor_error_score_curves(anomaly_errors, calib)
    normal_curves = apply_persistence(
        normal_raw, persistence_required, persistence_span)
    anomaly_curves = apply_persistence(
        anomaly_raw, persistence_required, persistence_span)
    return normal_curves, anomaly_curves, calib


def evaluate_configuration(model, normal_sequences, anomaly_sequences,
                           anomaly_labels, anomaly_metadata, window_size,
                           persistence_required, persistence_span,
                           score_mode,
                           threshold=None, calib=None,
                           target_fpr=TARGET_VALIDATION_FPR):
    normal_curves, anomaly_curves, calib = score_curves(
        model, normal_sequences, anomaly_sequences, window_size,
        persistence_required, persistence_span, score_mode, calib)
    normal_event_scores = np.asarray(
        [curve.max() for curve in normal_curves], dtype=np.float64)
    if threshold is None:
        threshold = threshold_for_target_fpr(normal_event_scores, target_fpr)

    labels = np.concatenate([
        np.zeros(len(normal_curves), dtype=int), anomaly_labels])
    metadata = ([{} for _ in normal_curves] + list(anomaly_metadata))
    curves = normal_curves + anomaly_curves
    first_sample = window_size - 1 + persistence_span - 1
    predictions, alerts, pre_onset, event_scores = event_decisions(
        curves, threshold, first_sample, labels, metadata,
        evidence_span=persistence_span)
    metrics = binary_event_metrics(labels, predictions, event_scores)
    metrics.update({
        "threshold": float(threshold),
        "window_size": int(window_size),
        "persistence_required": int(persistence_required),
        "persistence_span": int(persistence_span),
        "score_mode": score_mode,
        "pre_onset_alarm_rate": float(
            pre_onset[labels > 0].mean()) if np.any(labels > 0) else 0.0,
        "per_type_recall": {
            label: float(predictions[labels == anomaly_type].mean())
            for anomaly_type, label in ANOMALY_LABELS.items()
        },
    })
    details = {
        "calib": calib,
        "curves": curves,
        "predictions": predictions,
        "alerts": alerts,
        "pre_onset": pre_onset,
        "event_scores": event_scores,
        "labels": labels,
        "metadata": metadata,
        "first_sample": first_sample,
    }
    return metrics, details


def median_detection_delay(metrics, details, sequences):
    timing = {}
    labels = details["labels"]
    normal_count = int(np.count_nonzero(labels == 0))
    for anomaly_type, name in ANOMALY_LABELS.items():
        combined_indices = np.flatnonzero(labels == anomaly_type)
        records = []
        for combined_index in combined_indices:
            anomaly_index = int(combined_index - normal_count)
            alert = details["alerts"][combined_index]
            if alert is None:
                continue
            onset = details["metadata"][combined_index].get("onset_index")
            length = len(sequences[anomaly_index])
            records.append({
                "delay": None if onset is None else int(alert - int(onset)),
                "progress": float(alert / max(length - 1, 1)),
            })
        timing[name] = {
            "detected": len(records),
            "total": int(len(combined_indices)),
            "median_detection_delay_samples": (
                float(np.median([item["delay"] for item in records
                                 if item["delay"] is not None]))
                if any(item["delay"] is not None for item in records) else None
            ),
            "median_alert_progress": (
                float(np.median([item["progress"] for item in records]))
                if records else None
            ),
        }
    metrics["alert_timing"] = timing


def select_configuration(model, checkpoints, Xva, Xva_anom, yva_anom,
                         val_metadata, window_sizes=WINDOW_SIZES):
    best = None
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        model.to(DEVICE)
        for window_size in window_sizes:
            normal_summaries = sliding_window_error_summaries(
                model, Xva, window_size)
            anomaly_summaries = sliding_window_error_summaries(
                model, Xva_anom, window_size)
            for score_mode in SCORE_MODES:
                normal_errors = normal_summaries[score_mode]
                anomaly_errors = anomaly_summaries[score_mode]
                calib = calibrate_sensor_errors(normal_errors)
                normal_raw = sensor_error_score_curves(normal_errors, calib)
                anomaly_raw = sensor_error_score_curves(anomaly_errors, calib)
                for required, span in PERSISTENCE_OPTIONS:
                    normal_curves = apply_persistence(
                        normal_raw, required, span)
                    anomaly_curves = apply_persistence(
                        anomaly_raw, required, span)
                    normal_scores = np.asarray(
                        [curve.max() for curve in normal_curves])
                    threshold = threshold_for_target_fpr(
                        normal_scores, TARGET_VALIDATION_FPR)
                    labels = np.concatenate([
                        np.zeros(len(normal_curves), dtype=int), yva_anom])
                    curves = normal_curves + anomaly_curves
                    metadata = ([{} for _ in normal_curves] + val_metadata)
                    first_sample = window_size - 1 + span - 1
                    predictions, alerts, pre_onset, scores = event_decisions(
                        curves, threshold, first_sample, labels, metadata,
                        evidence_span=span)
                    metrics = binary_event_metrics(labels, predictions, scores)
                    metrics["pre_onset_alarm_rate"] = float(
                        pre_onset[labels > 0].mean())
                    per_type_recall = {
                        ANOMALY_LABELS[anomaly_type]: float(
                            predictions[labels == anomaly_type].mean())
                        for anomaly_type in ANOMALY_LABELS
                    }
                    candidate = {
                        "epoch": int(epoch),
                        "window_size": int(window_size),
                        "score_mode": score_mode,
                        "persistence_required": int(required),
                        "persistence_span": int(span),
                        "threshold": float(threshold),
                        "calib": calib,
                        "val_precision": metrics["precision"],
                        "val_recall": metrics["recall"],
                        "val_f1": metrics["f1"],
                        "val_fpr": metrics["fpr"],
                        "val_pre_onset_alarm_rate": metrics[
                            "pre_onset_alarm_rate"],
                        "val_per_type_recall": per_type_recall,
                        "val_min_type_recall": min(
                            per_type_recall.values()),
                        "state": {key: value.clone()
                                  for key, value in state.items()},
                        "alerts": alerts,
                    }
                    rank = (
                        candidate["val_recall"],
                        -candidate["val_fpr"],
                        candidate["val_f1"],
                        -candidate["window_size"],
                        -candidate["persistence_span"],
                        -SCORE_MODES.index(candidate["score_mode"]),
                    )
                    if best is None or rank > best["rank"]:
                        candidate["rank"] = rank
                        best = candidate
    if best is None:
        raise RuntimeError("No valid sliding-window configuration")
    print(
        "Selected sliding LSTM-AE: "
        f"epoch={best['epoch']}, W={best['window_size']}, "
        f"score={best['score_mode']}, "
        f"persistence={best['persistence_required']}/{best['persistence_span']}, "
        f"val recall={best['val_recall']:.3f}, "
        f"min-type recall={best['val_min_type_recall']:.3f}, "
        f"val FPR={best['val_fpr']:.3f}"
    )
    return best


def aggregate_results(items):
    output = {}
    for key in ("precision", "recall", "f1", "fpr", "auc",
                "pre_onset_alarm_rate"):
        values = np.asarray([item[key] for item in items], dtype=float)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    output["per_type"] = {}
    for label in ANOMALY_LABELS.values():
        values = np.asarray(
            [item["per_type_recall"][label] for item in items])
        output["per_type"][label] = {
            "recall_mean": float(values.mean()),
            "recall_std": float(values.std()),
        }
        for timing_key in ("median_detection_delay_samples",
                           "median_alert_progress"):
            timing = [item["alert_timing"][label][timing_key]
                      for item in items]
            timing = np.asarray([value for value in timing
                                 if value is not None], dtype=float)
            output["per_type"][label][f"{timing_key}_mean"] = (
                float(timing.mean()) if len(timing) else None)
            output["per_type"][label][f"{timing_key}_std"] = (
                float(timing.std()) if len(timing) else None)
    return output


def export_release(model, checkpoint, sensor_names):
    model = model.cpu().eval()
    window_size = checkpoint["window_size"]
    example = torch.zeros(1, window_size, len(sensor_names))
    traced = torch.jit.trace(model, example)
    model_path = OUTPUT_DIR / "sliding_window_lstm_ae.ts"
    traced.save(str(model_path))
    checkpoint_path = OUTPUT_DIR / "sliding_window_lstm_ae.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, model_path


def benchmark_runtime(checkpoint_path, model_path, runs):
    detector = SlidingWindowAnomalyDetector.from_artifacts(
        checkpoint_path, model_path)
    sample = detector.mean.copy()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(detector.window_size + detector.persistence_span + 100):
            detector.update(sample)
        durations = []
        for _ in range(runs):
            start = time.perf_counter_ns()
            detector.update(sample)
            durations.append((time.perf_counter_ns() - start) / 1000.0)
    finally:
        torch.set_num_threads(previous_threads)
    values = np.asarray(durations)
    return {
        "device_scope": "host_cpu_not_raspberry_pi",
        "includes": (
            "normalization, trailing-window update, TorchScript inference, "
            "per-sensor scoring, persistence, and alarm decision"
        ),
        "excludes": "sensor transport, persistence storage, and alert delivery",
        "runs": int(runs),
        "mean_us": float(values.mean()),
        "p50_us": float(np.percentile(values, 50)),
        "p95_us": float(np.percentile(values, 95)),
        "p99_us": float(np.percentile(values, 99)),
        "torchscript_size_kib": float(model_path.stat().st_size / 1024),
        "parameter_count": int(sum(
            parameter.numel() for parameter in detector.model.parameters())),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "threads_during_benchmark": 1,
    }


def verify_artifact_parity(checkpoint_path, model_path, sequences):
    """Verify eager offline scores match the exported stateful runtime."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model = SlidingWindowLSTMAutoEncoder(
        len(checkpoint["mean"]), checkpoint["hidden_size"],
        checkpoint["latent_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized = normalize(
        sequences, checkpoint["mean"], checkpoint["std"])
    offline_errors = sliding_window_errors(
        model, normalized, checkpoint["window_size"],
        checkpoint["score_mode"])
    offline_raw = sensor_error_score_curves(
        offline_errors, checkpoint["calib"])
    offline_persistent = apply_persistence(
        offline_raw, checkpoint["persistence_required"],
        checkpoint["persistence_span"])

    max_raw_difference = 0.0
    max_persistent_difference = 0.0
    decisions_match = True
    for sequence, expected_raw, expected_persistent in zip(
            sequences, offline_raw, offline_persistent):
        detector = SlidingWindowAnomalyDetector.from_artifacts(
            checkpoint_path, model_path)
        emitted = [detector.update(sample) for sample in sequence]
        actual_raw = np.asarray([
            item["raw_score"] for item in emitted
            if item["raw_score"] is not None])
        actual_persistent = np.asarray([
            item["score"] for item in emitted if item["score"] is not None])
        max_raw_difference = max(
            max_raw_difference,
            float(np.max(np.abs(actual_raw - expected_raw))))
        max_persistent_difference = max(
            max_persistent_difference,
            float(np.max(np.abs(actual_persistent - expected_persistent))))
        expected_decisions = expected_persistent > checkpoint["threshold"]
        actual_decisions = np.asarray([
            item["alarm"] for item in emitted if item["score"] is not None])
        decisions_match = decisions_match and bool(np.array_equal(
            expected_decisions, actual_decisions))

    tolerance = 1e-5
    return {
        "sequences_checked": int(len(sequences)),
        "max_raw_score_abs_difference": max_raw_difference,
        "max_persistent_score_abs_difference": max_persistent_difference,
        "alarm_decisions_match": decisions_match,
        "absolute_tolerance": tolerance,
        "passed": bool(
            max_raw_difference <= tolerance and
            max_persistent_difference <= tolerance and
            decisions_match),
    }


def plot_release(metrics, details, anomaly_sequences):
    set_plot_style()
    labels = list(ANOMALY_LABELS.values())
    short_labels = ["A: 響應速度", "B: 過程震盪", "C: 緩慢漂移"]
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 13),
        gridspec_kw={"height_ratios": [0.9, 1, 1, 1]})

    recalls = [metrics["per_type_recall"][label] for label in labels]
    bars = axes[0].bar(
        short_labels, recalls,
        color=[COLORS["series3"], COLORS["series4"], COLORS["series5"]])
    for bar, value in zip(bars, recalls):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value + 0.025,
            f"{value:.2f}", ha="center", color=COLORS["ink2"])
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Event recall")
    axes[0].set_title(
        "Sliding-window LSTM-AE "
        f"(F1={metrics['f1']:.3f}, FPR={metrics['fpr']:.3f})")

    normal_count = int(np.count_nonzero(details["labels"] == 0))
    for axis, anomaly_type, label in zip(axes[1:], ANOMALY_LABELS, labels):
        combined = np.flatnonzero(
            (details["labels"] == anomaly_type) &
            (details["predictions"] == 1))
        if len(combined):
            combined_index = int(combined[0])
        else:
            combined = np.flatnonzero(details["labels"] == anomaly_type)
            combined_index = int(combined[np.argmax(
                details["event_scores"][combined])])
        anomaly_index = combined_index - normal_count
        curve = details["curves"][combined_index]
        sequence_length = len(anomaly_sequences[anomaly_index])
        sample_indices = np.arange(len(curve)) + details["first_sample"]
        progress = sample_indices / max(sequence_length - 1, 1)
        axis.plot(progress, curve, color=COLORS["faulty"], linewidth=1.8,
                  label="Causal anomaly score")
        axis.axhline(metrics["threshold"], color=COLORS["ink"],
                     linestyle="--", linewidth=1.2, label="Alarm threshold")
        onset = details["metadata"][combined_index].get("onset_fraction")
        if onset is not None:
            axis.axvline(onset, color=COLORS["series3"], linestyle=":",
                         linewidth=1.4, label="Synthetic injection onset")
        alert = details["alerts"][combined_index]
        if alert is not None:
            curve_index = int(alert - details["first_sample"])
            axis.scatter(
                [alert / max(sequence_length - 1, 1)], [curve[curve_index]],
                s=55, color=COLORS["faulty"], zorder=5, label="First alarm")
        axis.set_xlim(0, 1)
        axis.set_ylabel(label)
        axis.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    axes[-1].set_xlabel("Normalized process progress (0=start, 1=end)")
    fig.suptitle(
        "Causal sliding-window LSTM-AE early-warning timelines", fontsize=13)
    fig.tight_layout()
    output = FIGURE_DIR / "08_sliding_window_results.png"
    fig.savefig(output, bbox_inches="tight")
    print(f"Figure saved: {output}")


def main():
    args = parse_args()
    raw = load_data()
    train_points = np.concatenate(raw["train"]).astype(np.float32)
    mean = train_points.mean(axis=0)
    std = train_points.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xva = normalize(raw["val"], mean, std)
    Xva_anom = normalize(raw["val_anom"], mean, std)
    Xte = normalize(raw["test"], mean, std)
    Xte_normal = [sequence for sequence, label in zip(Xte, raw["y_test"])
                  if label == 0]
    Xte_anomaly = [sequence for sequence, label in zip(Xte, raw["y_test"])
                   if label > 0]
    yte_anomaly = raw["y_test"][raw["y_test"] > 0]
    test_anomaly_metadata = [
        metadata for metadata, label in zip(
            raw["test_metadata"], raw["y_test"]) if label > 0]

    n_features = len(mean)
    cache_dir = OUTPUT_DIR / "sliding_window_seeds"
    cache_dir.mkdir(exist_ok=True)
    source_dir = OUTPUT_DIR / "seeds"
    seed_results = []
    print(f"Evaluation device: {DEVICE}")
    print("Source models: Step 3 LSTM-AEs trained on complete normal cycles")

    for seed in args.seeds:
        source_json = source_dir / f"seed_{seed}.json"
        source_pt = source_dir / f"seed_{seed}.pt"
        if not source_json.exists() or not source_pt.exists():
            raise FileNotFoundError(
                f"Missing Step 3 seed {seed}; run 03_train_lstm_ae.py first")
        source_metrics = json.loads(source_json.read_text(encoding="utf-8"))
        source_signature = source_metrics.get("run_signature")
        if not source_signature:
            raise RuntimeError(
                f"Step 3 seed {seed} has no reproducibility signature")
        signature, payload = run_signature(
            raw["path"], seed, source_signature)
        json_path = cache_dir / f"seed_{seed}.json"
        model_path = cache_dir / f"seed_{seed}.pt"
        cached = None
        if not args.force and json_path.exists() and model_path.exists():
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("run_signature") != signature:
                cached = None
        if cached is not None:
            seed_results.append(cached)
            print(f"seed {seed}: matching cache, recall={cached['recall']:.3f}")
            continue

        print(f"\n===== sliding-window seed {seed} =====")
        source_artifact = torch.load(
            source_pt, map_location="cpu", weights_only=False)
        model = SlidingWindowLSTMAutoEncoder(
            n_features, HIDDEN_SIZE, LATENT_SIZE)
        model.load_state_dict(source_artifact["state_dict"])
        model.to(DEVICE)
        best = select_configuration(
            model,
            [(source_metrics["best_epoch"], source_artifact["state_dict"])],
            Xva, Xva_anom, raw["y_val_anom"], raw["val_metadata"])
        model.load_state_dict(best["state"])
        model.to(DEVICE)
        metrics, details = evaluate_configuration(
            model, Xte_normal, Xte_anomaly, yte_anomaly,
            test_anomaly_metadata, best["window_size"],
            best["persistence_required"], best["persistence_span"],
            best["score_mode"],
            threshold=best["threshold"], calib=best["calib"])
        median_detection_delay(metrics, details, Xte_anomaly)
        metrics.update({
            "seed": int(seed),
            "best_epoch": best["epoch"],
            "validation_target_fpr": TARGET_VALIDATION_FPR,
            "val_precision": best["val_precision"],
            "val_recall": best["val_recall"],
            "val_f1": best["val_f1"],
            "val_fpr": best["val_fpr"],
            "val_pre_onset_alarm_rate": best[
                "val_pre_onset_alarm_rate"],
            "val_per_type_recall": best["val_per_type_recall"],
            "val_min_type_recall": best["val_min_type_recall"],
            "run_signature": signature,
            "signature_payload": payload,
            "source_step3_val_f1": source_metrics["val_f1"],
        })
        seed_results.append(metrics)
        json_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8")
        torch.save({
            "state_dict": best["state"],
            "history": source_artifact.get("hist"),
            "best": {key: value for key, value in best.items()
                     if key not in ("state", "rank", "alerts", "calib")},
            "calib": best["calib"],
            "metrics": metrics,
        }, model_path)
        print(
            f"seed {seed}: recall={metrics['recall']:.3f}, "
            f"FPR={metrics['fpr']:.3f}, F1={metrics['f1']:.3f}")

    release_metrics = max(
        seed_results,
        key=lambda item: (
            item["val_recall"], -item["val_fpr"], item["val_f1"]))
    release_seed = release_metrics["seed"]
    seed_artifact = torch.load(
        cache_dir / f"seed_{release_seed}.pt",
        map_location="cpu", weights_only=False)
    release_model = SlidingWindowLSTMAutoEncoder(
        n_features, HIDDEN_SIZE, LATENT_SIZE)
    release_model.load_state_dict(seed_artifact["state_dict"])
    release_model.to(DEVICE)
    release_eval, release_details = evaluate_configuration(
        release_model, Xte_normal, Xte_anomaly, yte_anomaly,
        test_anomaly_metadata, release_metrics["window_size"],
        release_metrics["persistence_required"],
        release_metrics["persistence_span"],
        release_metrics["score_mode"],
        threshold=release_metrics["threshold"],
        calib=seed_artifact["calib"])
    median_detection_delay(release_eval, release_details, Xte_anomaly)
    release_eval.update({
        "seed": release_seed,
        "best_epoch": release_metrics["best_epoch"],
        "validation_target_fpr": TARGET_VALIDATION_FPR,
        "val_precision": release_metrics["val_precision"],
        "val_recall": release_metrics["val_recall"],
        "val_f1": release_metrics["val_f1"],
        "val_fpr": release_metrics["val_fpr"],
        "val_per_type_recall": release_metrics["val_per_type_recall"],
        "val_min_type_recall": release_metrics["val_min_type_recall"],
    })

    checkpoint = {
        "state_dict": seed_artifact["state_dict"],
        "mean": mean,
        "std": std,
        "calib": seed_artifact["calib"],
        "threshold": release_eval["threshold"],
        "window_size": release_eval["window_size"],
        "persistence_required": release_eval["persistence_required"],
        "persistence_span": release_eval["persistence_span"],
        "score_mode": release_eval["score_mode"],
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "sensor_names": raw["sensor_names"],
        "seed": release_seed,
        "validation_target_fpr": TARGET_VALIDATION_FPR,
    }
    checkpoint_path, model_path = export_release(
        release_model, checkpoint, raw["sensor_names"])
    parity_sequences = [Xte_normal[0]]
    for anomaly_type in ANOMALY_LABELS:
        parity_sequences.append(Xte_anomaly[
            int(np.flatnonzero(yte_anomaly == anomaly_type)[0])])
    # Runtime accepts raw sensor values, not normalized model inputs.
    parity_raw_sequences = [
        sequence * std + mean for sequence in parity_sequences]
    artifact_parity = verify_artifact_parity(
        checkpoint_path, model_path, parity_raw_sequences)
    if not artifact_parity["passed"]:
        raise RuntimeError(f"Edge artifact parity failed: {artifact_parity}")
    edge_benchmark = benchmark_runtime(
        checkpoint_path, model_path, args.benchmark_runs)
    aggregate = aggregate_results(seed_results)
    report = {
        "protocol": {
            "model_training": (
                "Step 3 complete synthetic normal process trajectories only; "
                "online inference receives causal trailing windows"
            ),
            "model_selection": (
                "maximize overall anomaly recall after satisfying an "
                "empirical event-FPR constraint on separate validation data"
            ),
            "validation_target_fpr": TARGET_VALIDATION_FPR,
            "within_run_test_access": "once per seed after model selection",
            "score": (
                "validation-selected trailing-window reconstruction-error "
                "or first-difference reconstruction-error reduction, calibrated per "
                "sensor, max across sensors, followed by k-of-n persistence"
            ),
            "causality": "decision at t uses only samples at or before t",
            "true_positive_rule": (
                "complete persistence evidence span must be at or after "
                "synthetic anomaly onset"
            ),
        },
        "per_seed": seed_results,
        "aggregate": aggregate,
        "release_seed": release_seed,
        "release": release_eval,
        "artifact_parity": artifact_parity,
        "edge_benchmark": edge_benchmark,
        "limitations": [
            "Synthetic anomaly onset is not a measured physical failure onset.",
            "The fixed benchmark oversamples anomalies and does not represent fab prevalence.",
            "Step 3 checkpoint selection and online-configuration selection reuse the same validation cohort.",
            "The fixed development benchmark was inspected during V2 iteration; use the separately locked holdout for final claims.",
            "LAM 9600 sampling cannot validate sub-second process dynamics.",
            "Host CPU latency is not Raspberry Pi or production IPC latency.",
        ],
    }
    report_path = OUTPUT_DIR / "sliding_window_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_release(release_eval, release_details, Xte_anomaly)
    print("\n===== Sliding-window LSTM-AE summary =====")
    print(
        f"recall={aggregate['recall']['mean']:.3f} +/- "
        f"{aggregate['recall']['std']:.3f}, "
        f"FPR={aggregate['fpr']['mean']:.3f} +/- "
        f"{aggregate['fpr']['std']:.3f}, "
        f"F1={aggregate['f1']['mean']:.3f}")
    print(f"Release seed: {release_seed}")
    print(
        "Artifact parity: passed, max score difference="
        f"{artifact_parity['max_persistent_score_abs_difference']:.2e}")
    print(
        f"Host edge-update p95={edge_benchmark['p95_us']:.1f} us, "
        f"p99={edge_benchmark['p99_us']:.1f} us")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
