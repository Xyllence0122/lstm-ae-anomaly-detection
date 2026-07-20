# -*- coding: utf-8 -*-
"""Step 9: prevalence-aware low-FPR comparison of online detectors.

Large independent synthetic-normal cohorts are used to calibrate and test rare
false-alarm targets. The existing 300-anomaly fixed test cohort is retained so
per-type recall remains measurable. Reported deployment precision values are
mathematical projections under explicit prevalence assumptions, not fab data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from config import COLORS, FIGURE_DIR, OUTPUT_DIR, set_plot_style
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from models import (
    DEVICE,
    LSTMForecaster,
    SlidingWindowLSTMAutoEncoder,
    forecaster_pointwise_errors,
    sensor_peak_scores,
    sliding_window_errors,
    streaming_score_curves,
)
from online_evaluation import (
    apply_persistence,
    binary_event_metrics,
    calibrate_sensor_errors,
    event_decisions,
    projected_precision,
    sensor_error_score_curves,
    threshold_for_target_fpr,
    wilson_interval,
)


DEFAULT_CALIBRATION_NORMALS = 5000
DEFAULT_TEST_NORMALS = 10000
TARGET_FPRS = (0.01, 0.005, 0.001)
PREVALENCE_SCENARIOS = (0.05, 0.01, 0.001)
ANOMALY_LABELS = {
    1: "A: 暫態響應速度異常",
    2: "B: 過程震盪",
    3: "C: 緩慢漂移",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-normals", type=int,
        default=DEFAULT_CALIBRATION_NORMALS)
    parser.add_argument(
        "--test-normals", type=int, default=DEFAULT_TEST_NORMALS)
    args = parser.parse_args()
    if args.calibration_normals < 100:
        parser.error("--calibration-normals must be at least 100")
    if args.test_normals < 100:
        parser.error("--test-normals must be at least 100")
    return args


def load_generator_module():
    path = Path(__file__).with_name("02_generate_synthetic.py")
    spec = importlib.util.spec_from_file_location("synthetic_generator_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixed_anomaly_test():
    raw = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    labels = np.asarray(raw["y_test"])
    mask = labels > 0
    sequences = [sequence for sequence, keep in zip(raw["X_test"], mask)
                 if keep]
    metadata = [json.loads(str(item))
                for item, keep in zip(raw["test_metadata"], mask) if keep]
    return sequences, labels[mask], metadata


def generate_normal_cohorts(calibration_count, test_count):
    generator = load_generator_module()
    stats, sensors = generator.load_stats()
    length_range = (stats["len_min"], stats["len_max"])
    calibration = generator.gen_set(
        np.random.default_rng(91001), sensors, length_range,
        calibration_count, anomaly=0)
    test = generator.gen_set(
        np.random.default_rng(92001), sensors, length_range,
        test_count, anomaly=0)
    return calibration, test


def normalize(items, mean, std):
    return [(np.asarray(item, dtype=np.float32) - mean) / std
            for item in items]


def summarize_at_threshold(model_name, target_fpr, threshold,
                           normal_curves, anomaly_curves, anomaly_labels,
                           anomaly_metadata, first_sample, evidence_span):
    labels = np.concatenate([
        np.zeros(len(normal_curves), dtype=int), anomaly_labels])
    curves = normal_curves + anomaly_curves
    metadata = ([{} for _ in normal_curves] + anomaly_metadata)
    predictions, alerts, pre_onset, scores = event_decisions(
        curves, threshold, first_sample, labels, metadata,
        evidence_span=evidence_span)
    metrics = binary_event_metrics(labels, predictions, scores)
    metrics.update({
        "model": model_name,
        "target_fpr": float(target_fpr),
        "threshold": float(threshold),
        "normal_test_count": int(len(normal_curves)),
        "anomaly_test_count": int(len(anomaly_curves)),
        "false_alarms": int(predictions[labels == 0].sum()),
        "detected_anomalies": int(predictions[labels > 0].sum()),
        "fpr_wilson_95": wilson_interval(
            int(predictions[labels == 0].sum()), int(np.count_nonzero(labels == 0))),
        "recall_wilson_95": wilson_interval(
            int(predictions[labels > 0].sum()), int(np.count_nonzero(labels > 0))),
        "pre_onset_alarm_rate": float(pre_onset[labels > 0].mean()),
        "per_type_recall": {
            name: float(predictions[labels == anomaly_type].mean())
            for anomaly_type, name in ANOMALY_LABELS.items()
        },
        "projected_precision": {
            f"prevalence_{prevalence:g}": projected_precision(
                metrics["recall"], metrics["fpr"], prevalence)
            for prevalence in PREVALENCE_SCENARIOS
        },
    })
    progress = {}
    normal_count = len(normal_curves)
    for anomaly_type, name in ANOMALY_LABELS.items():
        detected_progress = []
        for combined_index in np.flatnonzero(labels == anomaly_type):
            alert = alerts[int(combined_index)]
            if alert is None:
                continue
            anomaly_index = int(combined_index - normal_count)
            length = int(anomaly_metadata[anomaly_index]["sequence_length"])
            detected_progress.append(alert / max(length - 1, 1))
        progress[name] = (
            float(np.median(detected_progress))
            if detected_progress else None)
    metrics["median_alert_progress_detected"] = progress
    return metrics


def evaluate_sliding(checkpoint, calibration_normal, test_normal,
                     anomalies, anomaly_labels, anomaly_metadata):
    model = SlidingWindowLSTMAutoEncoder(
        len(checkpoint["mean"]), checkpoint["hidden_size"],
        checkpoint["latent_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    mean, std = checkpoint["mean"], checkpoint["std"]
    calibration = normalize(calibration_normal, mean, std)
    normal = normalize(test_normal, mean, std)
    anomaly = normalize(anomalies, mean, std)
    window_size = int(checkpoint["window_size"])
    required = int(checkpoint["persistence_required"])
    span = int(checkpoint["persistence_span"])
    score_mode = checkpoint.get("score_mode", "last")

    calibration_errors = sliding_window_errors(
        model, calibration, window_size, score_mode)
    calib = calibrate_sensor_errors(calibration_errors)
    calibration_curves = apply_persistence(
        sensor_error_score_curves(calibration_errors, calib), required, span)
    normal_curves = apply_persistence(
        sensor_error_score_curves(sliding_window_errors(
            model, normal, window_size, score_mode), calib), required, span)
    anomaly_curves = apply_persistence(
        sensor_error_score_curves(sliding_window_errors(
            model, anomaly, window_size, score_mode), calib), required, span)
    calibration_scores = np.asarray(
        [curve.max() for curve in calibration_curves])
    first_sample = window_size - 1 + span - 1
    results = []
    for target_fpr in TARGET_FPRS:
        threshold = threshold_for_target_fpr(
            calibration_scores, target_fpr)
        results.append(summarize_at_threshold(
            "Sliding-Window LSTM-AE", target_fpr, threshold,
            normal_curves, anomaly_curves, anomaly_labels, anomaly_metadata,
            first_sample, span))
    return results


def evaluate_forecaster(checkpoint, calibration_normal, test_normal,
                        anomalies, anomaly_labels, anomaly_metadata):
    model = LSTMForecaster(
        len(checkpoint["mean"]), checkpoint["hidden_size"],
        checkpoint["num_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    mean, std = checkpoint["mean"], checkpoint["std"]
    calibration = normalize(calibration_normal, mean, std)
    normal = normalize(test_normal, mean, std)
    anomaly = normalize(anomalies, mean, std)
    score_window = int(checkpoint["window"])

    calibration_errors = forecaster_pointwise_errors(model, calibration)
    calibration_peaks = sensor_peak_scores(
        calibration_errors, score_window)
    calib = calibration_peaks.mean(axis=0) if checkpoint["use_calib"] else None
    calibration_curves = streaming_score_curves(
        calibration_errors, score_window, calib)
    normal_curves = streaming_score_curves(
        forecaster_pointwise_errors(model, normal), score_window, calib)
    anomaly_curves = streaming_score_curves(
        forecaster_pointwise_errors(model, anomaly), score_window, calib)
    calibration_scores = np.asarray(
        [curve.max() for curve in calibration_curves])
    results = []
    for target_fpr in TARGET_FPRS:
        threshold = threshold_for_target_fpr(
            calibration_scores, target_fpr)
        results.append(summarize_at_threshold(
            "Causal LSTM Forecaster", target_fpr, threshold,
            normal_curves, anomaly_curves, anomaly_labels, anomaly_metadata,
            score_window, 1))
    return results


def plot_results(results):
    set_plot_style()
    models = list(dict.fromkeys(item["model"] for item in results))
    colors = [COLORS["faulty"], COLORS["normal"]]
    x = np.arange(len(TARGET_FPRS))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))

    for index, (model_name, color) in enumerate(zip(models, colors)):
        subset = [item for item in results if item["model"] == model_name]
        offset = (index - (len(models) - 1) / 2) * width
        axes[0].bar(
            x + offset, [item["recall"] for item in subset], width,
            color=color, label=model_name)
        axes[1].bar(
            x + offset, [item["fpr"] for item in subset], width,
            color=color, label=model_name)
        axes[2].bar(
            x + offset,
            [item["projected_precision"]["prevalence_0.01"]
             for item in subset],
            width, color=color, label=model_name)

    labels = [f"{100 * value:g}%" for value in TARGET_FPRS]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Calibration target FPR")
    axes[0].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    max_observed_fpr = max(item["fpr"] for item in results)
    axes[1].set_ylim(0, max_observed_fpr * 1.25)
    axes[0].set_ylabel("Anomaly recall")
    axes[0].set_title("Recall under low-FPR calibration")
    axes[1].set_ylabel("Observed normal-wafer FPR")
    axes[1].set_title("Independent normal test cohort")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{100 * value:.1f}%"))
    axes[2].set_ylabel("Projected alert precision")
    axes[2].set_title("Projection at 1% anomaly prevalence")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Prevalence-aware online anomaly-detector comparison")
    fig.tight_layout()
    output = FIGURE_DIR / "09_low_fpr_comparison.png"
    fig.savefig(output, bbox_inches="tight")
    print(f"Figure saved: {output}")


def main():
    args = parse_args()
    sliding_path = OUTPUT_DIR / "sliding_window_lstm_ae.pt"
    forecaster_path = OUTPUT_DIR / "streaming_lstm_forecaster.pt"
    if not sliding_path.exists():
        raise FileNotFoundError("Run 08_train_sliding_window_lstm_ae.py first")
    anomalies, labels, metadata = load_fixed_anomaly_test()
    calibration_normal, test_normal = generate_normal_cohorts(
        args.calibration_normals, args.test_normals)
    print(
        f"Rare-event cohorts: calibration normal={len(calibration_normal)}, "
        f"test normal={len(test_normal)}, anomaly={len(anomalies)}")

    sliding_checkpoint = torch.load(
        sliding_path, map_location="cpu", weights_only=False)
    forecaster_checkpoint = torch.load(
        forecaster_path, map_location="cpu", weights_only=False)
    results = evaluate_sliding(
        sliding_checkpoint, calibration_normal, test_normal,
        anomalies, labels, metadata)
    results.extend(evaluate_forecaster(
        forecaster_checkpoint, calibration_normal, test_normal,
        anomalies, labels, metadata))

    comparison_path = OUTPUT_DIR / "comparison_metrics.json"
    batch_baselines = json.loads(
        comparison_path.read_text(encoding="utf-8"))
    report = {
        "protocol": {
            "calibration_normal_count": int(args.calibration_normals),
            "independent_test_normal_count": int(args.test_normals),
            "fixed_anomaly_test_count": int(len(anomalies)),
            "target_fprs": TARGET_FPRS,
            "prevalence_scenarios": PREVALENCE_SCENARIOS,
            "threshold_data": "independent synthetic normal calibration cohort",
            "test_data": (
                "independent synthetic normals plus the fixed A/B/C test "
                "anomalies; anomalies are intentionally retained for stable recall"
            ),
        },
        "online_results": results,
        "existing_complete_wafer_baselines": batch_baselines,
        "limitations": [
            "Prevalence values are explicit scenarios, not measured fab rates.",
            "Projected precision assumes recall and FPR transfer to deployment.",
            "Large normal cohorts are synthetic and cannot replace long real production runs.",
            "Complete-wafer baselines are not directly comparable on alert timing.",
        ],
    }
    output = OUTPUT_DIR / "prevalence_evaluation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(results)
    print("\n===== Low-FPR comparison =====")
    for item in results:
        print(
            f"{item['model']:26s} target={item['target_fpr']:.3%} "
            f"observed FPR={item['fpr']:.3%} recall={item['recall']:.3f} "
            f"projected precision@1%="
            f"{item['projected_precision']['prevalence_0.01']:.3f}")
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()
