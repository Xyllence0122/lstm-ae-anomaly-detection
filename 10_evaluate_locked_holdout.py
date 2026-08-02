# -*- coding: utf-8 -*-
"""Evaluate frozen online detectors once on an independent final holdout.

This script uses random seeds that are distinct from training, validation,
development test, and Step 9 low-FPR cohorts. Do not use its results to change
model hyperparameters; any later model change requires a new holdout seed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lstm-ae-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

from config import COLORS, FIGURE_DIR, OUTPUT_DIR, set_plot_style
from deployment_manifest import (
    file_sha256,
    load_deployment_manifest,
    verify_report_digest,
)


CALIBRATION_NORMALS = 5000
HOLDOUT_NORMALS = 10000
HOLDOUT_PER_ANOMALY = 1000
CALIBRATION_SEED = 193001
NORMAL_HOLDOUT_SEED = 194001
ANOMALY_HOLDOUT_SEED = 195001


def load_numbered_module(filename, module_name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_holdout():
    generator = load_numbered_module(
        "02_generate_synthetic.py", "locked_holdout_generator")
    stats, sensors = generator.load_stats()
    length_range = (stats["len_min"], stats["len_max"])
    calibration = generator.gen_set(
        np.random.default_rng(CALIBRATION_SEED), sensors, length_range,
        CALIBRATION_NORMALS, anomaly=0)
    normal = generator.gen_set(
        np.random.default_rng(NORMAL_HOLDOUT_SEED), sensors, length_range,
        HOLDOUT_NORMALS, anomaly=0)
    anomaly_rng = np.random.default_rng(ANOMALY_HOLDOUT_SEED)
    anomalies, labels, metadata = [], [], []
    for anomaly_type in (1, 2, 3):
        wafers, items = generator.gen_set(
            anomaly_rng, sensors, length_range, HOLDOUT_PER_ANOMALY,
            anomaly=anomaly_type, with_metadata=True)
        anomalies.extend(wafers)
        labels.extend([anomaly_type] * len(wafers))
        metadata.extend(items)
    return calibration, normal, anomalies, np.asarray(labels), metadata


def plot_results(results):
    set_plot_style()
    target = 0.001
    subset = [item for item in results if item["target_fpr"] == target]
    names = [item["model"].replace("Causal ", "") for item in subset]
    x = np.arange(len(subset))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    recalls = [item["recall"] for item in subset]
    fprs = [item["fpr"] for item in subset]
    recall_bars = axes[0].bar(x, recalls, width=0.58,
                              color=COLORS["faulty"])
    fpr_bars = axes[1].bar(x, fprs, width=0.58,
                           color=COLORS["normal"])
    for axis in axes:
        axis.set_xticks(x, names)
    axes[0].set_ylim(0, 0.65)
    axes[0].set_ylabel("Anomaly recall")
    axes[0].set_title("Recall")
    axes[0].bar_label(recall_bars, fmt="%.3f", padding=3)
    axes[1].set_ylim(0, max(fprs) * 1.35)
    axes[1].set_ylabel("Normal-wafer FPR")
    axes[1].set_title("Observed FPR")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{100 * value:.2f}%"))
    axes[1].bar_label(
        fpr_bars, labels=[f"{100 * value:.2f}%" for value in fprs],
        padding=3)
    fig.suptitle("Locked holdout at 0.1% calibration target FPR")
    fig.tight_layout()
    output = FIGURE_DIR / "11_locked_holdout.png"
    fig.savefig(output, bbox_inches="tight")
    print(f"Figure saved: {output}")


def main():
    output = OUTPUT_DIR / "final_holdout_evaluation.json"
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        manifest = load_deployment_manifest(
            OUTPUT_DIR / "edge_deployment_manifest.json",
            verify_artifacts=True, verify_provenance=True)
        final_report_entry = next(
            item for item in manifest["payload"]["reports"]
            if item["path"] == "outputs/final_holdout_evaluation.json")
        verify_report_digest(
            output, final_report_entry["json_pointer"],
            final_report_entry["sha256"])
        print(
            "Existing locked-holdout report and complete provenance verified; "
            "preserving results and regenerating only the figure.")
        plot_results(report["results"])
        return

    comparison = load_numbered_module(
        "09_compare_online_models.py", "locked_holdout_comparison")
    sliding_path = OUTPUT_DIR / "sliding_window_lstm_ae.pt"
    forecaster_path = OUTPUT_DIR / "streaming_lstm_forecaster.pt"
    calibration, normal, anomalies, labels, metadata = generate_holdout()
    print(
        f"Locked holdout: calibration normal={len(calibration)}, "
        f"test normal={len(normal)}, anomalies={len(anomalies)}")

    sliding_checkpoint = torch.load(
        sliding_path, map_location="cpu", weights_only=False)
    forecaster_checkpoint = torch.load(
        forecaster_path, map_location="cpu", weights_only=False)
    results = comparison.evaluate_sliding(
        sliding_checkpoint, calibration, normal,
        anomalies, labels, metadata)
    results.extend(comparison.evaluate_forecaster(
        forecaster_checkpoint, calibration, normal,
        anomalies, labels, metadata))

    report = {
        "protocol": {
            "status": "final holdout; not used for model selection",
            "calibration_normal_count": CALIBRATION_NORMALS,
            "holdout_normal_count": HOLDOUT_NORMALS,
            "holdout_per_anomaly_type": HOLDOUT_PER_ANOMALY,
            "random_seeds": {
                "calibration": CALIBRATION_SEED,
                "normal_holdout": NORMAL_HOLDOUT_SEED,
                "anomaly_holdout": ANOMALY_HOLDOUT_SEED,
            },
            "artifact_sha256": {
                "sliding_window_lstm_ae.pt": file_sha256(sliding_path),
                "streaming_lstm_forecaster.pt": file_sha256(forecaster_path),
            },
            "warning": (
                "Changing model settings after reading this report invalidates "
                "its final-holdout status."
            ),
        },
        "results": results,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(results)
    print("\n===== Frozen-model final holdout =====")
    for item in results:
        print(
            f"{item['model']:26s} target={item['target_fpr']:.3%} "
            f"FPR={item['fpr']:.3%} "
            f"CI=[{item['fpr_wilson_95'][0]:.3%}, "
            f"{item['fpr_wilson_95'][1]:.3%}] "
            f"recall={item['recall']:.3f} "
            f"CI=[{item['recall_wilson_95'][0]:.3f}, "
            f"{item['recall_wilson_95'][1]:.3f}]")
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()
