# -*- coding: utf-8 -*-
"""Step 7: causal streaming anomaly detection and edge-runtime benchmark.

This experiment closes a gap in the original LSTM-AE pipeline. The
autoencoder scores a complete wafer, while an early-warning claim requires a
detector whose decision at time t uses only observations available by time t.

The model is a normal-only LSTM one-step forecaster. A trailing-window
prediction-error score produces an online alarm. Synthetic anomaly labels are
used for checkpoint/threshold-rule selection, so the detector training is
unsupervised but the complete model-selection protocol is semi-supervised.

Outputs:
- outputs/streaming_lstm_forecaster.pt: release checkpoint and score settings
- outputs/streaming_lstm_step.ts: state-explicit TorchScript edge artifact
- outputs/streaming_early_warning.json: metrics and host CPU microbenchmark
- figures/07_streaming_early_warning.png: event recall and alarm timelines

The CPU benchmark describes this host only. It must not be reported as a
Raspberry Pi measurement.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from config import OUTPUT_DIR, FIGURE_DIR, COLORS, set_plot_style
import matplotlib.pyplot as plt
from models import (
    DEVICE,
    LSTMForecaster,
    LSTMForecasterStep,
    combine_peaks,
    forecaster_grid_select,
    forecaster_pointwise_errors,
    make_threshold,
    sensor_peak_scores,
    streaming_score_curves,
    train_forecaster_collect_checkpoints,
)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_EPOCHS = 200
CKPT_EVERY = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 1
ANOMALY_LABELS = {
    1: "A: 暫態到位過快",
    2: "B: 過程震盪",
    3: "C: 緩慢漂移",
}
PIPELINE_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated random seeds",
    )
    parser.add_argument("--benchmark-runs", type=int, default=2000)
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore matching per-seed caches and retrain",
    )
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(",")
                  if value.strip()]
    if not args.seeds:
        parser.error("--seeds must contain at least one integer")
    if args.epochs < CKPT_EVERY:
        parser.error(f"--epochs must be at least {CKPT_EVERY}")
    if args.benchmark_runs < 10:
        parser.error("--benchmark-runs must be at least 10")
    return args


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_signature(data_path: Path, epochs: int, seed: int):
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "data_sha256": file_sha256(data_path),
        "epochs": epochs,
        "checkpoint_every": CKPT_EVERY,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "seed": seed,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def load_data():
    path = OUTPUT_DIR / "synthetic_data.npz"
    data = np.load(path, allow_pickle=True)
    if "test_metadata" not in data.files:
        raise RuntimeError(
            "synthetic_data.npz has no anomaly-onset metadata. "
            "Run 02_generate_synthetic.py before this experiment."
        )
    val_metadata = [json.loads(str(item)) for item in data["val_metadata"]]
    metadata = [json.loads(str(item)) for item in data["test_metadata"]]
    sensor_names = [str(item) for item in data["sensor_names"]]
    return {
        "path": path,
        "train": list(data["X_train"]),
        "val": list(data["X_val"]),
        "val_anom": list(data["X_val_anom"]),
        "y_val_anom": data["y_val_anom"],
        "val_metadata": val_metadata,
        "test": list(data["X_test"]),
        "y_test": data["y_test"],
        "test_metadata": metadata,
        "sensor_names": sensor_names,
    }


def zscore_list(items, mean, std):
    return [(item - mean) / std for item in items]


def summarize_alert_timing(alert_indices, pre_onset_alarms, sequences,
                           y_test, metadata):
    by_type = {}
    for anomaly_type, label in ANOMALY_LABELS.items():
        indices = np.flatnonzero(y_test == anomaly_type)
        detected = [int(i) for i in indices if alert_indices[i] is not None]
        progress = [
            alert_indices[i] / max(len(sequences[i]) - 1, 1) for i in detected
        ]
        remaining = [1.0 - value for value in progress]
        delays = []
        for i in detected:
            onset = metadata[i].get("onset_index")
            if onset is not None:
                delays.append(alert_indices[i] - int(onset))
        by_type[label] = {
            "detected": len(detected),
            "total": int(len(indices)),
            "median_alert_progress": (
                float(np.median(progress)) if progress else None
            ),
            "median_remaining_fraction": (
                float(np.median(remaining)) if remaining else None
            ),
            "median_detection_delay_samples": (
                float(np.median(delays)) if delays else None
            ),
            "pre_onset_alarm_rate": float(
                np.mean([pre_onset_alarms[i] for i in indices])
            ),
        }
    return by_type


def online_decisions(curves, threshold, window, y_test, metadata):
    """Return valid post-onset detections and separately flag pre-onset alarms."""
    pred = np.zeros(len(curves), dtype=int)
    alert_indices = []
    pre_onset_alarms = []
    event_scores = []
    for index, curve in enumerate(curves):
        sample_indices = np.arange(len(curve)) + window
        over = curve > threshold
        onset = metadata[index].get("onset_index") if y_test[index] > 0 else None
        if onset is None:
            eligible = np.ones(len(curve), dtype=bool)
            pre_alarm = False
        else:
            onset = int(onset)
            eligible = sample_indices >= onset
            pre_alarm = bool(np.any(over & (sample_indices < onset)))
        valid_crossings = np.flatnonzero(over & eligible)
        if len(valid_crossings):
            first = int(valid_crossings[0])
            pred[index] = 1
            alert_indices.append(int(sample_indices[first]))
        else:
            alert_indices.append(None)
        pre_onset_alarms.append(pre_alarm)
        eligible_scores = curve[eligible]
        event_scores.append(float(eligible_scores.max()))
    return (pred, alert_indices, np.asarray(pre_onset_alarms, dtype=bool),
            np.asarray(event_scores))


def evaluate(model, Xva, Xte, y_test, metadata, window, use_calib,
             threshold_rule):
    val_errors = forecaster_pointwise_errors(model, Xva)
    val_peaks = sensor_peak_scores(val_errors, window)
    calib = val_peaks.mean(axis=0) if use_calib else None
    val_scores = combine_peaks(val_peaks, calib)
    threshold = make_threshold(val_scores, threshold_rule)

    test_errors = forecaster_pointwise_errors(model, Xte)
    curves = streaming_score_curves(test_errors, window, calib)
    pred, alert_indices, pre_onset_alarms, test_scores = online_decisions(
        curves, threshold, window, y_test, metadata)
    truth = (y_test > 0).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0)
    per_type_recall = {
        label: float(pred[y_test == anomaly_type].mean())
        for anomaly_type, label in ANOMALY_LABELS.items()
    }
    timing = summarize_alert_timing(
        alert_indices, pre_onset_alarms, Xte, y_test, metadata)
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(pred[y_test == 0].mean()),
        "per_type_recall": per_type_recall,
        "alert_timing": timing,
        "pre_onset_alarm_rate": float(
            pre_onset_alarms[y_test > 0].mean()
        ),
        "threshold": float(threshold),
    }
    details = {
        "calib": calib,
        "test_scores": test_scores,
        "curves": curves,
        "pred": pred,
        "alert_indices": alert_indices,
        "pre_onset_alarms": pre_onset_alarms,
    }
    return metrics, details


def aggregate_results(results):
    aggregate = {}
    for key in ("precision", "recall", "f1", "fpr",
                "pre_onset_alarm_rate"):
        values = np.asarray([item[key] for item in results], dtype=float)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    aggregate["per_type"] = {}
    for label in ANOMALY_LABELS.values():
        recalls = np.asarray(
            [item["per_type_recall"][label] for item in results], dtype=float)
        entry = {
            "recall_mean": float(recalls.mean()),
            "recall_std": float(recalls.std()),
        }
        for key in ("median_alert_progress", "median_remaining_fraction",
                    "median_detection_delay_samples", "pre_onset_alarm_rate"):
            values = [item["alert_timing"][label][key] for item in results]
            values = np.asarray([value for value in values if value is not None],
                                dtype=float)
            entry[f"{key}_mean"] = float(values.mean()) if len(values) else None
            entry[f"{key}_std"] = float(values.std()) if len(values) else None
        aggregate["per_type"][label] = entry
    return aggregate


def export_and_benchmark(model, n_features, runs):
    model = model.cpu().eval()
    wrapper = LSTMForecasterStep(model).eval()
    sample = torch.zeros(1, 1, n_features)
    hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE)
    cell = torch.zeros_like(hidden)
    traced = torch.jit.trace(wrapper, (sample, hidden, cell))
    artifact = OUTPUT_DIR / "streaming_lstm_step.ts"
    traced.save(str(artifact))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for _ in range(200):
                _, hidden, cell = traced(sample, hidden, cell)
            durations_us = []
            for _ in range(runs):
                start = time.perf_counter_ns()
                _, hidden, cell = traced(sample, hidden, cell)
                durations_us.append((time.perf_counter_ns() - start) / 1000.0)
    finally:
        torch.set_num_threads(previous_threads)

    durations = np.asarray(durations_us)
    return {
        "device_scope": "host_cpu_not_raspberry_pi",
        "includes": "one recurrent inference step only",
        "excludes": "sensor I/O, preprocessing, score smoothing, and alert transport",
        "runs": int(runs),
        "mean_us": float(durations.mean()),
        "p50_us": float(np.percentile(durations, 50)),
        "p95_us": float(np.percentile(durations, 95)),
        "torchscript_size_kib": float(artifact.stat().st_size / 1024),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "threads_during_benchmark": 1,
    }


def plot_release(metrics, details, Xte, y_test, metadata):
    set_plot_style()
    labels = list(ANOMALY_LABELS.values())
    short_labels = ["A: 到位過快", "B: 過程震盪", "C: 緩慢漂移"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 13),
                             gridspec_kw={"height_ratios": [0.9, 1, 1, 1]})

    recalls = [metrics["per_type_recall"][label] for label in labels]
    bars = axes[0].bar(short_labels, recalls,
                       color=[COLORS["series3"], COLORS["series4"],
                              COLORS["series5"]])
    for bar, value in zip(bars, recalls):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                     f"{value:.2f}", ha="center", color=COLORS["ink2"])
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Event recall")
    axes[0].set_title(
        f"Causal LSTM forecaster (F1={metrics['f1']:.3f}, "
        f"FPR={metrics['fpr']:.3f})"
    )

    threshold = metrics["threshold"]
    for axis, anomaly_type, label in zip(axes[1:], ANOMALY_LABELS, labels):
        candidates = np.flatnonzero((y_test == anomaly_type) &
                                    (details["pred"] == 1))
        if len(candidates):
            index = int(candidates[np.argmax([
                1.0 - details["alert_indices"][i] / max(len(Xte[i]) - 1, 1)
                for i in candidates
            ])])
        else:
            candidates = np.flatnonzero(y_test == anomaly_type)
            index = int(candidates[np.argmax(details["test_scores"][candidates])])

        curve = details["curves"][index]
        progress = np.linspace(
            metrics["window"] / max(len(Xte[index]) - 1, 1), 1.0, len(curve))
        axis.plot(progress, curve, color=COLORS["faulty"], linewidth=1.8,
                  label="Online anomaly score")
        axis.axhline(threshold, color=COLORS["ink"], linestyle="--",
                     linewidth=1.2, label="Alarm threshold")
        onset = metadata[index].get("onset_fraction")
        if onset is not None:
            axis.axvline(onset, color=COLORS["series3"], linestyle=":",
                         linewidth=1.4, label="Synthetic injection onset")
        alert = details["alert_indices"][index]
        if alert is not None:
            alert_progress = alert / max(len(Xte[index]) - 1, 1)
            curve_index = int(np.flatnonzero(curve > threshold)[0])
            axis.scatter([alert_progress], [curve[curve_index]], s=55,
                         color=COLORS["faulty"], zorder=5, label="First alarm")
        axis.set_xlim(0, 1)
        axis.set_ylabel(label)
        axis.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    axes[-1].set_xlabel("Normalized process progress (0=start, 1=end)")
    fig.suptitle("Streaming early-warning timelines (release seed)", fontsize=13)
    fig.tight_layout()
    output = FIGURE_DIR / "07_streaming_early_warning.png"
    fig.savefig(output, bbox_inches="tight")
    print(f"Figure saved: {output}")


def main():
    args = parse_args()
    set_plot_style()
    raw = load_data()
    train_all = np.concatenate(raw["train"])
    mean = train_all.mean(axis=0)
    std = train_all.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xtr = zscore_list(raw["train"], mean, std)
    Xva = zscore_list(raw["val"], mean, std)
    Xva_anom = zscore_list(raw["val_anom"], mean, std)
    Xte = zscore_list(raw["test"], mean, std)
    n_features = Xtr[0].shape[1]

    cache_dir = OUTPUT_DIR / "streaming_seeds"
    cache_dir.mkdir(exist_ok=True)
    seed_results = []
    print(f"Training device: {DEVICE}")
    for seed in args.seeds:
        signature, signature_payload = run_signature(
            raw["path"], args.epochs, seed)
        json_path = cache_dir / f"seed_{seed}.json"
        model_path = cache_dir / f"seed_{seed}.pt"
        cached = None
        if not args.force and json_path.exists() and model_path.exists():
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("run_signature") != signature:
                cached = None
        if cached is not None:
            seed_results.append(cached)
            print(f"seed {seed}: matching cache, F1={cached['f1']:.3f}")
            continue

        print(f"\n===== streaming seed {seed} =====")
        # Seed before construction so it controls initial weights as intended.
        torch.manual_seed(seed)
        model = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS)
        history, checkpoints = train_forecaster_collect_checkpoints(
            model, Xtr, Xva, epochs=args.epochs, ckpt_every=CKPT_EVERY,
            seed=seed)
        best = forecaster_grid_select(
            model, checkpoints, Xva, Xva_anom, raw["y_val_anom"],
            anom_onset_indices=[
                item["onset_index"] for item in raw["val_metadata"]
            ])
        model.load_state_dict(best["state"])
        model.to(DEVICE)
        metrics, details = evaluate(
            model, Xva, Xte, raw["y_test"], raw["test_metadata"],
            best["window"], best["use_calib"], best["thr_rule"])
        metrics.update({
            "seed": seed,
            "best_epoch": best["epoch"],
            "window": best["window"],
            "use_calib": best["use_calib"],
            "thr_rule": best["thr_rule"],
            "val_f1": best["f1"],
            "run_signature": signature,
            "signature_payload": signature_payload,
        })
        seed_results.append(metrics)
        json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        torch.save({
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()},
            "history": history,
            "calib": details["calib"],
            "threshold": metrics["threshold"],
            "metrics": metrics,
        }, model_path)
        print(f"seed {seed}: F1={metrics['f1']:.3f}, "
              f"recall={metrics['recall']:.3f}, FPR={metrics['fpr']:.3f}")

    release_metrics = max(seed_results, key=lambda item: item["val_f1"])
    release_seed = release_metrics["seed"]
    release_artifact = torch.load(
        cache_dir / f"seed_{release_seed}.pt", weights_only=False)
    release_model = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS)
    release_model.load_state_dict(release_artifact["state_dict"])
    release_model.to(DEVICE)
    # Recompute to guarantee that figures and the release artifact use the
    # current normalized data rather than cached plotting details.
    release_eval, release_details = evaluate(
        release_model, Xva, Xte, raw["y_test"], raw["test_metadata"],
        release_metrics["window"], release_metrics["use_calib"],
        release_metrics["thr_rule"])
    release_eval.update({
        "seed": release_seed,
        "window": release_metrics["window"],
        "thr_rule": release_metrics["thr_rule"],
        "use_calib": release_metrics["use_calib"],
        "best_epoch": release_metrics["best_epoch"],
        "val_f1": release_metrics["val_f1"],
    })

    torch.save({
        "state_dict": release_artifact["state_dict"],
        "mean": mean,
        "std": std,
        "calib": release_details["calib"],
        "threshold": release_eval["threshold"],
        "window": release_eval["window"],
        "thr_rule": release_eval["thr_rule"],
        "use_calib": release_eval["use_calib"],
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "sensor_names": raw["sensor_names"],
        "seed": release_seed,
    }, OUTPUT_DIR / "streaming_lstm_forecaster.pt")

    edge_benchmark = export_and_benchmark(
        release_model, n_features, args.benchmark_runs)
    aggregate = aggregate_results(seed_results)
    report = {
        "protocol": {
            "model_training": "normal synthetic sequences only",
            "model_selection": "labeled synthetic anomaly validation set",
            "test_access": "once per seed after model selection",
            "alarm_score": "causal trailing mean of one-step squared error",
            "true_positive_rule": (
                "first threshold crossing at or after synthetic injection onset"
            ),
            "early_warning_definition": (
                "first threshold crossing before sequence end; synthetic "
                "injection onset is available only for generated data"
            ),
        },
        "per_seed": seed_results,
        "aggregate": aggregate,
        "release_seed": release_seed,
        "release": release_eval,
        "edge_benchmark": edge_benchmark,
        "limitations": [
            "Synthetic anomaly timing is not a measured physical failure onset.",
            "LAM 9600 sampling does not validate millisecond thermal behavior.",
            "Host CPU latency is not Raspberry Pi latency.",
            "Sensor I/O and alert transport are excluded from the microbenchmark.",
        ],
    }
    report_path = OUTPUT_DIR / "streaming_early_warning.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    plot_release(release_eval, release_details, Xte, raw["y_test"],
                 raw["test_metadata"])

    print("\n===== Streaming summary =====")
    print(f"F1={aggregate['f1']['mean']:.3f} ± {aggregate['f1']['std']:.3f}, "
          f"recall={aggregate['recall']['mean']:.3f}, "
          f"FPR={aggregate['fpr']['mean']:.3f}")
    print(f"Release seed: {release_seed}")
    print(f"Host CPU step latency p50={edge_benchmark['p50_us']:.1f} us, "
          f"p95={edge_benchmark['p95_us']:.1f} us "
          "(not a Raspberry Pi result)")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
