# -*- coding: utf-8 -*-
"""Train and select the dynamic-sensitive V5 LSTM-AE on selection data."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import DEVICE, SlidingWindowLSTMAutoEncoder, _buckets_by_length
from v3_features import (
    fit_feature_spec,
    sample_training_windows,
    transform_sequences,
)
from v5_experiment import (
    ACCEPTANCE_TARGETS,
    TARGET_FPR,
    WINDOW_SIZES,
    acceptance_gate,
    enumerate_profiles,
    file_sha256,
    select_ensemble,
    stored_profile,
    weighted_reconstruction_loss,
)


PROJECT_DIR = Path(__file__).resolve().parent
V5_DIR = OUTPUT_DIR / "v5"
DATA_PATH = V5_DIR / "selection_data_v5.npz"
PROTOCOL_PATH = V5_DIR / "selection_protocol_v5.json"
HIDDEN_SIZE = 64
LATENT_SIZE = 16
VALIDATION_WINDOW_SEED = 710110


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="710201,710202,710203")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-size", type=int, default=4)
    parser.add_argument("--delta-feature-weight", type=float, default=2.0)
    parser.add_argument("--phase-feature-weight", type=float, default=0.25)
    parser.add_argument(
        "--temporal-difference-weight", type=float, default=0.5)
    parser.add_argument("--experiment", default="dynamic_grouped")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    args = parser.parse_args()
    args.seeds = [
        int(item.strip()) for item in args.seeds.split(",") if item.strip()
    ]
    if not args.seeds:
        parser.error("--seeds must not be empty")
    if args.epochs < 10 or args.epochs % 10:
        parser.error("--epochs must be a positive multiple of 10")
    if args.batch_size < 1 or args.samples_per_size < 2:
        parser.error("batch size must be positive and samples per size >= 2")
    for name in (
        "delta_feature_weight", "phase_feature_weight",
        "temporal_difference_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    return args


def load_data(path):
    raw = np.load(path, allow_pickle=True)
    return {
        "train": list(raw["X_train"]),
        "validation": list(raw["X_val"]),
        "anomaly": list(raw["X_val_anom"]),
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
            dtype=torch.float32, device=DEVICE)
        for length, indices in _buckets_by_length(windows).items()
    }


def train_model(model, train_windows, validation_windows, seed, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_buckets = make_buckets(train_windows)
    validation_buckets = make_buckets(validation_windows)
    history = {"train": [], "validation": []}
    checkpoints = []
    raw_sensor_count = (next(iter(train_buckets.values())).shape[-1] - 1) // 2

    def loss(batch):
        return weighted_reconstruction_loss(
            model, batch, raw_sensor_count,
            delta_feature_weight=args.delta_feature_weight,
            phase_feature_weight=args.phase_feature_weight,
            temporal_difference_weight=args.temporal_difference_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = []
        for tensor in train_buckets.values():
            order = rng.permutation(len(tensor))
            batches.extend([
                tensor[order[start:start + args.batch_size]]
                for start in range(0, len(order), args.batch_size)
            ])
        rng.shuffle(batches)
        losses = []
        for batch in batches:
            optimizer.zero_grad()
            value = loss(batch)
            value.backward()
            optimizer.step()
            losses.append(float(value.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                float(loss(tensor).item())
                for tensor in validation_buckets.values()
            ]))
        history["train"].append(float(np.mean(losses)))
        history["validation"].append(validation_loss)
        if epoch % 10 == 0:
            checkpoints.append((epoch, cpu_state(model)))
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train={history['train'][-1]:.6f} "
                f"validation={validation_loss:.6f}")
    return history, checkpoints


def candidate_rank(candidate):
    gate = candidate["gate"]
    metrics = candidate["metrics"]
    return (
        gate["passed"], sum(gate["checks"].values()),
        candidate["macro_recall"], candidate["min_type_recall"],
        metrics["recall"], -metrics["fpr"], metrics["f1"],
    )


def select_checkpoint(model, checkpoints, data, feature_spec):
    candidates = []
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        profiles, labels = enumerate_profiles(
            model, data["validation"], data["anomaly"], data["labels"],
            data["metadata"], feature_spec, TARGET_FPR)
        ensemble = select_ensemble(
            profiles, labels, len(data["validation"]), TARGET_FPR)
        gate = acceptance_gate(ensemble["metrics"], ensemble["per_type"])
        candidate = {
            "epoch": epoch,
            "metrics": ensemble["metrics"],
            "per_type": ensemble["per_type"],
            "macro_recall": ensemble["macro_recall"],
            "min_type_recall": ensemble["min_type_recall"],
            "projected_precision": ensemble["projected_precision"],
            "threshold": ensemble["threshold"],
            "profiles": ensemble["profiles"],
            "gate": gate,
            "state_dict": state,
        }
        candidates.append(candidate)
        print(
            f"  selection epoch={epoch} recall={ensemble['metrics']['recall']:.3f} "
            f"F1={ensemble['metrics']['f1']:.3f} "
            f"FPR={ensemble['metrics']['fpr']:.3%} "
            f"types={ensemble['per_type']} gate={gate['passed']}")
    return max(candidates, key=candidate_rank), candidates


def report_candidate(candidate):
    return {
        key: value for key, value in candidate.items()
        if key not in ("profiles", "state_dict")
    } | {
        "profiles": [stored_profile(profile) for profile in candidate["profiles"]]
    }


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    actual_data_hash = file_sha256(data_path)
    if actual_data_hash != protocol["data_sha256"]:
        raise RuntimeError("V5 selection data does not match its protocol")
    experiment_dir = V5_DIR / f"experiment_{args.experiment}"
    if experiment_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite V5 experiment: {experiment_dir}")
    experiment_dir.mkdir(parents=True)
    data = load_data(data_path)
    feature_spec = fit_feature_spec(data["train"], data["sensor_names"])
    transformed = {
        "train": transform_sequences(data["train"], feature_spec),
        "validation": transform_sequences(data["validation"], feature_spec),
        "anomaly": transform_sequences(data["anomaly"], feature_spec),
        "labels": data["labels"],
        "metadata": data["metadata"],
    }
    validation_windows = sample_training_windows(
        transformed["validation"][:500], WINDOW_SIZES, 2,
        seed=VALIDATION_WINDOW_SEED)
    resolved_arguments = {
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "samples_per_size": args.samples_per_size,
        "delta_feature_weight": args.delta_feature_weight,
        "phase_feature_weight": args.phase_feature_weight,
        "temporal_difference_weight": args.temporal_difference_weight,
        "experiment": args.experiment,
        "data_path": str(data_path),
    }
    started = time.time()
    seed_results = []
    for seed in args.seeds:
        train_windows = sample_training_windows(
            transformed["train"], WINDOW_SIZES,
            args.samples_per_size, seed=seed)
        torch.manual_seed(seed)
        model = SlidingWindowLSTMAutoEncoder(
            len(feature_spec["feature_names"]), HIDDEN_SIZE, LATENT_SIZE)
        history, checkpoints = train_model(
            model, train_windows, validation_windows, seed, args)
        selected, all_candidates = select_checkpoint(
            model, checkpoints, transformed, feature_spec)
        seed_results.append({
            "seed": seed,
            "history": history,
            "selected": selected,
            "checkpoints": all_candidates,
        })
    release = max(seed_results, key=lambda item: candidate_rank(item["selected"]))
    selected = release["selected"]
    artifact_path = experiment_dir / "candidate.pt"
    torch.save({
        "version": 5,
        "model_family": "Sliding-Window LSTM Autoencoder",
        "state_dict": selected["state_dict"],
        "feature_spec": feature_spec,
        "profiles": [stored_profile(item) for item in selected["profiles"]],
        "threshold": selected["threshold"],
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "training_seed": release["seed"],
        "selected_epoch": selected["epoch"],
        "loss_config": {
            "delta_feature_weight": args.delta_feature_weight,
            "phase_feature_weight": args.phase_feature_weight,
            "temporal_difference_weight": args.temporal_difference_weight,
        },
        "selection_data_sha256": actual_data_hash,
        "selection_protocol_sha256": file_sha256(PROTOCOL_PATH),
        "selection_metrics": selected["metrics"],
        "selection_per_type": selected["per_type"],
        "selection_gate": selected["gate"],
    }, artifact_path)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    report = {
        "status": "v5_selection_complete_locked_holdout_unopened",
        "protocol": {
            "selection_only": True,
            "target_fpr": TARGET_FPR,
            "acceptance_targets": ACCEPTANCE_TARGETS,
            "reserved_seeds": protocol["reserved_unopened_seeds"],
            "locked_holdout_access": "none",
        },
        "resolved_arguments": resolved_arguments,
        "fixed_configuration": {
            "window_sizes": list(WINDOW_SIZES),
            "feature_groups": ["raw", "delta", "raw_delta"],
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
            "validation_window_seed": VALIDATION_WINDOW_SEED,
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
        "feature_spec": feature_spec,
        "seed_results": [{
            "seed": item["seed"],
            "completed_epochs": len(item["history"]["train"]),
            "final_train_loss": item["history"]["train"][-1],
            "final_validation_loss": item["history"]["validation"][-1],
            "selected": report_candidate(item["selected"]),
            "checkpoint_diagnostics": [
                report_candidate(candidate) for candidate in item["checkpoints"]
            ],
        } for item in seed_results],
        "release_seed": release["seed"],
        "release": report_candidate(selected),
        "candidate_sha256": file_sha256(artifact_path),
        "selection_data_sha256": actual_data_hash,
        "selection_protocol_sha256": file_sha256(PROTOCOL_PATH),
        "resolved_command": command,
        "command_line_windows": subprocess.list2cmdline(command),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(DEVICE),
        },
        "code_sha256": {
            "trainer": file_sha256(Path(__file__)),
            "experiment_helpers": file_sha256(
                PROJECT_DIR / "v5_experiment.py"),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = experiment_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "release_seed": release["seed"],
        "release": report_candidate(selected),
        "candidate_sha256": report["candidate_sha256"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
