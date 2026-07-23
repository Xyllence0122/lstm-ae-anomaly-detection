# -*- coding: utf-8 -*-
"""Benchmark the hash-verified V4 runtime on the current host or Pi 5."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from deployment_manifest import file_sha256
from v3_data import generate_set, load_statistics
from v4_edge_runtime import (
    DEFAULT_MANIFEST,
    V4MultiscaleDetector,
    load_v4_manifest,
)


DEFAULT_OUTPUT = OUTPUT_DIR / "v4" / "host_benchmark_v4.json"
DEFAULT_STATS = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", type=int, default=100)
    parser.add_argument("--seed", type=int, default=496001)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.sequences < 1:
        parser.error("--sequences must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    return args


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(values.max()),
    }


def process_rss_bytes():
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(
                statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = kernel32.K32GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        success = get_memory(
            get_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if success else None
    return None


def cpu_temperature_celsius():
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.is_file():
        return None
    try:
        return float(path.read_text(encoding="ascii").strip()) / 1000.0
    except (OSError, ValueError):
        return None


def hardware_model():
    path = Path("/proc/device-tree/model")
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip("\x00\n ")
        except OSError:
            pass
    return platform.platform()


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    process_start_rss = process_rss_bytes()
    start_temperature = cpu_temperature_celsius()
    manifest, _, manifest_hash = load_v4_manifest(args.manifest)
    statistics_document = load_statistics(args.statistics)
    sequences = generate_set(
        np.random.default_rng(args.seed),
        statistics_document,
        args.sequences,
        anomaly=0,
    )
    after_data_rss = process_rss_bytes()
    expected_samples = int(sum(map(len, sequences)))

    load_start = time.perf_counter()
    detector = V4MultiscaleDetector.from_manifest(args.manifest)
    model_load_seconds = time.perf_counter() - load_start
    after_model_rss = process_rss_bytes()
    peak_rss = max(
        value for value in (
            process_start_rss, after_data_rss, after_model_rss)
        if value is not None
    ) if any(value is not None for value in (
            process_start_rss, after_data_rss, after_model_rss)) else None
    sensor_names = detector.raw_sensor_names
    nominal_interval = detector.timing_contract[
        "nominal_interval_seconds"]
    all_latencies = []
    inference_latencies = []
    alarm_count = 0
    samples = 0
    benchmark_start = time.perf_counter()
    for sequence_index, sequence in enumerate(sequences):
        detector.start_stream(
            f"benchmark-{sequence_index}", "synthetic-normal", "host")
        for sample_index, row in enumerate(sequence):
            sample = dict(zip(sensor_names, row))
            started = time.perf_counter_ns()
            result = detector.update(
                sample, sample_index * nominal_interval)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            all_latencies.append(elapsed_ms)
            if result["window_ready"]:
                inference_latencies.append(elapsed_ms)
            alarm_count += int(result["alarm"])
            samples += 1
        current_rss = process_rss_bytes()
        if current_rss is not None:
            peak_rss = (
                current_rss if peak_rss is None
                else max(peak_rss, current_rss)
            )
    wall_seconds = time.perf_counter() - benchmark_start
    after_replay_rss = process_rss_bytes()
    if after_replay_rss is not None:
        peak_rss = (
            after_replay_rss if peak_rss is None
            else max(peak_rss, after_replay_rss)
        )
    end_temperature = cpu_temperature_celsius()
    model_name = hardware_model()
    is_pi5 = "Raspberry Pi 5" in model_name

    report = {
        "status": (
            "raspberry_pi_5_measurement"
            if is_pi5 else "non_pi_host_development_measurement"
        ),
        "hardware": {
            "model": model_name,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "is_raspberry_pi_5": is_pi5,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": args.torch_threads,
        },
        "protocol": {
            "synthetic_normal_sequences": args.sequences,
            "seed": args.seed,
            "sample_count": samples,
            "nominal_interval_seconds": nominal_interval,
            "model_load_seconds": model_load_seconds,
            "wall_seconds": wall_seconds,
            "throughput_samples_per_second": samples / wall_seconds,
        },
        "latency": {
            "all_updates": percentile_summary(all_latencies),
            "updates_with_model_inference": percentile_summary(
                inference_latencies),
            "raw_all_update_ms": [
                round(value, 6) for value in all_latencies],
            "raw_inference_update_ms": [
                round(value, 6) for value in inference_latencies],
        },
        "resources": {
            "rss_process_start_bytes": process_start_rss,
            "rss_after_data_generation_bytes": after_data_rss,
            "rss_after_model_load_bytes": after_model_rss,
            "rss_peak_during_replay_bytes": peak_rss,
            "rss_after_replay_bytes": after_replay_rss,
            "rss_model_load_increment_bytes": (
                after_model_rss - after_data_rss
                if after_model_rss is not None and after_data_rss is not None
                else None
            ),
            "cpu_temperature_start_celsius": start_temperature,
            "cpu_temperature_end_celsius": end_temperature,
            "power_watts": None,
            "power_note": "requires an external meter or supported Pi sensor",
        },
        "runtime_observations": {
            "alarm_sample_count_on_synthetic_normal": alarm_count,
            "expected_replay_samples": expected_samples,
            "processed_replay_samples": samples,
            "replay_sample_count_match": samples == expected_samples,
            "equipment_input_dropped_samples": None,
            "equipment_input_exceptions": None,
            "equipment_io_note": (
                "Synthetic in-process replay has no equipment transport; "
                "live dropped samples and I/O exceptions are not measured."),
        },
        "provenance": {
            "manifest_path": str(Path(args.manifest).resolve()),
            "manifest_sha256": manifest_hash,
            "model_version": manifest["model_version"],
            "statistics_path": str(Path(args.statistics).resolve()),
            "statistics_sha256": file_sha256(args.statistics),
            "benchmark_code_sha256": file_sha256(Path(__file__)),
        },
        "interpretation": (
            "Only a report with hardware.is_raspberry_pi_5=true may be cited "
            "as Raspberry Pi 5 performance. This replay excludes equipment "
            "protocol I/O and external alarm delivery latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "hardware": report["hardware"],
        "protocol": report["protocol"],
        "latency_summary": {
            "all_updates": report["latency"]["all_updates"],
            "updates_with_model_inference": report["latency"][
                "updates_with_model_inference"],
        },
        "resources": report["resources"],
        "runtime_observations": report["runtime_observations"],
        "provenance": report["provenance"],
        "full_report_path": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
