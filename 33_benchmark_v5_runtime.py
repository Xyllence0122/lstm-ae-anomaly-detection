# -*- coding: utf-8 -*-
"""Run the hash-verified V5 runtime through the shared Pi benchmark core."""
from __future__ import annotations

import importlib.util

from config import OUTPUT_DIR, PROJECT_DIR
from v5_edge_runtime import (
    DEFAULT_MANIFEST,
    V5MultiscaleDetector,
    load_v5_manifest,
)


spec = importlib.util.spec_from_file_location(
    "v5_benchmark_core", PROJECT_DIR / "27_benchmark_v4_runtime.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
benchmark.DEFAULT_MANIFEST = DEFAULT_MANIFEST
benchmark.DEFAULT_OUTPUT = OUTPUT_DIR / "v5" / "host_benchmark_v5.json"
benchmark.V4MultiscaleDetector = V5MultiscaleDetector
benchmark.load_v4_manifest = load_v5_manifest


if __name__ == "__main__":
    benchmark.main()
