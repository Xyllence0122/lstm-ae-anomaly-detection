# -*- coding: utf-8 -*-
"""Run the historical real-data reuse audit for frozen V3.2."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from config import OUTPUT_DIR


PROJECT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v3_2_real_reuse",
    PROJECT_DIR / "19_evaluate_v3_1_real_external.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

audit.ARTIFACT_PATH = OUTPUT_DIR / "v3" / "sliding_window_lstm_ae_v3_2.pt"
audit.REPORT_PATH = (
    OUTPUT_DIR / "v3" / "historical_real_data_reuse_audit_v3_2.json")
audit.EXPECTED_ARTIFACT_SHA256 = (
    "e3ab0ba9954114bf4b8db5842838be88160ef7b65a308c4c8ffe6c0603a56b5d")
audit.EXTRA_PROVENANCE_PATHS = [Path(__file__)]


if __name__ == "__main__":
    audit.main()
