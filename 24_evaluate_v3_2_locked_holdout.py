# -*- coding: utf-8 -*-
"""Run the locked evaluator for corrected and calibrated V3.2."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from config import OUTPUT_DIR


PROJECT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v3_2_locked_evaluator", PROJECT_DIR / "16_evaluate_v3_locked_holdout.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

evaluator.ARTIFACT_PATH = (
    OUTPUT_DIR / "v3" / "sliding_window_lstm_ae_v3_2.pt")
evaluator.STATS_PATH = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
evaluator.PROTOCOL_PATH = OUTPUT_DIR / "v3" / "data_protocol_v3_2.json"
evaluator.REPORT_PATH = OUTPUT_DIR / "v3" / "final_holdout_v3_2.json"
evaluator.EXPECTED_ARTIFACT_SHA256 = (
    "e3ab0ba9954114bf4b8db5842838be88160ef7b65a308c4c8ffe6c0603a56b5d")
evaluator.EXPECTED_STATS_SHA256 = (
    "79118aea76f3d3bf8eb41c2b5d62e20ff56ab5adc073e00cc419dbbfb77b2aa6")
evaluator.NORMAL_HOLDOUT_SEED = 494001
evaluator.ANOMALY_HOLDOUT_SEED = 495001
evaluator.REPORT_STATUS = (
    "v3_2_locked_holdout_after_metadata_correction_and_normal_calibration")
evaluator.REPORT_LINEAGE = {
    "predecessors": [
        "outputs/v3/final_holdout_v3.json",
        "outputs/v3/final_holdout_v3_1.json",
    ],
    "erratum": "outputs/v3/report_errata_v3_family.json",
    "changes": [
        "correct Type A intervention-support metadata",
        "record complete training argv/config and completed epochs",
        "correct per-sample feature terminology and pre-onset diagnostic",
        "calibrate threshold with new 10,000-normal cohort seed 396001",
    ],
}
evaluator.CALIBRATION_REPORT_PATH = (
    OUTPUT_DIR / "v3" / "deployment_calibration_v3_2.json")
evaluator.CALIBRATION_CODE_PATH = PROJECT_DIR / "23_calibrate_v3_2.py"
evaluator.EXTRA_PROVENANCE_PATHS = [Path(__file__)]


if __name__ == "__main__":
    evaluator.main()
