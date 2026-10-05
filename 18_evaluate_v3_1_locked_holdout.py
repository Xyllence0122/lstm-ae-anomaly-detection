# -*- coding: utf-8 -*-
"""Run the shared locked evaluator for the normal-only-calibrated V3.1."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from config import OUTPUT_DIR


PROJECT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v3_locked_evaluator", PROJECT_DIR / "16_evaluate_v3_locked_holdout.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

evaluator.ARTIFACT_PATH = (
    OUTPUT_DIR / "v3" / "sliding_window_lstm_ae_v3_1.pt")
evaluator.REPORT_PATH = OUTPUT_DIR / "v3" / "final_holdout_v3_1.json"
evaluator.EXPECTED_ARTIFACT_SHA256 = (
    "5225deea1e1af34fc5fcab987af03d0aefaebae0c42433b71be4868dac6468b2")
evaluator.NORMAL_HOLDOUT_SEED = 394001
evaluator.ANOMALY_HOLDOUT_SEED = 395001
evaluator.EXTRA_PROVENANCE_PATHS = [Path(__file__)]
evaluator.REPORT_STATUS = "v3_1_locked_holdout_after_normal_only_calibration"
evaluator.REPORT_LINEAGE = {
    "predecessor": "outputs/v3/final_holdout_v3.json",
    "reason": "larger normal-only threshold calibration",
}
evaluator.CALIBRATION_REPORT_PATH = (
    OUTPUT_DIR / "v3" / "deployment_calibration_v3_1.json")
evaluator.CALIBRATION_CODE_PATH = PROJECT_DIR / "17_calibrate_v3_1.py"


if __name__ == "__main__":
    evaluator.main()
