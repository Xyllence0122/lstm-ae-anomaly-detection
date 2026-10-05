import json
import unittest
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from models import SlidingWindowLSTMAutoEncoder
from v3_features import transform_sequence
from v5_edge_runtime import V5MultiscaleDetector
from v5_experiment import profile_score_outputs


ARTIFACT_PATH = OUTPUT_DIR / "v5" / "sliding_window_lstm_ae_v5.pt"
SELECTION_DATA_PATH = OUTPUT_DIR / "v5" / "selection_data_v5.npz"
STATS_PATH = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"


class ZeroAutoEncoder(torch.nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


def small_feature_spec():
    return {
        "raw_sensor_names": ["Pressure", "Valve"],
        "feature_names": [
            "Pressure", "Valve", "delta::Pressure", "delta::Valve",
            "elapsed_phase"],
        "expected_cycle_samples": 5.0,
        "feature_mean": [0.0] * 5,
        "feature_std": [1.0] * 5,
        "score_feature_indices": [0, 1, 2, 3],
    }


def timing_contract():
    return {
        "timestamp_required": True,
        "minimum_interval_seconds": 0.5,
        "maximum_interval_seconds": 1.5,
        "nominal_interval_seconds": 1.0,
        "sensor_timeout_seconds": 3.0,
    }


class V5RuntimeTests(unittest.TestCase):
    def test_profile_feature_group_controls_evidence(self):
        profiles = [{
            "profile_id": "delta_only",
            "window_size": 2,
            "score_mode": "mean",
            "feature_group": "delta",
            "feature_indices": [2, 3],
            "persistence_required": 1,
            "persistence_span": 1,
            "base_scale": 1.0,
            "calibration": [1.0, 1.0],
        }]
        detector = V5MultiscaleDetector(
            ZeroAutoEncoder(), small_feature_spec(), profiles, 0.5,
            timing_contract(), "test-v5", "model-hash")
        detector.start_stream("W1", "R1", "EQ1", "stream-1")
        detector.update({"Pressure": 1.0, "Valve": 2.0}, 0.0)
        result = detector.update({"Pressure": 3.0, "Valve": 6.0}, 1.0)
        evidence = {item["feature"] for item in result["top_evidence"]}
        self.assertTrue(evidence)
        self.assertLessEqual(evidence, {"delta::Pressure", "delta::Valve"})

    def test_invalid_or_duplicate_profile_indices_fail_closed(self):
        base = {
            "window_size": 2,
            "score_mode": "mean",
            "feature_indices": [0, 0],
            "persistence_required": 1,
            "persistence_span": 1,
            "base_scale": 1.0,
            "calibration": [1.0, 1.0],
        }
        with self.assertRaisesRegex(ValueError, "feature indices"):
            V5MultiscaleDetector(
                ZeroAutoEncoder(), small_feature_spec(), [base], 0.5,
                timing_contract(), "test-v5", "model-hash")

    @unittest.skipUnless(
        ARTIFACT_PATH.is_file() and SELECTION_DATA_PATH.is_file(),
        "V5 frozen artifact is unavailable")
    def test_frozen_v5_stream_scores_match_offline_profiles(self):
        artifact = torch.load(
            ARTIFACT_PATH, map_location="cpu", weights_only=False)
        model = SlidingWindowLSTMAutoEncoder(
            len(artifact["feature_spec"]["feature_names"]),
            artifact["hidden_size"], artifact["latent_size"])
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        profiles = []
        for index, source in enumerate(artifact["profiles"], start=1):
            profiles.append({
                "profile_id": f"profile_{index}",
                **source,
            })
        data = np.load(SELECTION_DATA_PATH, allow_pickle=True)
        sequence = np.asarray(data["X_val_anom"][0], dtype=np.float32)
        transformed = transform_sequence(sequence, artifact["feature_spec"])
        offline = profile_score_outputs(model, [transformed], profiles)
        statistics = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        nominal = float(statistics["sampling"]["median_interval"])
        detector = V5MultiscaleDetector(
            model, artifact["feature_spec"], profiles,
            artifact["threshold"], {
                "timestamp_required": True,
                "minimum_interval_seconds": nominal * 0.9,
                "maximum_interval_seconds": nominal * 1.1,
                "nominal_interval_seconds": nominal,
                "sensor_timeout_seconds": nominal * 3,
            }, "test-v5", "model-hash")
        detector.start_stream("W1", "R1", "EQ1", "stream-1")
        names = artifact["feature_spec"]["raw_sensor_names"]
        for sample_index, row in enumerate(sequence):
            result = detector.update(
                dict(zip(names, row)), sample_index * nominal)
            available = []
            for output in offline:
                profile_id = output["profile"]["profile_id"]
                curve_index = sample_index - output["first_sample"]
                actual = result["profiles"][profile_id]["score"]
                if curve_index < 0:
                    self.assertIsNone(actual)
                    continue
                expected = float(output["curves"][0][curve_index])
                self.assertAlmostEqual(actual, expected, places=5)
                available.append(expected)
            if available:
                self.assertEqual(
                    result["alarm"], max(available) > artifact["threshold"])


if __name__ == "__main__":
    unittest.main()
