import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from models import (
    LSTMForecaster,
    forecaster_pointwise_errors,
    streaming_score_curves,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_generator_module():
    path = PROJECT_DIR / "02_generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("synthetic_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_streaming_experiment_module():
    path = PROJECT_DIR / "07_streaming_early_warning.py"
    spec = importlib.util.spec_from_file_location("streaming_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamingModelTests(unittest.TestCase):
    def test_forecaster_does_not_use_future_samples(self):
        torch.manual_seed(7)
        model = LSTMForecaster(n_features=1, hidden_size=4)
        common = np.arange(5, dtype=np.float32)[:, None]
        first = np.vstack([common, [[5.0], [6.0], [7.0]]])
        second = np.vstack([common, [[50.0], [60.0], [70.0]]])

        errors = forecaster_pointwise_errors(model, [first, second])

        # Error indices 0..3 predict/observe samples 1..4. Those samples and
        # every preceding input are identical, so later divergence cannot alter
        # these already-emittable errors.
        np.testing.assert_allclose(errors[0][:4], errors[1][:4], atol=1e-7)
        self.assertFalse(np.allclose(errors[0][4:], errors[1][4:]))

    def test_streaming_score_uses_trailing_window(self):
        errors = [np.asarray([
            [1.0, 4.0],
            [3.0, 2.0],
            [5.0, 8.0],
        ])]

        curve = streaming_score_curves(errors, window=2)[0]

        # Windows are rows [0,1] then [1,2]; max is taken across sensors.
        np.testing.assert_allclose(curve, [3.0, 5.0])

    def test_pre_onset_crossing_is_not_counted_as_detection(self):
        experiment = load_streaming_experiment_module()
        # With window=2, curve entries map to sample indices 2,3,4,5.
        # There is a threshold crossing before onset (sample 2) and a valid one
        # at onset (sample 4). Only the latter may become the detection time.
        curves = [np.asarray([2.0, 0.0, 3.0, 0.0])]
        pred, alerts, pre_onset, scores = experiment.online_decisions(
            curves, threshold=1.0, window=2, y_test=np.asarray([1]),
            metadata=[{"onset_index": 4}])

        np.testing.assert_array_equal(pred, [1])
        self.assertEqual(alerts, [4])
        np.testing.assert_array_equal(pre_onset, [True])
        np.testing.assert_allclose(scores, [3.0])


class SyntheticMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator_module()

    def test_fast_arrival_reaches_same_endpoint_and_records_onset(self):
        profile = np.linspace(0.0, 100.0, 101)
        sensors = [{
            "index": 8,
            "profile": profile.tolist(),
            "within_wafer_std": 0.0,
            "between_wafer_std": 0.0,
            "lag1_autocorr": 0.0,
            "quant_step": 0.0,
            "transient_amp": 100.0,
        }]
        normal = self.generator.make_wafer(
            np.random.default_rng(11), sensors, (100, 100), anomaly=0,
            fixed_len=100, no_offset=True)
        anomalous, metadata = self.generator.make_wafer(
            np.random.default_rng(11), sensors, (100, 100), anomaly=1,
            fixed_len=100, no_offset=True, return_metadata=True)

        self.assertGreater(anomalous[5, 0], normal[5, 0])
        self.assertAlmostEqual(anomalous[-1, 0], normal[-1, 0], places=6)
        self.assertEqual(metadata["onset_index"], 0)
        self.assertEqual(metadata["affected_sensor_positions"], [0])
        self.assertEqual(metadata["sequence_length"], 100)

    def test_oscillation_and_drift_onsets_are_inside_sequence(self):
        profile = np.linspace(0.0, 100.0, 101)
        sensor = {
            "index": 8,
            "profile": profile.tolist(),
            "within_wafer_std": 1.0,
            "between_wafer_std": 0.0,
            "lag1_autocorr": 0.0,
            "quant_step": 0.0,
            "transient_amp": 100.0,
        }
        for anomaly_type in (2, 3):
            _, metadata = self.generator.make_wafer(
                np.random.default_rng(20 + anomaly_type), [sensor], (100, 100),
                anomaly=anomaly_type, fixed_len=100, no_offset=True,
                return_metadata=True)
            self.assertGreater(metadata["onset_index"], 0)
            self.assertLess(metadata["onset_index"], 99)
            self.assertGreaterEqual(metadata["end_index"], metadata["onset_index"])


if __name__ == "__main__":
    unittest.main()
