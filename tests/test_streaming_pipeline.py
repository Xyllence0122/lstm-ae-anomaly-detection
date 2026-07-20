import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from models import (
    LSTMForecaster,
    SlidingWindowLSTMAutoEncoder,
    forecaster_pointwise_errors,
    sliding_window_errors,
    sliding_window_last_errors,
    streaming_score_curves,
)
from edge_runtime import StreamingAnomalyDetector
from edge_window_runtime import SlidingWindowAnomalyDetector
from online_evaluation import (
    persistence_score_curve,
    threshold_for_target_fpr,
    wilson_interval,
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
    def test_sliding_window_autoencoder_does_not_use_future_samples(self):
        torch.manual_seed(9)
        model = SlidingWindowLSTMAutoEncoder(
            n_features=1, hidden_size=4, latent_size=2)
        common = np.arange(6, dtype=np.float32)[:, None]
        first = np.vstack([common, [[6.0], [7.0], [8.0]]])
        second = np.vstack([common, [[60.0], [70.0], [80.0]]])

        errors = sliding_window_last_errors(
            model, [first, second], window_size=4)

        # Rows 0..2 end at source samples 3..5, before the sequences diverge.
        np.testing.assert_allclose(errors[0][:3], errors[1][:3], atol=1e-7)
        self.assertFalse(np.allclose(errors[0][3:], errors[1][3:]))

    def test_persistence_score_is_kth_largest_in_trailing_span(self):
        scores = np.asarray([1.0, 4.0, 2.0, 5.0])
        curve = persistence_score_curve(scores, required=2, span=3)
        np.testing.assert_allclose(curve, [2.0, 4.0])

    def test_target_fpr_threshold_respects_empirical_budget(self):
        normal_scores = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
        threshold = threshold_for_target_fpr(normal_scores, 0.2)
        self.assertLessEqual(np.mean(normal_scores > threshold), 0.2)
        zero_fpr_threshold = threshold_for_target_fpr(normal_scores, 0.0)
        self.assertEqual(np.count_nonzero(normal_scores > zero_fpr_threshold), 0)

    def test_wilson_interval_handles_boundary_proportions(self):
        lower_zero, upper_zero = wilson_interval(0, 10)
        lower_one, upper_one = wilson_interval(10, 10)
        self.assertEqual(lower_zero, 0.0)
        self.assertAlmostEqual(upper_zero, 0.2775328, places=6)
        self.assertAlmostEqual(lower_one, 0.7224672, places=6)
        self.assertAlmostEqual(upper_one, 1.0, places=12)

    def test_sliding_window_edge_runtime_and_reset(self):
        class ZeroAutoEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, window):
                return torch.zeros_like(window) + self.dummy

        detector = SlidingWindowAnomalyDetector(
            ZeroAutoEncoder(), mean=[0.0], std=[1.0], threshold=0.5,
            window_size=3, calib=[1.0], persistence_required=2,
            persistence_span=3, sensor_names=["Pressure"])
        outputs = [
            detector.update([0.0], timestamp=0.0),
            detector.update([0.0], timestamp=1.0),
            detector.update([1.0], timestamp=2.0),
            detector.update([0.0], timestamp=3.0),
            detector.update([1.0], timestamp=4.0),
        ]

        self.assertFalse(outputs[1]["window_ready"])
        self.assertTrue(outputs[2]["window_ready"])
        self.assertFalse(outputs[3]["alarm_ready"])
        self.assertTrue(outputs[4]["alarm_ready"])
        self.assertAlmostEqual(outputs[4]["score"], 1.0)
        self.assertTrue(outputs[4]["alarm"])
        detector.reset()
        self.assertFalse(detector.update([0.0], timestamp=0.0)["window_ready"])

    def test_sliding_window_runtime_rejects_nonmonotonic_timestamps(self):
        class IdentityAutoEncoder(torch.nn.Module):
            def forward(self, window):
                return window

        detector = SlidingWindowAnomalyDetector(
            IdentityAutoEncoder(), mean=[0.0], std=[1.0], threshold=1.0,
            window_size=2, calib=[1.0], sensor_names=["Pressure"])
        detector.update([0.0], timestamp=5.0)
        with self.assertRaises(ValueError):
            detector.update([0.0], timestamp=5.0)

    def test_sliding_window_mean_score_matches_edge_runtime(self):
        class ZeroAutoEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, window):
                return torch.zeros_like(window) + self.dummy

        model = ZeroAutoEncoder()
        sequence = np.asarray(
            [[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        expected = sliding_window_errors(
            model, [sequence], window_size=3, reduction="mean")[0][:, 0]
        detector = SlidingWindowAnomalyDetector(
            model, mean=[0.0], std=[1.0], threshold=100.0,
            window_size=3, calib=[1.0], score_mode="mean")
        actual = [detector.update(sample)["raw_score"] for sample in sequence]
        actual = np.asarray([value for value in actual if value is not None])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_edge_runtime_matches_online_error_order(self):
        class EchoStep(torch.nn.Module):
            def forward(self, sample, hidden, cell):
                return sample, hidden, cell

        detector = StreamingAnomalyDetector(
            EchoStep(), mean=[0.0], std=[1.0], threshold=0.5, window=2,
            calib=None, hidden_size=2, sensor_names=["Pressure"])

        first = detector.update([0.0])
        second = detector.update([1.0])
        third = detector.update([2.0])

        self.assertFalse(first["ready"])
        self.assertFalse(second["ready"])
        self.assertTrue(third["ready"])
        self.assertAlmostEqual(third["score"], 1.0)
        self.assertTrue(third["alarm"])
        self.assertAlmostEqual(third["per_sensor_score"]["Pressure"], 1.0)

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
