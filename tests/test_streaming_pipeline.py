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
    apply_persistence,
    persistence_score_curve,
    sensor_error_score_curves,
    threshold_for_target_fpr,
    wilson_interval,
)
from v3_data import (
    chronological_process_segment,
    generate_wafer,
    load_statistics,
    phase_map,
)
from v3_features import (
    fit_feature_spec,
    sample_training_windows,
    transform_sequence,
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
            detector.update({"Pressure": 0.0}, timestamp=0.0),
            detector.update({"Pressure": 0.0}, timestamp=1.0),
            detector.update({"Pressure": 1.0}, timestamp=2.0),
            detector.update({"Pressure": 0.0}, timestamp=3.0),
            detector.update({"Pressure": 1.0}, timestamp=4.0),
        ]

        self.assertFalse(outputs[1]["window_ready"])
        self.assertTrue(outputs[2]["window_ready"])
        self.assertFalse(outputs[3]["alarm_ready"])
        self.assertTrue(outputs[4]["alarm_ready"])
        self.assertAlmostEqual(outputs[4]["score"], 1.0)
        self.assertTrue(outputs[4]["alarm"])
        detector.reset()
        self.assertFalse(detector.update(
            {"Pressure": 0.0}, timestamp=0.0)["window_ready"])

    def test_sliding_window_runtime_rejects_nonmonotonic_timestamps(self):
        class IdentityAutoEncoder(torch.nn.Module):
            def forward(self, window):
                return window

        detector = SlidingWindowAnomalyDetector(
            IdentityAutoEncoder(), mean=[0.0], std=[1.0], threshold=1.0,
            window_size=2, calib=[1.0], sensor_names=["Pressure"])
        detector.update({"Pressure": 0.0}, timestamp=5.0)
        with self.assertRaises(ValueError):
            detector.update({"Pressure": 0.0}, timestamp=5.0)

    def test_sliding_window_runtime_rejects_schema_mismatch(self):
        class IdentityAutoEncoder(torch.nn.Module):
            def forward(self, window):
                return window

        detector = SlidingWindowAnomalyDetector(
            IdentityAutoEncoder(), mean=[0.0, 0.0], std=[1.0, 1.0],
            threshold=1.0, window_size=2, calib=[1.0, 1.0],
            sensor_names=["Cl2 Flow", "Pressure"])

        with self.assertRaises(ValueError):
            detector.update({"Pressure": 1.0, "Cl2 Flow": 2.0})
        with self.assertRaises(ValueError):
            detector.update({"Cl2 Flow": 2.0})
        with self.assertRaises(ValueError):
            detector.update(
                {"Cl2 Flow": 2.0, "Pressure": 1.0, "Vat Valve": 3.0})
        with self.assertRaises(ValueError):
            detector.update_ordered(
                [2.0, 1.0], ["Cl2 Flow", "Cl2 Flow"])
        with self.assertRaises(TypeError):
            detector.update([2.0, 1.0])

    def test_sliding_window_runtime_rejects_mixed_timestamp_presence(self):
        class IdentityAutoEncoder(torch.nn.Module):
            def forward(self, window):
                return window

        detector = SlidingWindowAnomalyDetector(
            IdentityAutoEncoder(), mean=[0.0], std=[1.0], threshold=1.0,
            window_size=2, calib=[1.0], sensor_names=["Pressure"])
        detector.update({"Pressure": 0.0}, timestamp=10.0)
        with self.assertRaises(ValueError):
            detector.update({"Pressure": 0.0}, timestamp=None)
        with self.assertRaises(ValueError):
            detector.update({"Pressure": 0.0}, timestamp=5.0)

        detector.reset()
        detector.update({"Pressure": 0.0})
        with self.assertRaises(ValueError):
            detector.update({"Pressure": 0.0}, timestamp=1.0)

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
        actual = [detector.update({"sensor_0": sample[0]})["raw_score"]
                  for sample in sequence]
        actual = np.asarray([value for value in actual if value is not None])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_final_profile_reproduces_offline_torchscript_decisions(self):
        checkpoint = torch.load(
            PROJECT_DIR / "outputs" / "sliding_window_lstm_ae.pt",
            map_location="cpu", weights_only=False)
        model = SlidingWindowLSTMAutoEncoder(
            len(checkpoint["mean"]), checkpoint["hidden_size"],
            checkpoint["latent_size"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        data = np.load(
            PROJECT_DIR / "outputs" / "v2_backup" / "synthetic_data.npz",
            allow_pickle=True)
        sequence = np.asarray(data["X_test"][300], dtype=np.float32)
        self.assertEqual(int(data["y_test"][300]), 2)

        normalized = (sequence - checkpoint["mean"]) / checkpoint["std"]
        errors = sliding_window_errors(
            model, [normalized], checkpoint["window_size"],
            checkpoint["score_mode"])
        raw_scores = sensor_error_score_curves(
            errors, checkpoint["calib"])[0]
        persistent_scores = apply_persistence(
            [raw_scores], checkpoint["persistence_required"],
            checkpoint["persistence_span"])[0]

        detector = SlidingWindowAnomalyDetector.from_artifacts()
        emitted = [
            detector.update(
                dict(zip(detector.sensor_names, sample)), timestamp=index)
            for index, sample in enumerate(sequence)
        ]
        actual_raw = np.asarray([
            item["raw_score"] for item in emitted
            if item["raw_score"] is not None])
        actual_alarms = np.asarray([
            item["alarm"] for item in emitted if item["alarm_ready"]])
        expected_alarms = persistent_scores > detector.threshold

        self.assertEqual(
            detector.profile_id, "final_calibration_fpr_1pct")
        self.assertAlmostEqual(detector.threshold, 2.2915715737547067)
        self.assertTrue(np.any(expected_alarms))
        np.testing.assert_allclose(actual_raw, raw_scores, atol=1e-5)
        np.testing.assert_array_equal(actual_alarms, expected_alarms)
        self.assertTrue(all(
            item["schema_hash"] == detector.schema_hash for item in emitted))

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


class V3DataContractTests(unittest.TestCase):
    def test_chronology_repair_sorts_and_averages_duplicate_times(self):
        wafer = np.zeros((5, 21), dtype=np.float64)
        wafer[:, 0] = [3.0, 1.0, 2.0, 1.0, 4.0]
        wafer[:, 1] = [4.0, 4.0, 4.0, 4.0, 2.0]
        wafer[:, 8] = [30.0, 10.0, 20.0, 14.0, 999.0]

        repaired = chronological_process_segment(wafer)

        np.testing.assert_allclose(repaired[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(repaired[:, 8], [12.0, 20.0, 30.0])
        self.assertTrue(np.all(np.diff(repaired[:, 0]) > 0))

    def test_phase_map_preserves_endpoints_and_transition(self):
        progress = np.asarray([0.0, 0.1, 1.0])
        mapped = phase_map(progress, transition=0.1, canonical=0.04)

        np.testing.assert_allclose(mapped, [0.0, 0.04, 1.0])

    def test_v3_features_do_not_use_future_or_actual_cycle_length(self):
        train = [np.arange(32, dtype=np.float32)[:, None]]
        spec = fit_feature_spec(train, ["Pressure"])
        common = np.arange(6, dtype=np.float32)[:, None]
        first = np.vstack([common, [[6.0], [7.0], [8.0]]])
        second = np.vstack([common, [[60.0], [70.0], [80.0], [90.0]]])

        first_features = transform_sequence(first, spec)
        second_features = transform_sequence(second, spec)

        np.testing.assert_allclose(
            first_features[:6], second_features[:6], atol=1e-7)
        self.assertFalse(np.allclose(
            first_features[6:], second_features[6:9]))

    def test_window_sampler_always_includes_start_and_end(self):
        sequence = np.arange(10, dtype=np.float32)[:, None]
        windows = sample_training_windows(
            [sequence], window_sizes=[4], samples_per_size=2, seed=123)

        self.assertEqual(len(windows), 2)
        np.testing.assert_array_equal(windows[0], sequence[:4])
        np.testing.assert_array_equal(windows[1], sequence[-4:])

    def test_dynamic_anomaly_obeys_observable_slew_contract(self):
        statistics = load_statistics(
            PROJECT_DIR / "outputs" / "v3" / "sensor_stats_v3.json")
        rng = np.random.default_rng(9981)
        for _ in range(20):
            sequence, metadata = generate_wafer(
                rng, statistics, anomaly=1, return_metadata=True)
            group = statistics["recipe_groups"][metadata["recipe_group_id"]]
            stop = max(10, int(np.ceil(0.25 * len(sequence))))
            for feature in metadata["affected_sensor_positions"]:
                observed = np.max(np.abs(np.diff(
                    sequence[:stop, feature])))
                required = (
                    metadata["anomaly_slew_ratio"] *
                    group["early_slew_abs_p99"][feature])
                self.assertGreaterEqual(observed + 1e-6, required)


if __name__ == "__main__":
    unittest.main()
