import unittest

import numpy as np
import torch

from v5_experiment import (
    ANOMALY_NAMES,
    acceptance_gate,
    feature_groups,
    score_profile_outputs,
    weighted_reconstruction_loss,
)


class IdentityModel(torch.nn.Module):
    def forward(self, values):
        return values


class ZeroModel(torch.nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


class V5ExperimentTests(unittest.TestCase):
    def test_feature_groups_are_disjoint_and_exclude_phase(self):
        spec = {
            "raw_sensor_names": ["a", "b"],
            "feature_names": [
                "a", "b", "delta::a", "delta::b", "elapsed_phase"],
        }
        groups = feature_groups(spec)
        self.assertEqual(groups["raw"], [0, 1])
        self.assertEqual(groups["delta"], [2, 3])
        self.assertEqual(groups["raw_delta"], [0, 1, 2, 3])
        self.assertNotIn(4, groups["raw_delta"])

    def test_feature_groups_reject_misordered_schema(self):
        spec = {
            "raw_sensor_names": ["a", "b"],
            "feature_names": [
                "b", "a", "delta::a", "delta::b", "elapsed_phase"],
        }
        with self.assertRaisesRegex(ValueError, "contiguous raw/delta"):
            feature_groups(spec)

    def test_weighted_loss_is_zero_for_exact_reconstruction(self):
        batch = torch.randn(3, 4, 5)
        loss = weighted_reconstruction_loss(
            IdentityModel(), batch, raw_sensor_count=2)
        self.assertEqual(float(loss), 0.0)

    def test_delta_feature_weight_changes_loss_as_predeclared(self):
        batch = torch.zeros(1, 4, 5)
        batch[:, :, 2] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        low = weighted_reconstruction_loss(
            ZeroModel(), batch, raw_sensor_count=2,
            delta_feature_weight=1.0, phase_feature_weight=0.0,
            temporal_difference_weight=0.0)
        high = weighted_reconstruction_loss(
            ZeroModel(), batch, raw_sensor_count=2,
            delta_feature_weight=4.0, phase_feature_weight=0.0,
            temporal_difference_weight=0.0)
        self.assertGreater(float(high), float(low))

    def test_acceptance_gate_requires_every_predeclared_target(self):
        metrics = {"fpr": 0.001, "recall": 0.76, "f1": 0.86}
        per_type = {
            ANOMALY_NAMES[1]: 0.61,
            ANOMALY_NAMES[2]: 0.71,
            ANOMALY_NAMES[3]: 0.81,
        }
        self.assertTrue(acceptance_gate(metrics, per_type)["passed"])
        per_type[ANOMALY_NAMES[1]] = 0.59
        result = acceptance_gate(metrics, per_type)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["type_a_recall"])

    def test_profile_scoring_excludes_pre_onset_and_keeps_earliest_alert(self):
        outputs = [{
            "profile": {"persistence_span": 1},
            "curves": [np.asarray([0.1, 2.0, 0.2, 1.5])],
            "first_sample": 3,
        }, {
            "profile": {"persistence_span": 1},
            "curves": [np.asarray([0.1, 0.2, 0.3, 1.8])],
            "first_sample": 3,
        }]
        scores, alerts, pre_onset = score_profile_outputs(
            outputs, np.asarray([1]), [{"onset_index": 5}], threshold=1.0)
        self.assertAlmostEqual(float(scores[0]), 1.8)
        self.assertEqual(alerts, [6])
        self.assertTrue(bool(pre_onset[0]))


if __name__ == "__main__":
    unittest.main()
