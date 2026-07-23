# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deployment_manifest import sensor_schema_hash
from v3_features import transform_sequence
from v4_edge_runtime import (
    DEFAULT_MANIFEST,
    JsonlAlarmRecorder,
    V4MultiscaleDetector,
    load_v4_manifest,
)


class ZeroAutoEncoder(torch.nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


def feature_spec():
    names = ["Pressure", "Valve"]
    return {
        "version": 1,
        "raw_sensor_names": names,
        "raw_schema_hash": sensor_schema_hash(names),
        "feature_names": [
            "Pressure", "Valve",
            "delta::Pressure", "delta::Valve",
            "elapsed_phase",
        ],
        "expected_cycle_samples": 5.0,
        "feature_mean": [0.0] * 5,
        "feature_std": [1.0] * 5,
        "score_feature_indices": [0, 1, 2, 3],
    }


def profiles():
    return [
        {
            "profile_id": "short_mean",
            "window_size": 2,
            "score_mode": "mean",
            "persistence_required": 1,
            "persistence_span": 1,
            "base_scale": 1.0,
            "calibration": [1.0] * 4,
        },
        {
            "profile_id": "long_mean_2of3",
            "window_size": 4,
            "score_mode": "mean",
            "persistence_required": 2,
            "persistence_span": 3,
            "base_scale": 1.0,
            "calibration": [1.0] * 4,
        },
    ]


def timing_contract():
    return {
        "timestamp_required": True,
        "nominal_interval_seconds": 1.0,
        "minimum_interval_seconds": 0.5,
        "maximum_interval_seconds": 1.5,
        "sensor_timeout_seconds": 3.0,
    }


def detector(threshold=0.5):
    return V4MultiscaleDetector(
        ZeroAutoEncoder(),
        feature_spec(),
        profiles(),
        threshold,
        timing_contract(),
        model_version="test-v4",
        artifact_sha256="model-hash",
        manifest_sha256="manifest-hash",
    )


class V4RuntimeTests(unittest.TestCase):
    def test_runtime_feature_rows_match_offline_transform(self):
        model = detector(threshold=100.0)
        model.start_stream("W1", "R1", "EQ1", "stream-1")
        sequence = np.asarray([
            [1.0, 3.0],
            [2.0, 4.0],
            [4.0, 3.0],
        ], dtype=np.float32)
        for index, row in enumerate(sequence):
            model.update(
                {"Pressure": row[0], "Valve": row[1]},
                timestamp=float(index),
            )
        expected = transform_sequence(sequence, feature_spec())
        np.testing.assert_allclose(
            np.stack(model.feature_buffer), expected, atol=0.0, rtol=0.0)

    def test_runtime_requires_explicit_stream_context(self):
        model = detector()
        with self.assertRaises(RuntimeError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=0.0)
        with self.assertRaises(ValueError):
            model.start_stream("", "R1", "EQ1", "stream-1")
        with self.assertRaises(ValueError):
            model.start_stream("W1", "R1", "EQ1", "")

    def test_runtime_rejects_schema_and_nonfinite_values(self):
        model = detector()
        model.start_stream("W1", "R1", "EQ1", "stream-1")
        with self.assertRaises(ValueError):
            model.update(
                {"Valve": 2.0, "Pressure": 1.0}, timestamp=0.0)
        with self.assertRaises(ValueError):
            model.update(
                {"Pressure": np.nan, "Valve": 2.0}, timestamp=0.0)

    def test_runtime_enforces_timestamp_and_cadence_contract(self):
        model = detector()
        model.start_stream("W1", "R1", "EQ1", "stream-1")
        with self.assertRaises(ValueError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=None)
        model.update(
            {"Pressure": 1.0, "Valve": 2.0}, timestamp=10.0)
        with self.assertRaises(ValueError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=10.0)
        with self.assertRaises(ValueError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=10.1)
        with self.assertRaises(ValueError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=12.0)
        result = model.update(
            {"Pressure": 1.0, "Valve": 2.0}, timestamp=11.0)
        self.assertEqual(result["sampling_interval_seconds"], 1.0)

    def test_multiscale_readiness_and_reset_are_deterministic(self):
        model = detector(threshold=0.1)
        sample = {"Pressure": 1.0, "Valve": 2.0}
        model.start_stream("W1", "R1", "EQ1", "stream-1")
        first_run = [
            model.update(sample, float(index))
            for index in range(6)
        ]
        self.assertFalse(first_run[0]["alarm_ready"])
        self.assertTrue(first_run[1]["profiles"]["short_mean"][
            "alarm_ready"])
        self.assertFalse(first_run[4]["profiles"]["long_mean_2of3"][
            "alarm_ready"])
        self.assertTrue(first_run[5]["profiles"]["long_mean_2of3"][
            "alarm_ready"])
        model.start_stream("W2", "R1", "EQ1", "stream-2")
        second_run = [
            model.update(sample, float(index))
            for index in range(6)
        ]
        self.assertEqual(
            [item["score"] for item in first_run],
            [item["score"] for item in second_run],
        )
        self.assertEqual(second_run[0]["wafer_id"], "W2")

    def test_liveness_reports_sensor_timeout(self):
        model = detector()
        model.start_stream("W1", "R1", "EQ1", "stream-1")
        self.assertEqual(
            model.liveness(0.0)["reason"], "no_sample_received")
        model.update(
            {"Pressure": 1.0, "Valve": 2.0}, timestamp=10.0)
        self.assertTrue(model.liveness(12.9)["healthy"])
        self.assertFalse(model.liveness(13.1)["healthy"])
        with self.assertRaises(RuntimeError):
            model.update(
                {"Pressure": 1.0, "Valve": 2.0}, timestamp=11.0)
        model.start_stream("W1", "R1", "EQ1", "stream-2")
        model.update(
            {"Pressure": 1.0, "Valve": 2.0}, timestamp=11.0)

    def test_liveness_nonfinite_time_latches_stream(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                model = detector()
                model.start_stream(
                    "W1", "R1", "EQ1", f"stream-{value}")
                model.update(
                    {"Pressure": 1.0, "Valve": 2.0}, timestamp=10.0)
                with self.assertRaises(ValueError):
                    model.liveness(value)
                self.assertTrue(model.timeout_latched)
                with self.assertRaises(RuntimeError):
                    model.update(
                        {"Pressure": 1.0, "Valve": 2.0},
                        timestamp=11.0,
                    )

    def test_jsonl_recorder_saves_pre_and_post_alarm_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = JsonlAlarmRecorder(
                path, pre_samples=2, post_samples=2)
            base = {
                "wafer_id": "W1",
                "recipe_id": "R1",
                "equipment_id": "EQ1",
                "stream_instance_id": "stream-1",
                "threshold": 1.0,
                "trigger_profile_id": "short",
                "top_evidence": [],
                "model_version": "v4",
                "model_artifact_sha256": "model",
                "deployment_manifest_sha256": "manifest",
                "raw_sensor_schema_hash": "schema",
            }
            alarms = [False, False, True, True, False]
            for index, alarm in enumerate(alarms):
                result = {
                    **base,
                    "sample_index": index,
                    "timestamp": float(index),
                    "score": 2.0 if alarm else 0.0,
                    "alarm": alarm,
                }
                recorder.update(
                    {"Pressure": index, "Valve": index + 1}, result)
            recorder.close()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["status"], "complete")
            self.assertEqual(event["alarm_sample_index"], 2)
            self.assertEqual(
                [item["sample_index"] for item in event["pre_context"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [item["sample_index"] for item in event["post_context"]],
                [3, 4],
            )

    def test_jsonl_recorder_does_not_mix_stream_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = JsonlAlarmRecorder(
                path, pre_samples=2, post_samples=0)
            base = {
                "wafer_id": "W1",
                "recipe_id": "R1",
                "equipment_id": "EQ1",
                "threshold": 1.0,
                "trigger_profile_id": "short",
                "top_evidence": [],
                "model_version": "v4",
                "model_artifact_sha256": "model",
                "deployment_manifest_sha256": "manifest",
                "raw_sensor_schema_hash": "schema",
            }
            for stream_id, index, pressure, alarm in (
                    ("stream-1", 0, 99.0, False),
                    ("stream-2", 0, 0.0, False),
                    ("stream-2", 1, 1.0, True)):
                recorder.update(
                    {"Pressure": pressure, "Valve": pressure + 1},
                    {
                        **base,
                        "stream_instance_id": stream_id,
                        "sample_index": index,
                        "timestamp": float(index),
                        "score": 2.0 if alarm else 0.0,
                        "alarm": alarm,
                    },
                )
            recorder.close()
            event = json.loads(path.read_text(
                encoding="utf-8").splitlines()[0])
            self.assertEqual(event["wafer_id"], "W1")
            self.assertEqual(event["stream_instance_id"], "stream-2")
            self.assertEqual(
                {item["sample_index"] for item in event["pre_context"]},
                {0, 1},
            )
            self.assertTrue(all(
                item["sensors"]["Pressure"] in (0.0, 1.0)
                for item in event["pre_context"]))

    def test_jsonl_write_failure_preserves_active_event_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = JsonlAlarmRecorder(
                path, pre_samples=2, post_samples=1)
            base = {
                "wafer_id": "W1",
                "recipe_id": "R1",
                "equipment_id": "EQ1",
                "stream_instance_id": "stream-1",
                "threshold": 1.0,
                "trigger_profile_id": "short",
                "top_evidence": [],
                "model_version": "v4",
                "model_artifact_sha256": "model",
                "deployment_manifest_sha256": "manifest",
                "raw_sensor_schema_hash": "schema",
            }

            def send(index, alarm):
                recorder.update(
                    {"Pressure": index, "Valve": index + 1},
                    {
                        **base,
                        "sample_index": index,
                        "timestamp": float(index),
                        "score": 2.0 if alarm else 0.0,
                        "alarm": alarm,
                    },
                )

            send(0, False)
            send(1, True)
            original_write = recorder._write
            recorder._write = lambda event: (
                _ for _ in ()).throw(OSError("disk full"))
            with self.assertRaises(OSError):
                send(2, False)
            self.assertEqual(len(recorder.active), 1)
            self.assertEqual(
                recorder.active[0]["remaining_post_samples"], 0)
            self.assertTrue(recorder.storage_error_latched)
            with self.assertRaises(RuntimeError):
                send(3, False)
            recorder._write = original_write
            recorder.retry_pending_writes()
            self.assertFalse(recorder.storage_error_latched)
            send(3, False)
            self.assertEqual(len(recorder.active), 0)
            recorder.close()
            event = json.loads(path.read_text(
                encoding="utf-8").splitlines()[0])
            self.assertEqual(
                [item["sample_index"] for item in event["post_context"]],
                [2],
            )

    def test_jsonl_recorder_uses_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = JsonlAlarmRecorder(path)
            with self.assertRaises(RuntimeError):
                JsonlAlarmRecorder(path)
            first.close()
            second = JsonlAlarmRecorder(path)
            second.close()

    def test_jsonl_event_id_is_idempotent_for_stable_stream_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            result = {
                "wafer_id": "W1",
                "recipe_id": "R1",
                "equipment_id": "EQ1",
                "stream_instance_id": "stable-run-id",
                "sample_index": 7,
                "timestamp": 7.0,
                "score": 2.0,
                "threshold": 1.0,
                "alarm": True,
                "trigger_profile_id": "short",
                "top_evidence": [],
                "model_version": "v4",
                "model_artifact_sha256": "model",
                "deployment_manifest_sha256": "manifest",
                "raw_sensor_schema_hash": "schema",
            }
            sample = {"Pressure": 1.0, "Valve": 2.0}
            first = JsonlAlarmRecorder(path, post_samples=0)
            first.update(sample, result)
            first.close()
            second = JsonlAlarmRecorder(path, post_samples=0)
            second.update(sample, result)
            second.close()
            self.assertEqual(
                len(path.read_text(encoding="utf-8").splitlines()), 1)

    @unittest.skipUnless(
        DEFAULT_MANIFEST.is_file(),
        "V4 deployment package has not been built")
    def test_built_manifest_and_runtime_load_with_verified_hashes(self):
        document, _, manifest_hash = load_v4_manifest(DEFAULT_MANIFEST)
        model = V4MultiscaleDetector.from_manifest(DEFAULT_MANIFEST)
        self.assertEqual(document["manifest_version"], 4)
        self.assertEqual(model.manifest_sha256, manifest_hash)
        self.assertEqual(
            model.schema_hash,
            document["model_contract"]["raw_sensor_schema_hash"],
        )

    @unittest.skipUnless(
        DEFAULT_MANIFEST.is_file(),
        "V4 deployment package has not been built")
    def test_manifest_sidecar_rejects_modified_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment_manifest_v4.json"
            sidecar = path.with_suffix(".sha256")
            path.write_bytes(DEFAULT_MANIFEST.read_bytes() + b" ")
            sidecar.write_bytes(
                DEFAULT_MANIFEST.with_suffix(".sha256").read_bytes())
            with self.assertRaises(ValueError):
                load_v4_manifest(path)


if __name__ == "__main__":
    unittest.main()
