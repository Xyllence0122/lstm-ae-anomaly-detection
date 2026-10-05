import json
import unittest
from pathlib import Path

import numpy as np

from v5_mqtt_protocol import (
    EdgeMessageProcessor,
    ProtocolError,
    decode_json_object,
    encode_json,
    percentile_summary,
    topic_names,
    validate_sensor_message,
)


class FakeDetector:
    raw_sensor_names = ["Pressure", "Valve"]
    schema_hash = "schema-hash"

    def __init__(self):
        self.started = []
        self.updates = []

    def start_stream(self, wafer_id, recipe_id, equipment_id, stream_id):
        self.started.append((wafer_id, recipe_id, equipment_id, stream_id))

    def update(self, sample, timestamp):
        self.updates.append((sample, timestamp))
        alarm = sample["Pressure"] > 5
        return {
            "alarm_ready": True,
            "alarm": alarm,
            "score": sample["Pressure"],
            "threshold": 5.0,
            "trigger_profile_id": "test-profile",
            "top_evidence": [],
            "model_version": "test-v5",
            "model_artifact_sha256": "model-hash",
            "deployment_manifest_sha256": "manifest-hash",
        }


def sensor_message(index=0, stream="stream-1", pressure=1.0):
    return {
        "protocol_version": 1,
        "message_id": f"{stream}-{index}",
        "run_id": "run-1",
        "stream_instance_id": stream,
        "wafer_id": "wafer-1",
        "recipe_id": "recipe-1",
        "equipment_id": "equipment-1",
        "sample_index": index,
        "sequence_length": 3,
        "sample_timestamp": float(index),
        "published_unix_ns": 1,
        "sensor_schema_hash": "schema-hash",
        "sensor_columns": ["Pressure", "Valve"],
        "sensor_values": [pressure, 2.0],
    }


class V5MqttProtocolTests(unittest.TestCase):
    def test_json_duplicate_keys_fail_closed(self):
        with self.assertRaisesRegex(ProtocolError, "duplicate JSON key"):
            decode_json_object('{"message_id":"a","message_id":"b"}')

    def test_schema_order_and_unknown_fields_fail_closed(self):
        message = sensor_message()
        validated = validate_sensor_message(
            message, ["Pressure", "Valve"], "schema-hash")
        self.assertEqual(validated["sensor_values"], [1.0, 2.0])

        wrong_order = dict(message)
        wrong_order["sensor_columns"] = ["Valve", "Pressure"]
        with self.assertRaisesRegex(ProtocolError, "column order"):
            validate_sensor_message(
                wrong_order, ["Pressure", "Valve"], "schema-hash")

        unknown = dict(message)
        unknown["ground_truth"] = "must-not-reach-edge"
        with self.assertRaisesRegex(ProtocolError, "unknown"):
            validate_sensor_message(
                unknown, ["Pressure", "Valve"], "schema-hash")

    def test_processor_deduplicates_qos_redelivery_and_emits_alarm_edge(self):
        detector = FakeDetector()
        results = []
        alarms = []
        processor = EdgeMessageProcessor(
            detector, results.append, alarms.append,
            clock_ns=lambda: 10, performance_ns=iter(
                [0, 1_000_000, 2_000_000, 3_000_000,
                 4_000_000, 6_000_000]).__next__)

        first = sensor_message(index=0)
        processor.handle_payload(encode_json(first))
        processor.handle_payload(encode_json(first))
        second = sensor_message(index=1, pressure=10.0)
        processor.handle_payload(encode_json(second))

        self.assertEqual(len(detector.updates), 2)
        self.assertEqual(processor.statistics["duplicate_redeliveries"], 1)
        self.assertTrue(results[1]["duplicate_redelivery"])
        self.assertEqual(len(alarms), 1)
        self.assertTrue(alarms[0]["decision"]["rising_edge"])

    def test_gap_is_rejected_and_new_stream_counts_incomplete_transition(self):
        detector = FakeDetector()
        results = []
        processor = EdgeMessageProcessor(
            detector, results.append, lambda _value: None)
        processor.handle_payload(encode_json(sensor_message(index=0)))
        rejected = processor.handle_payload(
            encode_json(sensor_message(index=2)))
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("expected 1", rejected["reason"])
        processor.handle_payload(encode_json(
            sensor_message(index=0, stream="stream-2")))
        self.assertEqual(
            processor.statistics["incomplete_stream_transitions"], 1)

    def test_invalid_schema_response_keeps_message_identity(self):
        detector = FakeDetector()
        results = []
        processor = EdgeMessageProcessor(
            detector, results.append, lambda _value: None)
        message = sensor_message()
        message["sensor_schema_hash"] = "wrong"
        response = processor.handle_payload(encode_json(message))
        self.assertEqual(response["message_id"], message["message_id"])
        self.assertEqual(response["status"], "rejected")

    def test_latency_summary_and_topic_validation(self):
        self.assertEqual(percentile_summary([]), None)
        summary = percentile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(topic_names("lab/v5")["alarm"], "lab/v5/alarm")
        with self.assertRaises(ValueError):
            topic_names("lab/+")

    @unittest.skipUnless(
        Path("outputs/v5/deployment_manifest_v5.json").is_file() and
        Path("outputs/v3/sensor_stats_v3_2.json").is_file(),
        "frozen V5 deployment package is unavailable")
    def test_frozen_runtime_mqtt_decisions_match_direct_updates(self):
        from v3_data import generate_set, load_statistics
        from v5_edge_runtime import V5MultiscaleDetector

        statistics = load_statistics(
            Path("outputs/v3/sensor_stats_v3_2.json"))
        sequence = generate_set(
            np.random.default_rng(740199), statistics, 1,
            anomaly=2)[0]
        direct = V5MultiscaleDetector.from_manifest()
        wrapped = V5MultiscaleDetector.from_manifest()
        direct.start_stream("wafer-1", "recipe-1", "equipment-1", "stream-1")
        responses = []
        alarms = []
        processor = EdgeMessageProcessor(
            wrapped, responses.append, alarms.append)
        nominal = float(
            wrapped.timing_contract["nominal_interval_seconds"])

        direct_decisions = []
        for index, row in enumerate(sequence):
            message = {
                **sensor_message(index=index),
                "sequence_length": len(sequence),
                "sample_timestamp": index * nominal,
                "sensor_schema_hash": wrapped.schema_hash,
                "sensor_columns": wrapped.raw_sensor_names,
                "sensor_values": [float(value) for value in row],
            }
            response = processor.handle_payload(encode_json(message))
            expected = direct.update(
                dict(zip(direct.raw_sensor_names, row)), index * nominal)
            self.assertEqual(response["status"], "accepted")
            self.assertEqual(
                response["decision"]["alarm"], expected["alarm"])
            if expected["score"] is None:
                self.assertIsNone(response["decision"]["score"])
            else:
                self.assertAlmostEqual(
                    response["decision"]["score"], expected["score"],
                    places=10)
            direct_decisions.append(expected["alarm"])

        self.assertEqual(len(responses), len(sequence))
        self.assertEqual(
            processor.statistics["rejected_messages"], 0)
        self.assertEqual(
            len(alarms), sum(
                alarm and (index == 0 or not direct_decisions[index - 1])
                for index, alarm in enumerate(direct_decisions)))


if __name__ == "__main__":
    unittest.main()
