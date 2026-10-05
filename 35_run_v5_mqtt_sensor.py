# -*- coding: utf-8 -*-
"""Publish deterministic synthetic sensors and measure MQTT V5 round trips."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

from config import OUTPUT_DIR
from deployment_manifest import file_sha256
from v3_data import generate_set, load_statistics
from v4_hashing import normalized_text_sha256
from v5_edge_runtime import (
    DEFAULT_MANIFEST,
    V5MultiscaleDetector,
    load_v5_manifest,
)
from v5_mqtt_protocol import (
    DEFAULT_TOPIC_PREFIX,
    PROTOCOL_VERSION,
    decode_json_object,
    encode_json,
    percentile_summary,
    topic_names,
)


DEFAULT_STATS = OUTPUT_DIR / "v3" / "sensor_stats_v3_2.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "v5" / "mqtt_e2e_v5.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic-prefix", default=DEFAULT_TOPIC_PREFIX)
    parser.add_argument("--username")
    parser.add_argument("--password-env", default="MQTT_PASSWORD")
    parser.add_argument("--tls-ca", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normal", type=int, default=10)
    parser.add_argument("--type-a", type=int, default=3)
    parser.add_argument("--type-b", type=int, default=3)
    parser.add_argument("--type-c", type=int, default=3)
    parser.add_argument("--seed", type=int, default=740101)
    parser.add_argument("--publish-interval-ms", type=float, default=20.0)
    parser.add_argument("--response-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    counts = (args.normal, args.type_a, args.type_b, args.type_c)
    if any(value < 0 for value in counts) or sum(counts) < 1:
        parser.error("sequence counts must be nonnegative with a positive sum")
    if args.publish_interval_ms < 0 or args.response_timeout_seconds <= 0:
        parser.error("interval must be nonnegative and timeout positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    return args


def configure_security(client, args):
    if args.username:
        password = os.environ.get(args.password_env)
        if password is None:
            raise ValueError(
                f"environment variable {args.password_env} is required")
        client.username_pw_set(args.username, password)
    if args.tls_ca is not None:
        client.tls_set(ca_certs=str(args.tls_ca))


def classification_metrics(tp, fn, fp, tn):
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and
            precision + recall else None),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def main():
    args = parse_args()
    topics = topic_names(args.topic_prefix)
    manifest, manifest_path, manifest_hash = load_v5_manifest(args.manifest)
    contract = manifest["model_contract"]
    sensor_names = list(contract["feature_spec"]["raw_sensor_names"])
    schema_hash = contract["raw_sensor_schema_hash"]
    nominal_interval = float(
        manifest["timing_contract"]["nominal_interval_seconds"])
    statistics_record = manifest["artifacts"]["source_statistics"]
    if normalized_text_sha256(args.statistics) != statistics_record["sha256"]:
        raise ValueError("statistics do not match the V5 deployment manifest")
    statistics = load_statistics(args.statistics)
    rng = np.random.default_rng(args.seed)
    generated = []
    for anomaly_type, count in enumerate((
            args.normal, args.type_a, args.type_b, args.type_c)):
        if not count:
            continue
        sequences, metadata = generate_set(
            rng, statistics, count, anomaly=anomaly_type,
            with_metadata=True)
        generated.extend(zip(sequences, metadata))

    run_id = f"mqtt-v5-{uuid.uuid4()}"
    prepared = [
        (f"{run_id}-sequence-{index}", sequence, metadata)
        for index, (sequence, metadata) in enumerate(generated)
    ]
    offline_detector = V5MultiscaleDetector.from_manifest(args.manifest)
    offline_expectations = {}
    offline_started = time.perf_counter()
    for sequence_index, (stream_id, sequence, _metadata) in enumerate(prepared):
        offline_detector.start_stream(
            f"synthetic-{sequence_index}", "synthetic-v3-generator",
            "sensor-simulator", stream_id)
        previous_alarm = False
        for sample_index, row in enumerate(sequence):
            message_id = f"{stream_id}-sample-{sample_index}"
            decision = offline_detector.update(
                dict(zip(sensor_names, row)),
                sample_index * nominal_interval)
            offline_expectations[message_id] = {
                "alarm_ready": bool(decision["alarm_ready"]),
                "alarm": bool(decision["alarm"]),
                "rising_edge": bool(
                    decision["alarm"] and not previous_alarm),
                "score": decision["score"],
                "trigger_profile_id": decision["trigger_profile_id"],
            }
            previous_alarm = bool(decision["alarm"])
    offline_replay_seconds = time.perf_counter() - offline_started

    lock = threading.Lock()
    connection_finished = threading.Event()
    subscribed = threading.Event()
    connection_errors = []
    sent = {}
    results = {}
    result_rtt_ms = {}
    alarm_rtt_ms = []
    alarm_message_ids = set()
    alarms_by_stream = set()
    duplicate_results = 0
    duplicate_alarms = 0
    malformed_responses = 0

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"v5-simulator-{platform.node()}-{os.getpid()}",
        protocol=mqtt.MQTTv311,
    )
    configure_security(client, args)

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        nonlocal connection_errors
        if reason_code.is_failure:
            connection_errors.append(
                f"broker rejected MQTT connection: {reason_code}")
            connection_finished.set()
            return
        result, _mid = _client.subscribe([
            (topics["result"], 1), (topics["alarm"], 1)])
        if result != mqtt.MQTT_ERR_SUCCESS:
            connection_errors.append(
                f"MQTT subscribe request failed with rc={result}")
        connection_finished.set()

    def on_subscribe(_client, _userdata, _mid, reason_codes, _properties):
        nonlocal connection_errors
        failures = [str(code) for code in reason_codes if code.is_failure]
        if failures:
            connection_errors.append(
                f"broker rejected MQTT subscriptions: {failures}")
        subscribed.set()

    def on_connect_fail(_client, _userdata):
        connection_errors.append(
            "TCP connection to the MQTT broker failed")
        connection_finished.set()

    def on_message(_client, _userdata, message):
        nonlocal duplicate_results, duplicate_alarms, malformed_responses
        arrived = time.perf_counter_ns()
        try:
            document = decode_json_object(message.payload)
            if document.get("run_id") != run_id:
                return
            message_id = document.get("message_id")
            stream_id = document.get("stream_instance_id")
            with lock:
                start = sent.get(message_id)
                if message.topic == topics["alarm"]:
                    if message_id in alarm_message_ids:
                        duplicate_alarms += 1
                        return
                    alarm_message_ids.add(message_id)
                    if start is not None:
                        alarm_rtt_ms.append((arrived - start) / 1_000_000.0)
                    if stream_id:
                        alarms_by_stream.add(stream_id)
                    return
                if message_id in results:
                    duplicate_results += 1
                    return
                results[message_id] = document
                if start is not None:
                    result_rtt_ms[message_id] = (
                        arrived - start) / 1_000_000.0
                decision = document.get("decision") or {}
                if decision.get("rising_edge") and stream_id:
                    alarms_by_stream.add(stream_id)
        except Exception:
            with lock:
                malformed_responses += 1

    client.on_connect = on_connect
    client.on_connect_fail = on_connect_fail
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()
    if not connection_finished.wait(15):
        client.disconnect()
        client.loop_stop()
        raise RuntimeError(
            "MQTT handshake timed out; verify the Pi IP, listener, firewall, "
            "and LAN client isolation")
    if connection_errors:
        client.disconnect()
        client.loop_stop()
        raise RuntimeError(connection_errors[0])
    if not subscribed.wait(15):
        client.disconnect()
        client.loop_stop()
        raise RuntimeError("MQTT subscription acknowledgement timed out")
    if connection_errors:
        client.disconnect()
        client.loop_stop()
        raise RuntimeError(connection_errors[0])

    stream_truth = {}
    expected_by_stream = {}
    publish_started = time.perf_counter()
    interval_seconds = args.publish_interval_ms / 1000.0
    try:
        for sequence_index, (stream_id, sequence, metadata) in enumerate(
                prepared):
            stream_truth[stream_id] = int(metadata["anomaly_type"])
            expected_by_stream[stream_id] = len(sequence)
            for sample_index, row in enumerate(sequence):
                message_id = f"{stream_id}-sample-{sample_index}"
                document = {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_id": message_id,
                    "run_id": run_id,
                    "stream_instance_id": stream_id,
                    "wafer_id": f"synthetic-{sequence_index}",
                    "recipe_id": "synthetic-v3-generator",
                    "equipment_id": "sensor-simulator",
                    "sample_index": sample_index,
                    "sequence_length": len(sequence),
                    "sample_timestamp": sample_index * nominal_interval,
                    "published_unix_ns": time.time_ns(),
                    "sensor_schema_hash": schema_hash,
                    "sensor_columns": sensor_names,
                    "sensor_values": [float(value) for value in row],
                }
                with lock:
                    sent[message_id] = time.perf_counter_ns()
                info = client.publish(
                    topics["sensor"], encode_json(document),
                    qos=1, retain=False)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(
                        f"MQTT publish failed with rc={info.rc}")
                if interval_seconds:
                    time.sleep(interval_seconds)
        publish_seconds = time.perf_counter() - publish_started
        deadline = time.monotonic() + args.response_timeout_seconds
        while time.monotonic() < deadline:
            with lock:
                if len(results) >= len(sent):
                    break
            time.sleep(0.05)
        time.sleep(min(1.0, args.response_timeout_seconds))
    finally:
        client.disconnect()
        client.loop_stop()

    with lock:
        sent_ids = set(sent)
        accepted = {
            key: value for key, value in results.items()
            if value.get("status") == "accepted"
        }
        rejected = {
            key: value for key, value in results.items()
            if value.get("status") == "rejected"
        }
        missing_ids = sorted(sent_ids - set(results))
        edge_processing = [
            float(value["edge_processing_ms"])
            for value in accepted.values()
            if value.get("edge_processing_ms") is not None
        ]
        rtt = [
            result_rtt_ms[key] for key in accepted
            if key in result_rtt_ms
        ]
        transport_overhead = [
            max(result_rtt_ms[key] - float(value["edge_processing_ms"]), 0.0)
            for key, value in accepted.items()
            if key in result_rtt_ms and
            value.get("edge_processing_ms") is not None
        ]

    accepted_by_stream = {}
    for value in accepted.values():
        accepted_by_stream[value["stream_instance_id"]] = (
            accepted_by_stream.get(value["stream_instance_id"], 0) + 1)
    complete_streams = {
        stream_id for stream_id, expected in expected_by_stream.items()
        if accepted_by_stream.get(stream_id, 0) == expected
    }
    tp = fn = fp = tn = 0
    per_type = {str(index): {"detected": 0, "total": 0}
                for index in range(4)}
    for stream_id in complete_streams:
        anomaly_type = stream_truth[stream_id]
        detected = stream_id in alarms_by_stream
        per_type[str(anomaly_type)]["total"] += 1
        per_type[str(anomaly_type)]["detected"] += int(detected)
        if anomaly_type:
            tp += int(detected)
            fn += int(not detected)
        else:
            fp += int(detected)
            tn += int(not detected)
    for value in per_type.values():
        value["detection_rate"] = (
            value["detected"] / value["total"] if value["total"] else None)

    alarm_mismatches = 0
    readiness_mismatches = 0
    rising_edge_mismatches = 0
    profile_mismatches = 0
    score_presence_mismatches = 0
    score_absolute_differences = []
    missing_offline_expectations = 0
    for message_id, response in accepted.items():
        expected = offline_expectations.get(message_id)
        if expected is None:
            missing_offline_expectations += 1
            continue
        actual = response["decision"]
        alarm_mismatches += int(actual["alarm"] != expected["alarm"])
        readiness_mismatches += int(
            actual["alarm_ready"] != expected["alarm_ready"])
        rising_edge_mismatches += int(
            actual["rising_edge"] != expected["rising_edge"])
        profile_mismatches += int(
            actual["trigger_profile_id"] != expected["trigger_profile_id"])
        if (actual["score"] is None) != (expected["score"] is None):
            score_presence_mismatches += 1
        elif actual["score"] is not None:
            score_absolute_differences.append(abs(
                float(actual["score"]) - float(expected["score"])))
    parity_passed = (
        len(accepted) == len(offline_expectations) and
        missing_offline_expectations == 0 and alarm_mismatches == 0 and
        readiness_mismatches == 0 and rising_edge_mismatches == 0 and
        profile_mismatches == 0 and score_presence_mismatches == 0)

    raw_output = args.output.with_name(
        f"{args.output.stem}_latencies.npz")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_output,
        result_round_trip_ms=np.asarray(rtt, dtype=np.float64),
        edge_processing_ms=np.asarray(edge_processing, dtype=np.float64),
        mqtt_round_trip_overhead_ms=np.asarray(
            transport_overhead, dtype=np.float64),
        external_alarm_round_trip_ms=np.asarray(
            alarm_rtt_ms, dtype=np.float64),
    )
    report = {
        "status": (
            "complete" if not missing_ids and not rejected and
            len(complete_streams) == len(stream_truth) and parity_passed
            else "incomplete"),
        "protocol": {
            "run_id": run_id,
            "broker": f"{args.broker}:{args.port}",
            "topics": topics,
            "qos": 1,
            "publish_interval_ms": args.publish_interval_ms,
            "nominal_sensor_interval_seconds": nominal_interval,
            "mode": (
                "approximately_realtime" if abs(
                    interval_seconds - nominal_interval) < 0.01
                else "accelerated_transport_replay"),
            "seed": args.seed,
            "sequence_counts": {
                "normal": args.normal, "type_a": args.type_a,
                "type_b": args.type_b, "type_c": args.type_c,
            },
        },
        "delivery": {
            "published_samples": len(sent),
            "received_results": len(results),
            "accepted_results": len(accepted),
            "rejected_results": len(rejected),
            "missing_results": len(missing_ids),
            "missing_message_ids_first_20": missing_ids[:20],
            "duplicate_results": duplicate_results,
            "duplicate_alarm_messages": duplicate_alarms,
            "malformed_responses": malformed_responses,
            "complete_sequences": len(complete_streams),
            "incomplete_sequences": len(stream_truth) - len(complete_streams),
            "publish_wall_seconds": publish_seconds,
            "publish_samples_per_second": (
                len(sent) / publish_seconds if publish_seconds else None),
        },
        "latency": {
            "decision_round_trip": percentile_summary(rtt),
            "edge_processing": percentile_summary(edge_processing),
            "mqtt_transport_queue_round_trip_overhead": percentile_summary(
                transport_overhead),
            "external_alarm_round_trip": percentile_summary(alarm_rtt_ms),
            "definition": (
                "Decision round trip is measured on the simulator host from "
                "publish call to result receipt. Subtracting Pi-reported edge "
                "processing estimates combined outbound/inbound MQTT, broker, "
                "queue, and callback overhead without cross-host clock sync."),
        },
        "sequence_detection_on_complete_replays_only": {
            "metrics": classification_metrics(tp, fn, fp, tn),
            "per_type": per_type,
            "alarm_rising_edge_streams": len(alarms_by_stream),
        },
        "offline_edge_decision_parity": {
            "passed": parity_passed,
            "offline_replay_seconds": offline_replay_seconds,
            "expected_samples": len(offline_expectations),
            "compared_accepted_samples": len(accepted),
            "missing_offline_expectations": missing_offline_expectations,
            "alarm_mismatches": alarm_mismatches,
            "alarm_readiness_mismatches": readiness_mismatches,
            "rising_edge_mismatches": rising_edge_mismatches,
            "trigger_profile_mismatches": profile_mismatches,
            "score_presence_mismatches": score_presence_mismatches,
            "maximum_score_absolute_difference": (
                max(score_absolute_differences)
                if score_absolute_differences else None),
            "interpretation": (
                "The simulator computes reference decisions before MQTT "
                "publishing with the same locked manifest. No ground-truth "
                "label is sent to the edge service."),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "paho_mqtt": importlib.metadata.version("paho-mqtt"),
            "host": platform.node(),
            "machine": platform.machine(),
        },
        "provenance": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "model_version": manifest["model_version"],
            "statistics_path": str(args.statistics.resolve()),
            "statistics_sha256": statistics_record["sha256"],
            "simulator_source_sha256": normalized_text_sha256(Path(__file__)),
            "protocol_source_sha256": normalized_text_sha256(
                Path(__file__).with_name("v5_mqtt_protocol.py")),
            "raw_latency_path": str(raw_output.resolve()),
            "raw_latency_sha256": file_sha256(raw_output),
        },
        "limitations": [
            "Synthetic sequences come from the existing generator family and "
            "are not an independent model holdout.",
            "This report must not be used to tune weights, profiles, or the "
            "locked threshold.",
            "Accelerated replay verifies transport capacity but is not a "
            "wall-clock long-duration stability test.",
            "MQTT overhead combines both directions, broker, queues, and "
            "callbacks; one-way network delay is not isolated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
