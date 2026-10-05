# -*- coding: utf-8 -*-
"""Strict MQTT message contract and edge processor for V5 experiments."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping


PROTOCOL_VERSION = 1
DEFAULT_TOPIC_PREFIX = "tanet/v5"
SENSOR_TOPIC_SUFFIX = "sensor"
RESULT_TOPIC_SUFFIX = "result"
ALARM_TOPIC_SUFFIX = "alarm"


class ProtocolError(ValueError):
    """A malformed or unsafe MQTT experiment message."""


def topic_names(prefix=DEFAULT_TOPIC_PREFIX):
    prefix = str(prefix).strip().strip("/")
    if not prefix or "+" in prefix or "#" in prefix:
        raise ValueError("topic prefix must be nonempty and contain no wildcard")
    return {
        "sensor": f"{prefix}/{SENSOR_TOPIC_SUFFIX}",
        "result": f"{prefix}/{RESULT_TOPIC_SUFFIX}",
        "alarm": f"{prefix}/{ALARM_TOPIC_SUFFIX}",
    }


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def decode_json_object(payload):
    """Decode UTF-8 JSON while rejecting duplicate keys and non-objects."""
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("payload is not valid UTF-8") from exc
    try:
        value = json.loads(
            payload, object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"non-finite JSON number: {value}")))
    except ProtocolError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("payload root must be a JSON object")
    return value


def encode_json(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def _required_string(document, name):
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a nonempty string")
    return value


def _required_integer(document, name, minimum=0):
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _required_finite_number(document, name):
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError(f"{name} must be finite")
    return value


SENSOR_MESSAGE_FIELDS = {
    "protocol_version", "message_id", "run_id", "stream_instance_id",
    "wafer_id", "recipe_id", "equipment_id", "sample_index",
    "sequence_length", "sample_timestamp", "published_unix_ns",
    "sensor_schema_hash", "sensor_columns", "sensor_values",
}


def validate_sensor_message(document, sensor_names, schema_hash):
    """Validate one message without using any ground-truth anomaly label."""
    if not isinstance(document, Mapping):
        raise ProtocolError("sensor message must be an object")
    unknown = sorted(set(document) - SENSOR_MESSAGE_FIELDS)
    missing = sorted(SENSOR_MESSAGE_FIELDS - set(document))
    if missing or unknown:
        raise ProtocolError(
            f"sensor message fields mismatch: missing={missing}, "
            f"unknown={unknown}")
    if document["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported MQTT experiment protocol version")
    validated = {
        name: _required_string(document, name)
        for name in (
            "message_id", "run_id", "stream_instance_id", "wafer_id",
            "recipe_id", "equipment_id")
    }
    validated["sample_index"] = _required_integer(
        document, "sample_index")
    validated["sequence_length"] = _required_integer(
        document, "sequence_length", minimum=1)
    if validated["sample_index"] >= validated["sequence_length"]:
        raise ProtocolError("sample_index must be below sequence_length")
    validated["sample_timestamp"] = _required_finite_number(
        document, "sample_timestamp")
    validated["published_unix_ns"] = _required_integer(
        document, "published_unix_ns", minimum=1)
    if document["sensor_schema_hash"] != schema_hash:
        raise ProtocolError("sensor schema hash mismatch")
    columns = document["sensor_columns"]
    values = document["sensor_values"]
    if not isinstance(columns, list) or not all(
            isinstance(item, str) for item in columns):
        raise ProtocolError("sensor_columns must be a string list")
    if len(set(columns)) != len(columns):
        raise ProtocolError("sensor_columns contains duplicates")
    if columns != list(sensor_names):
        raise ProtocolError(
            f"sensor column order mismatch: expected {list(sensor_names)}, "
            f"got {columns}")
    if not isinstance(values, list) or len(values) != len(columns):
        raise ProtocolError("sensor_values length does not match columns")
    numeric_values = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"sensor value {index} is not numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ProtocolError(f"sensor value {index} is not finite")
        numeric_values.append(value)
    validated.update({
        "protocol_version": PROTOCOL_VERSION,
        "sensor_schema_hash": schema_hash,
        "sensor_columns": columns,
        "sensor_values": numeric_values,
    })
    return validated


def percentile_summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None

    def percentile(percent):
        position = (len(values) - 1) * percent / 100.0
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return values[low]
        return values[low] + (
            values[high] - values[low]) * (position - low)

    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "maximum_ms": values[-1],
    }


class EdgeMessageProcessor:
    """Apply ordered MQTT samples once and emit decisions and alarm edges."""

    def __init__(
            self, detector, publish_result: Callable[[dict], None],
            publish_alarm: Callable[[dict], None], service_metadata=None,
            clock_ns=time.time_ns, performance_ns=time.perf_counter_ns):
        self.detector = detector
        self.publish_result = publish_result
        self.publish_alarm = publish_alarm
        self.service_metadata = dict(service_metadata or {})
        self.clock_ns = clock_ns
        self.performance_ns = performance_ns
        self.current_stream = None
        self.expected_index = 0
        self.sequence_length = None
        self.stream_context = None
        self.response_cache = {}
        self.last_alarm = False
        self.statistics = {
            "received_messages": 0,
            "accepted_samples": 0,
            "duplicate_redeliveries": 0,
            "rejected_messages": 0,
            "incomplete_stream_transitions": 0,
            "alarm_rising_edges": 0,
        }

    def _base_response(self, message=None, received_unix_ns=None):
        message = message or {}
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_id": message.get("message_id"),
            "run_id": message.get("run_id"),
            "stream_instance_id": message.get("stream_instance_id"),
            "sample_index": message.get("sample_index"),
            "edge_received_unix_ns": (
                self.clock_ns() if received_unix_ns is None
                else received_unix_ns),
            "service_metadata": self.service_metadata,
        }

    def _reject(
            self, reason, message=None, received_unix_ns=None,
            processing_started_ns=None):
        self.statistics["rejected_messages"] += 1
        response = self._base_response(message, received_unix_ns)
        response.update({
            "status": "rejected",
            "reason": str(reason),
            "edge_processing_ms": (
                None if processing_started_ns is None else
                (self.performance_ns() - processing_started_ns) / 1_000_000.0),
            "edge_published_unix_ns": self.clock_ns(),
        })
        self.publish_result(response)
        return response

    def handle_payload(
            self, payload, received_unix_ns=None,
            received_performance_ns=None):
        self.statistics["received_messages"] += 1
        if received_unix_ns is None:
            received_unix_ns = self.clock_ns()
        if received_performance_ns is None:
            received_performance_ns = self.performance_ns()
        try:
            decoded = decode_json_object(payload)
        except ProtocolError as exc:
            return self._reject(
                exc, received_unix_ns=received_unix_ns,
                processing_started_ns=received_performance_ns)
        try:
            message = validate_sensor_message(
                decoded, self.detector.raw_sensor_names,
                self.detector.schema_hash)
        except ProtocolError as exc:
            return self._reject(
                exc, decoded, received_unix_ns,
                received_performance_ns)

        stream_id = message["stream_instance_id"]
        sample_index = message["sample_index"]
        if stream_id == self.current_stream:
            cached = self.response_cache.get(message["message_id"])
            if cached is not None and cached["sample_index"] == sample_index:
                self.statistics["duplicate_redeliveries"] += 1
                response = dict(cached)
                response["duplicate_redelivery"] = True
                response["edge_received_unix_ns"] = received_unix_ns
                response["edge_processing_ms"] = (
                    (self.performance_ns() - received_performance_ns) /
                    1_000_000.0
                )
                response["edge_published_unix_ns"] = self.clock_ns()
                self.publish_result(response)
                return response
            if cached is not None:
                return self._reject(
                    "message_id was reused for a different sample", message,
                    received_unix_ns, received_performance_ns)
            if sample_index != self.expected_index:
                return self._reject(
                    f"sample sequence mismatch: expected "
                    f"{self.expected_index}, got {sample_index}", message,
                    received_unix_ns, received_performance_ns)
            expected_context = (
                message["wafer_id"], message["recipe_id"],
                message["equipment_id"], message["sequence_length"])
            if expected_context != self.stream_context:
                return self._reject(
                    "stream context changed before a new stream ID", message,
                    received_unix_ns, received_performance_ns)
        else:
            if sample_index != 0:
                return self._reject(
                    "a new stream must begin at sample_index 0", message,
                    received_unix_ns, received_performance_ns)
            if (
                    self.current_stream is not None and
                    self.expected_index != self.sequence_length):
                self.statistics["incomplete_stream_transitions"] += 1
            self.detector.start_stream(
                message["wafer_id"], message["recipe_id"],
                message["equipment_id"], stream_id)
            self.current_stream = stream_id
            self.expected_index = 0
            self.sequence_length = message["sequence_length"]
            self.stream_context = (
                message["wafer_id"], message["recipe_id"],
                message["equipment_id"], message["sequence_length"])
            self.response_cache = {}
            self.last_alarm = False

        sample = dict(zip(
            message["sensor_columns"], message["sensor_values"]))
        try:
            decision = self.detector.update(
                sample, message["sample_timestamp"])
        except (RuntimeError, TypeError, ValueError) as exc:
            return self._reject(
                f"runtime rejected sample: {exc}", message,
                received_unix_ns, received_performance_ns)
        processing_ms = (
            self.performance_ns() - received_performance_ns) / 1_000_000.0
        rising_edge = bool(decision["alarm"] and not self.last_alarm)
        response = self._base_response(message, received_unix_ns)
        response.update({
            "status": "accepted",
            "duplicate_redelivery": False,
            "edge_processing_ms": processing_ms,
            "edge_published_unix_ns": self.clock_ns(),
            "decision": {
                "alarm_ready": bool(decision["alarm_ready"]),
                "alarm": bool(decision["alarm"]),
                "rising_edge": rising_edge,
                "score": decision["score"],
                "threshold": decision["threshold"],
                "trigger_profile_id": decision["trigger_profile_id"],
                "top_evidence": decision["top_evidence"],
                "model_version": decision["model_version"],
                "model_artifact_sha256": decision["model_artifact_sha256"],
                "deployment_manifest_sha256": decision[
                    "deployment_manifest_sha256"],
            },
        })
        self.statistics["accepted_samples"] += 1
        self.expected_index += 1
        self.response_cache[message["message_id"]] = response
        self.last_alarm = bool(decision["alarm"])
        self.publish_result(response)
        if rising_edge:
            self.statistics["alarm_rising_edges"] += 1
            alarm = dict(response)
            alarm["event_type"] = "alarm_rising_edge"
            self.publish_alarm(alarm)
        return response
