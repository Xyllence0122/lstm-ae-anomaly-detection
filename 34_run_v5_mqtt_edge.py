# -*- coding: utf-8 -*-
"""Run the locked V5 detector as an MQTT edge inference service."""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import torch

from config import OUTPUT_DIR
from v4_hashing import normalized_text_sha256
from v5_edge_runtime import DEFAULT_MANIFEST, V5MultiscaleDetector
from v5_mqtt_protocol import (
    DEFAULT_TOPIC_PREFIX,
    EdgeMessageProcessor,
    encode_json,
    topic_names,
)


DEFAULT_OUTPUT = OUTPUT_DIR / "v5" / "mqtt_edge_service_v5.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic-prefix", default=DEFAULT_TOPIC_PREFIX)
    parser.add_argument("--username")
    parser.add_argument("--password-env", default="MQTT_PASSWORD")
    parser.add_argument("--tls-ca", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queue-size", type=int, default=10000)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.queue_size < 1 or args.torch_threads < 1:
        parser.error("queue size and torch threads must be positive")
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


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    topics = topic_names(args.topic_prefix)
    detector = V5MultiscaleDetector.from_manifest(args.manifest)
    inbox = queue.Queue(maxsize=args.queue_size)
    connection_ready = False
    connection_error = None
    queue_overflow = 0
    service_started = time.time_ns()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"v5-edge-{platform.node()}-{os.getpid()}",
        protocol=mqtt.MQTTv311,
    )
    configure_security(client, args)

    def publish(topic, document):
        info = client.publish(topic, encode_json(document), qos=1, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed with rc={info.rc}")

    metadata = {
        "service": "V5 MQTT edge inference",
        "python": sys.version,
        "torch": torch.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
        "service_source_sha256": normalized_text_sha256(Path(__file__)),
        "protocol_source_sha256": normalized_text_sha256(
            Path(__file__).with_name("v5_mqtt_protocol.py")),
        "deployment_manifest_sha256": detector.manifest_sha256,
        "model_version": detector.model_version,
        "model_artifact_sha256": detector.artifact_sha256,
    }
    processor = EdgeMessageProcessor(
        detector,
        lambda value: publish(topics["result"], value),
        lambda value: publish(topics["alarm"], value),
        service_metadata=metadata,
    )

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        nonlocal connection_ready, connection_error
        if reason_code.is_failure:
            connection_error = f"broker rejected connection: {reason_code}"
            return
        result, _mid = _client.subscribe(topics["sensor"], qos=1)
        if result != mqtt.MQTT_ERR_SUCCESS:
            connection_error = f"subscribe failed with rc={result}"
            return
        connection_ready = True

    def on_message(_client, _userdata, message):
        nonlocal queue_overflow
        try:
            inbox.put_nowait((
                time.time_ns(), time.perf_counter_ns(),
                bytes(message.payload)))
        except queue.Full:
            queue_overflow += 1

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()
    deadline = time.monotonic() + 15.0
    while not connection_ready and connection_error is None:
        if time.monotonic() >= deadline:
            connection_error = "timed out waiting for MQTT connection"
            break
        time.sleep(0.05)
    if connection_error is not None:
        client.loop_stop()
        client.disconnect()
        raise RuntimeError(connection_error)

    print(json.dumps({
        "status": "ready",
        "broker": f"{args.broker}:{args.port}",
        "topics": topics,
        "model_version": detector.model_version,
        "schema_hash": detector.schema_hash,
    }, ensure_ascii=False, indent=2))
    print("Press Ctrl+C after the sensor simulator has written its report.")
    try:
        while True:
            try:
                received_unix_ns, received_performance_ns, payload = (
                    inbox.get(timeout=0.5))
            except queue.Empty:
                continue
            processor.handle_payload(
                payload, received_unix_ns, received_performance_ns)
            inbox.task_done()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        report = {
            "status": "stopped",
            "broker": f"{args.broker}:{args.port}",
            "topics": topics,
            "service_started_unix_ns": service_started,
            "service_stopped_unix_ns": time.time_ns(),
            "queue_overflow_messages": queue_overflow,
            "pending_queue_messages": inbox.qsize(),
            "processor": processor.statistics,
            "provenance": metadata,
            "interpretation": (
                "This is transport/runtime evidence, not an independent "
                "model holdout and must not be used for tuning."),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
