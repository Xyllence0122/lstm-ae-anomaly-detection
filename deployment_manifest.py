# -*- coding: utf-8 -*-
"""Hash and verify deployment artifacts and historical evaluation provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def canonical_json_bytes(value):
    """Serialize a value deterministically for content-addressed identifiers."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path):
    """Hash source text consistently across LF and CRLF Git checkouts."""
    value = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(value)


def sensor_schema_payload(sensor_names):
    return {
        "schema_version": 1,
        "sensor_names": list(sensor_names),
        "value_type": "float32",
    }


def sensor_schema_hash(sensor_names):
    return sha256_bytes(canonical_json_bytes(sensor_schema_payload(sensor_names)))


def content_id(payload):
    return f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"


def _json_pointer_value(value, json_pointer):
    for key in json_pointer.strip("/").split("/"):
        if key:
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _verify_result_bindings(report, result_value, document):
    payload = document["payload"]
    manifest_id = document["manifest_id"]
    bindings = report.get("result_bindings", [])
    for binding in bindings:
        if binding.get("manifest_id") != manifest_id:
            raise ValueError("report binding refers to a different manifest")
        if "result_index" in binding:
            item = result_value[binding["result_index"]]
        else:
            item = report[binding["result_key"]]
        for key in ("model", "target_fpr", "threshold"):
            if key in binding and binding[key] != item[key]:
                raise ValueError(
                    f"report binding {key} does not match its result")
        artifact_id = binding.get("checkpoint_artifact")
        if artifact_id:
            artifact = payload["artifacts"].get(artifact_id)
            if artifact is None:
                raise ValueError(
                    f"report binding uses unknown artifact: {artifact_id}")
            if binding.get("checkpoint_sha256") != artifact["sha256"]:
                raise ValueError(
                    f"report binding hash does not match {artifact_id}")
        profile_id = binding.get("profile_id")
        if profile_id:
            profile = payload["profiles"].get(profile_id)
            if profile is None or profile["threshold"] != binding["threshold"]:
                raise ValueError(
                    f"report binding does not match profile {profile_id}")

    deployment_binding = report.get("deployment_binding")
    if deployment_binding:
        if deployment_binding.get("manifest_id") != manifest_id:
            raise ValueError(
                "deployment report binding refers to a different manifest")
        profile_id = deployment_binding["profile_id"]
        profile = payload["profiles"].get(profile_id)
        if profile is None:
            raise ValueError(f"unknown bound deployment profile: {profile_id}")
        if deployment_binding["threshold"] != profile["threshold"]:
            raise ValueError(
                f"deployment binding threshold differs from {profile_id}")
        artifact = payload["artifacts"][
            deployment_binding["checkpoint_artifact"]]
        if deployment_binding["checkpoint_sha256"] != artifact["sha256"]:
            raise ValueError("deployment binding checkpoint hash mismatch")


def _verify_git_blob(project_dir, commit, entry):
    command = ["git", "show", f"{commit}:{entry['path']}"]
    result = subprocess.run(
        command, cwd=project_dir, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"cannot read provenance blob {commit}:{entry['path']}: {detail}"
        )
    actual = sha256_bytes(result.stdout)
    if actual != entry["sha256"]:
        raise ValueError(
            f"provenance hash mismatch for {commit}:{entry['path']}: "
            f"expected {entry['sha256']}, got {actual}"
        )


def _verify_runtime_environment(project_dir, provenance):
    for entry in provenance.get("deployment_runtime_files", []):
        runtime_path = project_dir / entry["path"]
        if not runtime_path.is_file():
            raise ValueError(
                f"deployment runtime file is missing: {runtime_path}")
        if entry.get("hash_mode") != "normalized_lf_text_sha256":
            raise ValueError(
                f"unsupported runtime hash mode for {entry['path']}")
        actual = normalized_text_sha256(runtime_path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"deployment runtime hash mismatch for {entry['path']}: "
                f"expected {entry['sha256']}, got {actual}")
    environment = provenance.get("environment", {})
    expected_python = environment.get("python")
    if expected_python and platform.python_version() != expected_python:
        raise ValueError(
            f"Python version mismatch: expected {expected_python}, got "
            f"{platform.python_version()}")
    for distribution, expected in environment.get(
            "distributions", {}).items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"required distribution is missing: {distribution}") from exc
        if actual != expected:
            raise ValueError(
                f"dependency version mismatch for {distribution}: "
                f"expected {expected}, got {actual}")


def load_deployment_manifest(path, project_dir=None, *,
                             verify_artifacts=True,
                             verify_provenance=False,
                             verify_runtime=False):
    """Load a deployment manifest and fail on any requested hash mismatch."""
    path = Path(path).resolve()
    project_dir = Path(project_dir or path.parent.parent).resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if document.get("manifest_version") != 1:
        raise ValueError("unsupported deployment manifest version")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("deployment manifest payload is missing")
    expected_id = content_id(payload)
    if document.get("manifest_id") != expected_id:
        raise ValueError(
            "deployment manifest content hash mismatch: "
            f"expected {expected_id}, got {document.get('manifest_id')}"
        )

    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        recorded = sidecar.read_text(encoding="ascii").strip().split()[0]
        actual = sha256_bytes(raw)
        if recorded != actual:
            raise ValueError(
                f"deployment manifest file hash mismatch: expected "
                f"{recorded}, got {actual}"
            )

    profiles = payload.get("profiles", {})
    if not profiles:
        raise ValueError("deployment manifest contains no operating profiles")
    for profile_id, profile in profiles.items():
        profile_body = {
            key: value for key, value in profile.items()
            if key != "profile_hash"
        }
        expected_profile_hash = content_id(profile_body)
        if profile.get("profile_hash") != expected_profile_hash:
            raise ValueError(
                f"profile hash mismatch for {profile_id}: expected "
                f"{expected_profile_hash}, got {profile.get('profile_hash')}"
            )

    if verify_artifacts:
        for artifact_id, artifact in payload.get("artifacts", {}).items():
            artifact_path = project_dir / artifact["path"]
            if not artifact_path.is_file():
                raise ValueError(
                    f"deployment artifact is missing: {artifact_path}"
                )
            actual = file_sha256(artifact_path)
            if actual != artifact["sha256"]:
                raise ValueError(
                    f"artifact hash mismatch for {artifact_id}: expected "
                    f"{artifact['sha256']}, got {actual}"
                )

    provenance = payload.get("provenance", {})
    if verify_runtime or verify_provenance:
        _verify_runtime_environment(project_dir, provenance)

    if verify_provenance:
        commit = provenance.get("evaluation_source_commit")
        if not commit:
            raise ValueError("evaluation source commit is missing")
        for entry in provenance.get("git_blobs", []):
            _verify_git_blob(project_dir, commit, entry)
        for report in payload.get("reports", []):
            report_path = project_dir / report["path"]
            verify_report_digest(
                report_path, report["json_pointer"], report["sha256"])
            report_document = json.loads(
                report_path.read_text(encoding="utf-8"))
            _verify_result_bindings(
                report_document,
                _json_pointer_value(report_document, report["json_pointer"]),
                document)

    return document


def verify_report_digest(report_path, json_pointer, expected_digest):
    """Verify a metrics subtree without depending on added audit metadata."""
    value = json.loads(Path(report_path).read_text(encoding="utf-8"))
    value = _json_pointer_value(value, json_pointer)
    actual = sha256_bytes(canonical_json_bytes(value))
    if actual != expected_digest:
        raise ValueError(
            f"report result digest mismatch for {report_path}: expected "
            f"{expected_digest}, got {actual}"
        )
    return actual
