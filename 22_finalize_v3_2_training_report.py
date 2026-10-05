# -*- coding: utf-8 -*-
"""Create a terminology-corrected copy of the completed V3.2 run report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import OUTPUT_DIR


EXPERIMENT_DIR = OUTPUT_DIR / "v3" / "experiment_final_training_v3_2"
SOURCE_PATH = EXPERIMENT_DIR / "selection_report.json"
OUTPUT_PATH = EXPERIMENT_DIR / "selection_report_corrected.json"
OLD_NAME = "A: dynamic slew excursion"
NEW_NAME = "A: per-sample difference excursion"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_name(value):
    if isinstance(value, dict):
        return {
            (NEW_NAME if key == OLD_NAME else key): replace_name(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_name(item) for item in value]
    return NEW_NAME if value == OLD_NAME else value


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT_PATH}")
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    corrected = replace_name(source)
    corrected["terminology_revision"] = {
        "source_report": str(SOURCE_PATH),
        "source_report_sha256": file_sha256(SOURCE_PATH),
        "replacement": {OLD_NAME: NEW_NAME},
        "metrics_or_artifact_changed": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(corrected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Corrected report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
