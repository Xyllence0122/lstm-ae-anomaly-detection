# -*- coding: utf-8 -*-
"""Cross-platform hashing helpers for the V4 deployment package."""
from __future__ import annotations

import hashlib
from pathlib import Path


def normalized_text_bytes(path):
    """Return text bytes with LF line endings on every supported platform."""
    value = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return value.replace(b"\r", b"\n")


def normalized_text_sha256(path):
    return hashlib.sha256(normalized_text_bytes(path)).hexdigest()
