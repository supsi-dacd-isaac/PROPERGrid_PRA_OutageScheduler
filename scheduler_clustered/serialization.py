"""Strict-JSON serialization helpers."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping
import math

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert model outputs to strict JSON-compatible values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
