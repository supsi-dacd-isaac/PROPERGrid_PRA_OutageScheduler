from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from scheduler_clustered.serialization import json_safe


@dataclass
class Sample:
    value: np.float64
    vector: np.ndarray


@pytest.mark.unit
def test_json_safe_converts_numpy_dataclasses_and_nonfinite_values() -> None:
    value = {
        "sample": Sample(np.float64(2.5), np.asarray([1, 2], dtype=np.int64)),
        "tuple": (np.int64(4), np.inf, np.nan),
        "set": {"a", "b"},
    }
    converted = json_safe(value)
    assert converted["sample"] == {"value": 2.5, "vector": [1, 2]}
    assert converted["tuple"] == [4, None, None]
    assert sorted(converted["set"]) == ["a", "b"]
    json.dumps(converted, allow_nan=False)
