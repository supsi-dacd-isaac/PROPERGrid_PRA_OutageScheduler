"""Core data models for contingency analysis and PRA/PSA.

The classes in this module are intentionally lightweight and independent of
pandapower. They provide one canonical representation for outages and
contingencies across the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_ELEMENT_ALIASES = {
    "line": "line",
    "branch": "line",
    "trafo": "trafo",
    "transformer": "trafo",
    "gen": "gen",
    "generator": "gen",
    "sgen": "sgen",
    "load": "load",
}


@dataclass(frozen=True, order=True)
class Outage:
    """A single component outage.

    Parameters
    ----------
    element_type:
        Pandapower element table name. Currently supported by the analysis
        engine are ``line``, ``trafo``, ``gen``, ``sgen`` and ``load``.
    element_index:
        Index of the element in the corresponding pandapower table.
    """

    element_type: str
    element_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_type", normalize_element_type(self.element_type))
        object.__setattr__(self, "element_index", int(self.element_index))

    def to_dict(self) -> dict[str, Any]:
        return {"element_type": self.element_type, "element_index": self.element_index}

    @classmethod
    def from_any(cls, value: Any) -> "Outage":
        """Build an outage from an Outage, dict, tuple or list."""
        if isinstance(value, Outage):
            return value
        if isinstance(value, Mapping):
            return cls(
                element_type=value.get("element_type", value.get("type")),
                element_index=value.get("element_index", value.get("index")),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            return cls(element_type=value[0], element_index=value[1])
        raise TypeError(f"Cannot convert {value!r} to Outage")

    @property
    def short_id(self) -> str:
        return f"{self.element_type}:{self.element_index}"


@dataclass(frozen=True)
class Contingency:
    """A contingency composed of one or more outages."""

    outages: tuple[Outage, ...]
    contingency_id: str | None = None
    probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outages = tuple(Outage.from_any(o) for o in self.outages)
        if len(outages) == 0:
            raise ValueError("A contingency must contain at least one outage")
        object.__setattr__(self, "outages", outages)
        if self.contingency_id is None:
            object.__setattr__(self, "contingency_id", "+".join(o.short_id for o in outages))
        if self.probability is not None and not (0.0 <= float(self.probability) <= 1.0):
            raise ValueError("Contingency probability must be in [0, 1]")

    @classmethod
    def from_any(cls, value: Any) -> "Contingency":
        """Build a contingency from the formats used in earlier prototypes.

        Accepted inputs include:
        - ``Contingency``
        - a dict with ``outages`` or with ``element_type``/``element_index``
        - a single ``Outage``
        - a tuple/list ``("line", 3)``
        - a list of outage dicts/tuples
        """
        if isinstance(value, Contingency):
            return value
        if isinstance(value, Outage):
            return cls((value,))
        if isinstance(value, Mapping):
            if "outages" in value:
                return cls(
                    outages=tuple(Outage.from_any(o) for o in value["outages"]),
                    contingency_id=value.get("contingency_id", value.get("id")),
                    probability=value.get("probability"),
                    metadata=value.get("metadata", {}),
                )
            return cls(
                outages=(Outage.from_any(value),),
                contingency_id=value.get("contingency_id", value.get("id")),
                probability=value.get("probability"),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 2 and isinstance(value[0], str):
                return cls((Outage.from_any(value),))
            return cls(tuple(Outage.from_any(v) for v in value))
        raise TypeError(f"Cannot convert {value!r} to Contingency")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contingency_id": self.contingency_id,
            "outages": [o.to_dict() for o in self.outages],
            "probability": self.probability,
            "metadata": dict(self.metadata),
        }

    @property
    def order(self) -> int:
        """Number of outaged elements, i.e. k for an N-k contingency."""
        return len(self.outages)


def normalize_element_type(element_type: str) -> str:
    if element_type is None:
        raise ValueError("element_type cannot be None")
    key = str(element_type).strip().lower()
    if key not in _ELEMENT_ALIASES:
        raise ValueError(
            f"Unsupported element type {element_type!r}. "
            f"Supported aliases are {sorted(_ELEMENT_ALIASES)}."
        )
    return _ELEMENT_ALIASES[key]


def normalize_contingencies(contingencies: Iterable[Any]) -> list[Contingency]:
    """Normalize a heterogeneous contingency list to ``Contingency`` objects."""
    return [Contingency.from_any(c) for c in contingencies]
