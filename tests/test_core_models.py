from __future__ import annotations

import pytest

from pra_psa.core.models import (
    Contingency,
    Outage,
    normalize_contingencies,
    normalize_element_type,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("line", "line"),
        ("branch", "line"),
        ("transformer", "trafo"),
        ("generator", "gen"),
        (" SGEN ", "sgen"),
    ],
)
def test_element_type_aliases(raw: str, expected: str) -> None:
    assert normalize_element_type(raw) == expected


@pytest.mark.unit
def test_unknown_element_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported element type"):
        normalize_element_type("bus")


@pytest.mark.unit
def test_outage_normalisation_and_serialisation() -> None:
    outage = Outage.from_any({"type": "transformer", "index": "7"})
    assert outage == Outage("trafo", 7)
    assert outage.short_id == "trafo:7"
    assert outage.to_dict() == {"element_type": "trafo", "element_index": 7}


@pytest.mark.unit
def test_contingency_accepts_legacy_and_canonical_inputs() -> None:
    contingency = Contingency.from_any(
        {
            "id": "double",
            "outages": [("line", 1), {"type": "generator", "index": 2}],
            "probability": 0.02,
            "metadata": {"cause": "common"},
        }
    )
    assert contingency.contingency_id == "double"
    assert contingency.order == 2
    assert contingency.outages == (Outage("line", 1), Outage("gen", 2))
    assert contingency.to_dict()["metadata"] == {"cause": "common"}


@pytest.mark.unit
def test_default_contingency_identifier_is_stable() -> None:
    contingency = Contingency((Outage("line", 4), Outage("trafo", 1)))
    assert contingency.contingency_id == "line:4+trafo:1"


@pytest.mark.unit
def test_contingency_validation() -> None:
    with pytest.raises(ValueError, match="at least one outage"):
        Contingency(())
    with pytest.raises(ValueError, match="probability"):
        Contingency((Outage("line", 0),), probability=1.01)


@pytest.mark.unit
def test_heterogeneous_contingency_collection_is_normalised() -> None:
    result = normalize_contingencies(
        [
            Outage("line", 0),
            ("trafo", 1),
            {"outages": [("gen", 2)], "contingency_id": "g2"},
        ]
    )
    assert [item.contingency_id for item in result] == ["line:0", "trafo:1", "g2"]
