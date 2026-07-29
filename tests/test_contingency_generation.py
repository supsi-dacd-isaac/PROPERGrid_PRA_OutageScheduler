from __future__ import annotations

import pytest

from pra_psa.core.contingencies import (
    apply_contingency,
    build_n1_contingencies,
    build_nk_contingencies,
    contingency_ids,
)
from pra_psa.core.models import Contingency, Outage


@pytest.mark.unit
def test_n1_catalogue_respects_service_status(fake_network) -> None:
    contingencies = build_n1_contingencies(
        fake_network,
        include=("line", "trafo", "gen"),
        only_in_service=True,
    )
    assert contingency_ids(contingencies) == [
        "N-1::line:0",
        "N-1::line:4",
        "N-1::trafo:1",
        "N-1::gen:3",
        "N-1::gen:5",
    ]


@pytest.mark.unit
def test_n1_catalogue_can_include_out_of_service_assets(fake_network) -> None:
    contingencies = build_n1_contingencies(
        fake_network,
        include=("line",),
        only_in_service=False,
    )
    assert [c.outages[0].element_index for c in contingencies] == [0, 2, 4]


@pytest.mark.unit
def test_nk_generation_and_reproducible_subsampling(fake_network) -> None:
    all_pairs = build_nk_contingencies(
        fake_network,
        k=2,
        include=("line",),
    )
    assert len(all_pairs) == 1  # only lines 0 and 4 are in service
    assert all_pairs[0].order == 2

    pool = [Outage("line", i) for i in range(5)]
    first = build_nk_contingencies(
        fake_network,
        k=2,
        candidate_pool=pool,
        max_contingencies=4,
        random_state=17,
    )
    second = build_nk_contingencies(
        fake_network,
        k=2,
        candidate_pool=pool,
        max_contingencies=4,
        random_state=17,
    )
    assert [c.contingency_id for c in first] == [c.contingency_id for c in second]


@pytest.mark.unit
def test_invalid_nk_order_is_rejected(fake_network) -> None:
    with pytest.raises(ValueError, match="k must be"):
        build_nk_contingencies(fake_network, k=0)


@pytest.mark.unit
def test_apply_contingency_copy_does_not_mutate_source(fake_network) -> None:
    contingency = Contingency((Outage("line", 0), Outage("gen", 5)))
    modified = apply_contingency(fake_network, contingency, copy_net=True)
    assert bool(modified.line.at[0, "in_service"]) is False
    assert bool(modified.gen.at[5, "in_service"]) is False
    assert bool(fake_network.line.at[0, "in_service"]) is True
    assert bool(fake_network.gen.at[5, "in_service"]) is True


@pytest.mark.unit
def test_apply_contingency_in_place(fake_network) -> None:
    returned = apply_contingency(fake_network, ("trafo", 1), copy_net=False)
    assert returned is fake_network
    assert bool(fake_network.trafo.at[1, "in_service"]) is False


@pytest.mark.unit
def test_apply_contingency_reports_bad_asset(fake_network) -> None:
    with pytest.raises(KeyError, match="index 99"):
        apply_contingency(fake_network, ("line", 99))
    with pytest.raises(ValueError, match="no table"):
        apply_contingency(fake_network, ("load", 0))
