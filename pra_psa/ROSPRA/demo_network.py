"""Deterministic small system used to verify the complete PRA workflow."""

from __future__ import annotations

import numpy as np

from .network import Branch, DCNetwork


def make_six_bus_demo_network() -> DCNetwork:
    """Return a meshed six-bus case with deliberately tight branch limits."""

    branches = (
        Branch("L01", 0, 1, 0.10, 92.0, length_km=45.0),
        Branch("L12", 1, 2, 0.12, 72.0, length_km=38.0),
        Branch("L23", 2, 3, 0.10, 67.0, length_km=42.0),
        Branch("L34", 3, 4, 0.11, 68.0, length_km=36.0),
        Branch("L45", 4, 5, 0.10, 60.0, length_km=31.0),
        Branch("L50", 5, 0, 0.13, 70.0, length_km=48.0),
        Branch("L14", 1, 4, 0.18, 52.0, length_km=55.0),
        Branch("L25", 2, 5, 0.17, 48.0, length_km=50.0),
    )
    load = np.array([8.0, 48.0, 42.0, 35.0, 45.0, 37.0])
    generation = np.array([145.0, 0.0, 0.0, 70.0, 0.0, 0.0])
    capacity = np.array([175.0, 0.0, 0.0, 105.0, 0.0, 0.0])
    return DCNetwork(6, branches, load, generation, capacity)


def make_demo_load_history(n_hours: int = 24 * 90, random_state: int = 7) -> np.ndarray:
    """Generate correlated positive nodal loads solely for software demonstration."""

    if n_hours < 48:
        raise ValueError("n_hours must be at least 48.")
    network = make_six_bus_demo_network()
    rng = np.random.default_rng(random_state)
    hour = np.arange(n_hours)
    daily = 1.0 + 0.12 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
    weekly = 1.0 - 0.06 * (np.mod(hour // 24, 7) >= 5)
    common = rng.standard_t(df=6, size=n_hours) * 0.035
    idiosyncratic = rng.normal(0.0, 0.018, size=(n_hours, network.n_buses))
    multiplier = daily[:, None] * weekly[:, None] * (1.0 + common[:, None] + idiosyncratic)
    return np.maximum(network.load_mw[None, :] * multiplier, 0.0)

