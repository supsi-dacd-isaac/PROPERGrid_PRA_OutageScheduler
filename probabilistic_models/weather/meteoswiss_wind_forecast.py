#!/usr/bin/env python3
"""Retrieve MeteoSwiss local wind forecasts for a MATPOWER-style network.

The workflow:

1. reads ``bus.csv`` and ``branch.csv``;
2. builds weather targets at every bus and at configurable, evenly spaced
   points along every retained branch (20 points by default);
3. maps each target to the nearest MeteoSwiss local-forecast point;
4. retrieves the newest complete set of wind forecast CSV files;
5. constructs point- and branch-level wind tables; and
6. plots a spatially varying line colour over the official Swiss boundary WMS.

Downloaded and processed weather data are written below
``data/weather/meteoswiss/local_forecast``. Maps are written separately below
``outputs/weather/meteoswiss/local_forecast``.

Only NumPy, pandas and Matplotlib are required beyond the Python standard
library. Network downloads use ``urllib`` so the script also works without
``httpx``, ``requests``, ecCodes, Docker or WSL.
"""

# %% Imports and public constants
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION_ID = "ch.meteoschweiz.ogd-local-forecasting"
COLLECTION_ROOT = f"https://data.geo.admin.ch/{COLLECTION_ID}"
POINT_METADATA_URL = (
    f"{COLLECTION_ROOT}/ogd-local-forecasting_meta_point.csv"
)
WMS_BASE_URL = "https://wms.geo.admin.ch/"
BOUNDARY_LAYER = "ch.swisstopo.swissboundaries3d-land-flaeche.fill"
LOCAL_TZ = ZoneInfo("Europe/Zurich")

# Longitude minimum, latitude minimum, longitude maximum, latitude maximum.
SWITZERLAND_BBOX = (5.70, 45.70, 10.70, 47.90)

WIND_PARAMETERS: dict[str, str] = {
    "dkl010h0": "wind_direction_deg_from",
    "fu3010h0": "wind_speed_kmh",
    "fu3010h1": "wind_gust_kmh",
    "fu3q10h0": "wind_speed_q10_kmh",
    "fu3q10h1": "wind_gust_q10_kmh",
    "fu3q90h0": "wind_speed_q90_kmh",
    "fu3q90h1": "wind_gust_q90_kmh",
}

LINE_COLOUR_VARIABLES: dict[str, str] = {
    "wind_speed_q10_ms": "Pointwise Q10 hourly-mean wind speed [m/s]",
    "wind_speed_ms": "Main (non-quantile) hourly-mean wind speed [m/s]",
    "wind_speed_q90_ms": "Pointwise Q90 hourly-mean wind speed [m/s]",
    "wind_gust_q10_ms": "Pointwise Q10 hourly-maximum gust [m/s]",
    "wind_gust_ms": "Main (non-quantile) hourly-maximum gust [m/s]",
    "wind_gust_q90_ms": "Pointwise Q90 hourly-maximum gust [m/s]",
}

DEFAULT_SAMPLES_PER_LINE = 20
DEFAULT_HORIZON_HOURS = 24
DEFAULT_LINE_COLOUR_VARIABLE = "wind_gust_q90_ms"
DEFAULT_COLOUR_MAP = "plasma"

BUS_ID = "bus_i"
LINE_ID = "line_id"
FROM_BUS = "fbus"
TO_BUS = "tbus"

USER_AGENT = "PROPER-MeteoSwiss-wind/1.0"


# %% HTTP and cache utilities
def _request(
    url: str,
    *,
    timeout: float = 120.0,
    attempts: int = 3,
    allow_not_found: bool = False,
) -> bytes | None:
    """Return URL contents with bounded retry handling."""

    last_error: Exception | None = None
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404 and allow_not_found:
                return None
            last_error = error
            if error.code < 500 or attempt == attempts:
                raise
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == attempts:
                raise

        time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError(f"Could not retrieve {url}") from last_error


def fetch_json(
    url: str,
    *,
    timeout: float = 60.0,
    allow_not_found: bool = False,
) -> dict[str, Any] | None:
    payload = _request(
        url,
        timeout=timeout,
        allow_not_found=allow_not_found,
    )
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def download_cached(
    url: str,
    destination: Path,
    *,
    refresh: bool = False,
    timeout: float = 240.0,
    attempts: int = 3,
) -> Path:
    """Download atomically and reuse a non-empty cached file."""

    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0 and not refresh:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{destination.name}.",
            suffix=".part",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)

        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            byte_count = 0

            with urlopen(request, timeout=timeout) as response:
                expected_size_text = response.headers.get("Content-Length")
                expected_size = (
                    int(expected_size_text) if expected_size_text else None
                )
                expected_digest = response.headers.get("x-amz-meta-sha256")

                with temporary_path.open("wb") as output_file:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output_file.write(block)
                        digest.update(block)
                        byte_count += len(block)

            if expected_size is not None and byte_count != expected_size:
                raise IOError(
                    f"Incomplete download for {url}: received {byte_count} "
                    f"bytes, expected {expected_size}."
                )

            if expected_digest and digest.hexdigest() != expected_digest:
                raise IOError(f"SHA-256 mismatch for {url}")

            os.replace(temporary_path, destination)
            return destination

        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError(f"Could not download {url}") from last_error


def discover_project_root(
    search_starts: Sequence[Path] | None = None,
) -> Path:
    """Find a PROPER root containing the expected Swiss2025 network files."""

    if search_starts is None:
        search_starts = [Path.cwd(), Path(__file__).resolve().parent]

    visited: set[Path] = set()
    for start in search_starts:
        resolved_start = Path(start).expanduser().resolve()
        for candidate in (resolved_start, *resolved_start.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            network_directory = (
                candidate
                / "data"
                / "powersystems"
                / "Swiss2025"
                / "network_csvs"
            )
            if (
                (network_directory / "bus.csv").is_file()
                and (network_directory / "branch.csv").is_file()
            ):
                return candidate

    raise FileNotFoundError(
        "Could not locate data/powersystems/Swiss2025/network_csvs. "
        "Run from inside the PROPER repository or pass --bus-file and "
        "--branch-file explicitly."
    )


# %% Network validation and geometry
def _require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing columns: {sorted(missing)}"
        )


def normalize_matpower_bus_ids(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert validated MATPOWER bus identifiers to integers."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        invalid = values.loc[numeric.isna()].head(10).tolist()
        raise ValueError(
            f"{column_name} contains missing or invalid IDs: {invalid}"
        )
    if not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{column_name} contains non-integer bus IDs")
    return numeric.round().astype("int64")


def haversine_km(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
) -> np.ndarray:
    """Great-circle distance in kilometres."""

    earth_radius_km = 6371.0088
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(
        np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)
    )
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def geodesic_midpoint(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle midpoint between WGS84 endpoint pairs."""

    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    lambda1 = np.radians(np.asarray(lon1, dtype=float))
    lambda2 = np.radians(np.asarray(lon2, dtype=float))
    delta_lambda = lambda2 - lambda1

    bx = np.cos(phi2) * np.cos(delta_lambda)
    by = np.cos(phi2) * np.sin(delta_lambda)
    phi_mid = np.arctan2(
        np.sin(phi1) + np.sin(phi2),
        np.sqrt((np.cos(phi1) + bx) ** 2 + by**2),
    )
    lambda_mid = lambda1 + np.arctan2(by, np.cos(phi1) + bx)

    latitude = np.degrees(phi_mid)
    longitude = (np.degrees(lambda_mid) + 540) % 360 - 180
    return latitude, longitude


def geodesic_sample_points(
    line_geometry: pd.DataFrame,
    samples_per_line: int,
) -> pd.DataFrame:
    """Create equally spaced great-circle samples, including both endpoints."""

    if isinstance(samples_per_line, bool) or int(samples_per_line) < 2:
        raise ValueError("samples_per_line must be an integer of at least 2")
    samples_per_line = int(samples_per_line)

    fractions = np.linspace(0.0, 1.0, samples_per_line)
    lat1 = np.radians(line_geometry["from_lat"].to_numpy(float))
    lon1 = np.radians(line_geometry["from_lon"].to_numpy(float))
    lat2 = np.radians(line_geometry["to_lat"].to_numpy(float))
    lon2 = np.radians(line_geometry["to_lon"].to_numpy(float))

    start_vectors = np.column_stack(
        (
            np.cos(lat1) * np.cos(lon1),
            np.cos(lat1) * np.sin(lon1),
            np.sin(lat1),
        )
    )
    end_vectors = np.column_stack(
        (
            np.cos(lat2) * np.cos(lon2),
            np.cos(lat2) * np.sin(lon2),
            np.sin(lat2),
        )
    )
    angular_distance = np.arccos(
        np.clip(np.sum(start_vectors * end_vectors, axis=1), -1.0, 1.0)
    )

    sample_vectors = np.empty(
        (len(line_geometry), samples_per_line, 3), dtype=float
    )
    distinct_endpoints = angular_distance > 1e-12

    if distinct_endpoints.any():
        omega = angular_distance[distinct_endpoints, None]
        denominator = np.sin(omega)
        start_weight = np.sin((1.0 - fractions[None, :]) * omega) / denominator
        end_weight = np.sin(fractions[None, :] * omega) / denominator
        sample_vectors[distinct_endpoints] = (
            start_weight[:, :, None]
            * start_vectors[distinct_endpoints, None, :]
            + end_weight[:, :, None]
            * end_vectors[distinct_endpoints, None, :]
        )

    if (~distinct_endpoints).any():
        sample_vectors[~distinct_endpoints] = start_vectors[
            ~distinct_endpoints, None, :
        ]

    sample_vectors /= np.linalg.norm(sample_vectors, axis=2, keepdims=True)
    sample_latitudes = np.degrees(np.arcsin(sample_vectors[:, :, 2]))
    sample_longitudes = np.degrees(
        np.arctan2(sample_vectors[:, :, 1], sample_vectors[:, :, 0])
    )

    line_ids = np.repeat(
        line_geometry[LINE_ID].astype(str).to_numpy(), samples_per_line
    )
    sample_indices = np.tile(np.arange(samples_per_line), len(line_geometry))
    sample_fractions = np.tile(fractions, len(line_geometry))
    width = max(2, len(str(samples_per_line - 1)))
    target_ids = [
        f"line:{line_id}:sample_{sample_index:0{width}d}"
        for line_id, sample_index in zip(line_ids, sample_indices)
    ]
    sample_roles = np.full(len(target_ids), "interior", dtype=object)
    sample_roles[sample_indices == 0] = "from_bus"
    sample_roles[sample_indices == samples_per_line - 1] = "to_bus"

    return pd.DataFrame(
        {
            LINE_ID: line_ids,
            "target_id": target_ids,
            "target_type": "line_sample",
            "sample_index": sample_indices,
            "sample_fraction": sample_fractions,
            "sample_role": sample_roles,
            "lat": sample_latitudes.reshape(-1),
            "lon": sample_longitudes.reshape(-1),
        }
    )


def initial_bearing_deg(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
) -> np.ndarray:
    """Initial great-circle bearing, clockwise from north."""

    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    delta_lambda = np.radians(
        np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)
    )
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = (
        np.cos(phi1) * np.sin(phi2)
        - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    )
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def prepare_network_targets(
    bus_file: Path,
    branch_file: Path,
    *,
    samples_per_line: int = DEFAULT_SAMPLES_PER_LINE,
    active_branches_only: bool = True,
    exclude_transformers: bool = True,
) -> dict[str, pd.DataFrame]:
    """Read the network and construct bus and branch-sample targets."""

    bus_file = Path(bus_file)
    branch_file = Path(branch_file)
    buses = pd.read_csv(bus_file)
    branches_all = pd.read_csv(branch_file).reset_index(drop=True)

    _require_columns(buses, [BUS_ID, "lat", "lon"], bus_file.name)
    _require_columns(
        branches_all,
        [FROM_BUS, TO_BUS, "status", "ratio"],
        branch_file.name,
    )

    buses[BUS_ID] = normalize_matpower_bus_ids(buses[BUS_ID], BUS_ID)
    for column in (FROM_BUS, TO_BUS):
        branches_all[column] = normalize_matpower_bus_ids(
            branches_all[column], column
        )

    if buses[BUS_ID].duplicated().any():
        duplicates = (
            buses.loc[buses[BUS_ID].duplicated(keep=False), BUS_ID]
            .unique()
            .tolist()
        )
        raise ValueError(f"{BUS_ID} must be unique: {duplicates[:10]}")

    buses[["lat", "lon"]] = buses[["lat", "lon"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_coordinates = (
        buses["lat"].isna()
        | buses["lon"].isna()
        | ~buses["lat"].between(-90, 90)
        | ~buses["lon"].between(-180, 180)
    )
    if invalid_coordinates.any():
        invalid = buses.loc[
            invalid_coordinates, [BUS_ID, "lat", "lon"]
        ].head(10)
        raise ValueError(f"Invalid bus coordinates:\n{invalid}")

    branches_all.insert(
        0, "source_row", np.arange(len(branches_all), dtype=int)
    )
    if LINE_ID not in branches_all.columns:
        branches_all[LINE_ID] = branches_all["source_row"].map(
            lambda row: f"branch_{row:06d}"
        )
    if branches_all[LINE_ID].isna().any() or branches_all[LINE_ID].duplicated().any():
        raise ValueError("line_id must be complete and unique")
    branches_all[LINE_ID] = branches_all[LINE_ID].astype(str)

    branch_mask = pd.Series(True, index=branches_all.index)
    if active_branches_only:
        branch_status = pd.to_numeric(
            branches_all["status"], errors="coerce"
        )
        branch_mask &= branch_status.eq(1)
    if exclude_transformers:
        # MATPOWER convention: a zero ratio denotes a non-transformer branch.
        tap_ratio = pd.to_numeric(
            branches_all["ratio"], errors="coerce"
        ).fillna(0.0)
        branch_mask &= np.isclose(tap_ratio, 0.0)

    branches = branches_all.loc[branch_mask].copy()
    available_buses = set(buses[BUS_ID])
    referenced_buses = set(branches[FROM_BUS]) | set(branches[TO_BUS])
    unknown_buses = sorted(referenced_buses - available_buses)
    if unknown_buses:
        raise ValueError(
            "Branches reference IDs absent from bus.csv: "
            f"{unknown_buses[:10]}"
        )

    bus_coordinates = buses.set_index(BUS_ID)[["lat", "lon"]]
    line_geometry = branches.copy()
    line_geometry["from_lat"] = line_geometry[FROM_BUS].map(
        bus_coordinates["lat"]
    )
    line_geometry["from_lon"] = line_geometry[FROM_BUS].map(
        bus_coordinates["lon"]
    )
    line_geometry["to_lat"] = line_geometry[TO_BUS].map(
        bus_coordinates["lat"]
    )
    line_geometry["to_lon"] = line_geometry[TO_BUS].map(
        bus_coordinates["lon"]
    )
    (
        line_geometry["mid_lat"],
        line_geometry["mid_lon"],
    ) = geodesic_midpoint(
        line_geometry["from_lat"],
        line_geometry["from_lon"],
        line_geometry["to_lat"],
        line_geometry["to_lon"],
    )
    line_geometry["endpoint_distance_km"] = haversine_km(
        line_geometry["from_lat"],
        line_geometry["from_lon"],
        line_geometry["to_lat"],
        line_geometry["to_lon"],
    )
    line_geometry["line_bearing_deg"] = initial_bearing_deg(
        line_geometry["from_lat"],
        line_geometry["from_lon"],
        line_geometry["to_lat"],
        line_geometry["to_lon"],
    )
    if "length" in line_geometry.columns:
        line_geometry["reported_length"] = pd.to_numeric(
            line_geometry["length"], errors="coerce"
        )

    bus_targets = buses[[BUS_ID, "lat", "lon"]].copy()
    bus_targets[BUS_ID] = bus_targets[BUS_ID].astype("Int64")
    bus_targets["target_id"] = "bus:" + bus_targets[BUS_ID].astype(str)
    bus_targets["target_type"] = "bus"
    bus_targets[LINE_ID] = pd.Series(
        pd.NA, index=bus_targets.index, dtype="string"
    )

    line_samples = geodesic_sample_points(
        line_geometry,
        samples_per_line=samples_per_line,
    )
    line_samples[LINE_ID] = line_samples[LINE_ID].astype("string")
    line_samples[BUS_ID] = pd.Series(
        pd.NA, index=line_samples.index, dtype="Int64"
    )
    line_samples["sample_index"] = line_samples["sample_index"].astype(
        "Int64"
    )
    line_samples["sample_fraction"] = line_samples[
        "sample_fraction"
    ].astype("Float64")
    line_samples["sample_role"] = line_samples["sample_role"].astype(
        "string"
    )

    bus_targets["sample_index"] = pd.Series(
        pd.NA, index=bus_targets.index, dtype="Int64"
    )
    bus_targets["sample_fraction"] = pd.Series(
        pd.NA, index=bus_targets.index, dtype="Float64"
    )
    bus_targets["sample_role"] = pd.Series(
        pd.NA, index=bus_targets.index, dtype="string"
    )

    target_columns = [
        "target_id",
        "target_type",
        BUS_ID,
        LINE_ID,
        "sample_index",
        "sample_fraction",
        "sample_role",
        "lat",
        "lon",
    ]
    targets = pd.concat(
        [bus_targets[target_columns], line_samples[target_columns]],
        ignore_index=True,
    )

    return {
        "buses": buses,
        "branches_all": branches_all,
        "branches": branches,
        "line_geometry": line_geometry,
        "line_samples": line_samples,
        "targets": targets,
    }


# %% MeteoSwiss point metadata and spatial matching
def load_meteoswiss_points(
    data_directory: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download and validate the MeteoSwiss point catalogue."""

    metadata_path = (
        Path(data_directory)
        / "metadata"
        / "ogd-local-forecasting_meta_point.csv"
    )
    download_cached(
        POINT_METADATA_URL,
        metadata_path,
        refresh=refresh,
    )
    points = pd.read_csv(metadata_path, sep=";", encoding="latin-1")

    required_columns = [
        "point_id",
        "point_type_id",
        "point_name",
        "point_height_masl",
        "point_coordinates_wgs84_lat",
        "point_coordinates_wgs84_lon",
    ]
    _require_columns(points, required_columns, metadata_path.name)

    for column in (
        "point_id",
        "point_type_id",
        "point_height_masl",
        "point_coordinates_wgs84_lat",
        "point_coordinates_wgs84_lon",
    ):
        points[column] = pd.to_numeric(points[column], errors="coerce")

    points = points.dropna(
        subset=[
            "point_id",
            "point_type_id",
            "point_coordinates_wgs84_lat",
            "point_coordinates_wgs84_lon",
        ]
    ).copy()
    points[["point_id", "point_type_id"]] = points[
        ["point_id", "point_type_id"]
    ].astype("int64")

    point_keys = ["point_type_id", "point_id"]
    duplicated_keys = points.duplicated(point_keys, keep=False)
    if duplicated_keys.any():
        comparison_columns = [
            "point_name",
            "point_height_masl",
            "point_coordinates_wgs84_lat",
            "point_coordinates_wgs84_lon",
        ]
        conflicting = (
            points.loc[duplicated_keys]
            .groupby(point_keys, dropna=False)[comparison_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicting.any():
            raise ValueError(
                "MeteoSwiss point metadata contains conflicting rows for "
                f"composite keys: {conflicting[conflicting].index.tolist()[:10]}"
            )
        # The live catalogue occasionally contains byte-for-byte duplicate
        # station rows. They do not represent distinct forecast points.
        points = points.drop_duplicates(point_keys, keep="first").copy()
    return points


def attach_nearest_poi(
    targets: pd.DataFrame,
    points: pd.DataFrame,
    *,
    max_distance_km: float = 10.0,
    allowed_point_types: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Attach the horizontally nearest MeteoSwiss forecast point.

    The search is performed in bounded vectorized batches. This remains exact
    on a spherical Earth while avoiding one Python loop over roughly
    ``number_of_lines * samples_per_line`` targets.
    """

    _require_columns(targets, ["target_id", "lat", "lon"], "targets")
    if targets["target_id"].duplicated().any():
        raise ValueError("target_id must be unique")

    candidates = points.copy()
    if allowed_point_types is not None:
        allowed = {int(point_type) for point_type in allowed_point_types}
        candidates = candidates.loc[
            candidates["point_type_id"].isin(allowed)
        ].copy()
    if candidates.empty:
        raise ValueError("No eligible MeteoSwiss forecast points remain")

    target_coordinates = targets[["lat", "lon"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if target_coordinates.isna().any().any():
        raise ValueError("Targets contain invalid latitude or longitude")

    def unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        latitude_rad = np.radians(latitude)
        longitude_rad = np.radians(longitude)
        return np.column_stack(
            (
                np.cos(latitude_rad) * np.cos(longitude_rad),
                np.cos(latitude_rad) * np.sin(longitude_rad),
                np.sin(latitude_rad),
            )
        )

    candidate_vectors = unit_vectors(
        candidates["point_coordinates_wgs84_lat"].to_numpy(float),
        candidates["point_coordinates_wgs84_lon"].to_numpy(float),
    )
    target_vectors = unit_vectors(
        target_coordinates["lat"].to_numpy(float),
        target_coordinates["lon"].to_numpy(float),
    )

    nearest_positions = np.empty(len(targets), dtype=int)
    nearest_distances = np.empty(len(targets), dtype=float)
    earth_radius_km = 6371.0088
    batch_size = 1_000

    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        cosine_distances = target_vectors[start:stop] @ candidate_vectors.T
        positions = np.argmax(cosine_distances, axis=1)
        nearest_positions[start:stop] = positions
        nearest_cosines = cosine_distances[
            np.arange(stop - start), positions
        ]
        nearest_distances[start:stop] = earth_radius_km * np.arccos(
            np.clip(nearest_cosines, -1.0, 1.0)
        )

    nearest = candidates.iloc[nearest_positions].reset_index(drop=True)
    matches = pd.DataFrame(
        {
            "target_id": targets["target_id"].to_numpy(),
            "point_id": nearest["point_id"].to_numpy(dtype="int64"),
            "point_type_id": nearest["point_type_id"].to_numpy(
                dtype="int64"
            ),
            "poi_name": nearest["point_name"].to_numpy(),
            "poi_type": nearest.get(
                "point_type_en", pd.Series(pd.NA, index=nearest.index)
            ).to_numpy(),
            "poi_height_masl": nearest["point_height_masl"].to_numpy(),
            "poi_lat": nearest[
                "point_coordinates_wgs84_lat"
            ].to_numpy(float),
            "poi_lon": nearest[
                "point_coordinates_wgs84_lon"
            ].to_numpy(float),
            "poi_distance_km": nearest_distances,
            "poi_match_ok": nearest_distances <= float(max_distance_km),
        }
    )

    return targets.merge(
        matches,
        on="target_id",
        how="left",
        validate="one_to_one",
    )


# %% Wind forecast discovery and retrieval
def _wind_assets_from_item(
    item: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Index wind asset URLs as run -> parameter -> URL."""

    runs: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"\.(\d{12})\.([^.]+)\.csv$", re.IGNORECASE)

    for asset_key, asset in item.get("assets", {}).items():
        match = pattern.search(asset_key)
        if not match:
            continue
        run_token, parameter = match.groups()
        parameter = parameter.lower()
        if parameter in WIND_PARAMETERS and asset.get("href"):
            runs.setdefault(run_token, {})[parameter] = asset["href"]
    return runs


def find_complete_wind_run(
    *,
    run_token: str | None = None,
    lookback_days: int = 3,
) -> tuple[str, dict[str, str]]:
    """Return a requested or the newest complete wind forecast run."""

    if run_token is not None:
        if not re.fullmatch(r"\d{12}", run_token):
            raise ValueError("run_token must use YYYYMMDDHHMM")
        candidate_dates = [datetime.strptime(run_token[:8], "%Y%m%d").date()]
    else:
        utc_today = datetime.now(timezone.utc).date()
        local_today = datetime.now(LOCAL_TZ).date()
        candidate_dates = sorted(
            {
                reference_date - timedelta(days=offset)
                for reference_date in (utc_today, local_today)
                for offset in range(lookback_days)
            },
            reverse=True,
        )

    complete_runs: dict[str, dict[str, str]] = {}
    for production_date in candidate_dates:
        item_id = production_date.strftime("%Y%m%d") + "-ch"
        item_url = (
            f"{STAC_BASE_URL}/collections/{COLLECTION_ID}/items/{item_id}"
        )
        item = fetch_json(
            item_url,
            timeout=90.0,
            allow_not_found=True,
        )
        if item is None:
            continue

        complete_on_date: dict[str, dict[str, str]] = {}
        for candidate_run, urls in _wind_assets_from_item(item).items():
            if set(urls) == set(WIND_PARAMETERS):
                complete_runs[candidate_run] = urls
                complete_on_date[candidate_run] = urls

        # Candidate dates are newest first. Once a date contains a complete
        # run, older items cannot contain a newer run.
        if run_token is None and complete_on_date:
            latest_run = max(complete_on_date)
            return latest_run, complete_on_date[latest_run]

    if run_token is not None:
        if run_token not in complete_runs:
            raise RuntimeError(
                f"Run {run_token} is absent or does not contain all wind files"
            )
        return run_token, complete_runs[run_token]

    if not complete_runs:
        raise RuntimeError("No complete wind forecast run was found")

    latest_run = max(complete_runs)
    return latest_run, complete_runs[latest_run]


def read_parameter_for_points(
    csv_path: Path,
    parameter: str,
    selected_points: pd.DataFrame,
    *,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read one all-point CSV and retain selected composite point keys."""

    header = pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin-1",
        nrows=0,
    ).columns
    time_columns = [
        column for column in header if column.lower() in {"date", "time"}
    ]
    if len(time_columns) != 1:
        raise ValueError(
            f"Could not resolve one timestamp column in {csv_path.name}: "
            f"{list(header)}"
        )
    time_column = time_columns[0]
    required_columns = [
        "point_id",
        "point_type_id",
        time_column,
        parameter,
    ]
    missing = set(required_columns) - set(header)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )

    wanted = selected_points[["point_id", "point_type_id"]].copy()
    wanted[["point_id", "point_type_id"]] = wanted[
        ["point_id", "point_type_id"]
    ].astype("int64")
    wanted = wanted.drop_duplicates()

    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin-1",
        usecols=required_columns,
        chunksize=chunksize,
    ):
        for key in ("point_id", "point_type_id"):
            chunk[key] = pd.to_numeric(chunk[key], errors="coerce")
        chunk = chunk.dropna(subset=["point_id", "point_type_id"])
        chunk[["point_id", "point_type_id"]] = chunk[
            ["point_id", "point_type_id"]
        ].astype("int64")
        selected = chunk.merge(
            wanted,
            on=["point_id", "point_type_id"],
            how="inner",
            validate="many_to_one",
        )
        if not selected.empty:
            selected_chunks.append(selected)

    if not selected_chunks:
        raise RuntimeError(
            f"No selected points were present in {csv_path.name}"
        )

    result = pd.concat(selected_chunks, ignore_index=True)
    result["valid_time_utc"] = pd.to_datetime(
        result[time_column].astype("string"),
        format="%Y%m%d%H%M",
        utc=True,
        errors="coerce",
    )
    result[parameter] = pd.to_numeric(result[parameter], errors="coerce")
    result = result.dropna(subset=["valid_time_utc"])
    keys = ["point_id", "point_type_id", "valid_time_utc"]
    duplicated_rows = result.duplicated(keys, keep=False)
    if duplicated_rows.any():
        conflicting = (
            result.loc[duplicated_rows]
            .groupby(keys, dropna=False)[parameter]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicting.any():
            raise ValueError(
                f"Conflicting duplicate forecast rows in {csv_path.name}"
            )
        result = result.drop_duplicates(keys, keep="first")
    return result[keys + [parameter]]


def retrieve_wind_forecast(
    target_mapping: pd.DataFrame,
    data_directory: Path,
    *,
    run_token: str | None = None,
    horizon_hours: int | None = DEFAULT_HORIZON_HOURS,
    refresh: bool = False,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Download all wind variables and map them back to network targets."""

    selected_points = (
        target_mapping.loc[
            target_mapping["poi_match_ok"],
            ["point_id", "point_type_id"],
        ]
        .drop_duplicates()
        .astype("int64")
    )
    if selected_points.empty:
        raise RuntimeError(
            "No network target is within the allowed POI distance"
        )

    resolved_run, wind_urls = find_complete_wind_run(run_token=run_token)
    print(f"Using MeteoSwiss wind run {resolved_run} UTC")
    raw_directory = Path(data_directory) / "raw" / resolved_run
    series: list[pd.Series] = []

    for parameter in WIND_PARAMETERS:
        url = wind_urls[parameter]
        filename = Path(urlparse(url).path).name
        cached_path = raw_directory / filename
        cache_message = (
            "Refreshing" if refresh else "Using cache for"
        ) if cached_path.is_file() and cached_path.stat().st_size > 0 else "Downloading"
        print(f"{cache_message} {parameter}")
        csv_path = download_cached(
            url,
            cached_path,
            refresh=refresh,
        )
        parameter_data = read_parameter_for_points(
            csv_path,
            parameter,
            selected_points,
        )
        indexed = parameter_data.set_index(
            ["point_id", "point_type_id", "valid_time_utc"]
        )[parameter]
        series.append(indexed)

    wind_forecast = pd.concat(series, axis=1, join="outer").reset_index()
    wind_forecast = wind_forecast.rename(columns=WIND_PARAMETERS)
    wind_forecast["run_time_utc"] = pd.to_datetime(
        resolved_run, format="%Y%m%d%H%M", utc=True
    )
    wind_forecast["forecast_lead_hours"] = (
        wind_forecast["valid_time_utc"]
        - wind_forecast["run_time_utc"]
    ).dt.total_seconds() / 3600.0

    if horizon_hours is not None:
        if isinstance(horizon_hours, bool) or int(horizon_hours) < 1:
            raise ValueError("horizon_hours must be None or a positive integer")
        horizon_hours = int(horizon_hours)
        # Hourly wind values describe the preceding interval. Lead hours
        # 1..H therefore represent H future hourly intervals after the run.
        wind_forecast = wind_forecast.loc[
            wind_forecast["forecast_lead_hours"].gt(0)
            & wind_forecast["forecast_lead_hours"].le(horizon_hours)
        ].copy()
        if wind_forecast.empty:
            raise RuntimeError(
                f"Run {resolved_run} contains no values in lead hours "
                f"1..{horizon_hours}"
            )

    kilometre_per_hour_columns = [
        column
        for column in wind_forecast.columns
        if column.endswith("_kmh")
    ]
    for column in kilometre_per_hour_columns:
        wind_forecast[column.replace("_kmh", "_ms")] = (
            wind_forecast[column] / 3.6
        )

    target_wind = target_mapping.merge(
        wind_forecast,
        on=["point_id", "point_type_id"],
        how="left",
        validate="many_to_many",
    )
    forecast_columns = [
        column
        for column in wind_forecast.columns
        if column not in {"point_id", "point_type_id"}
    ]
    target_wind.loc[
        ~target_wind["poi_match_ok"], forecast_columns
    ] = np.nan

    return resolved_run, wind_forecast, target_wind


# %% Branch exposure and Swiss map
def build_line_wind_exposure(
    line_geometry: pd.DataFrame,
    line_samples: pd.DataFrame,
    target_wind: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the sample-level table and aggregate quantiles by branch."""

    forecast_index = (
        target_wind[
            ["run_time_utc", "valid_time_utc", "forecast_lead_hours"]
        ]
        .dropna(subset=["valid_time_utc"])
        .drop_duplicates()
    )
    if forecast_index.empty:
        raise ValueError("No forecast times are available for line samples")

    sample_time_grid = line_samples.merge(forecast_index, how="cross")
    point_columns = [
        "target_id",
        "point_id",
        "point_type_id",
        "poi_distance_km",
        "poi_match_ok",
    ]
    value_columns = [
        "target_id",
        "valid_time_utc",
        "wind_direction_deg_from",
        "wind_speed_ms",
        "wind_speed_q10_ms",
        "wind_speed_q90_ms",
        "wind_gust_ms",
        "wind_gust_q10_ms",
        "wind_gust_q90_ms",
    ]
    line_target_mask = target_wind["target_id"].isin(
        line_samples["target_id"]
    )
    sample_metadata = (
        target_wind.loc[line_target_mask, point_columns]
        .drop_duplicates("target_id")
    )
    sample_values = target_wind.loc[
        line_target_mask & target_wind["valid_time_utc"].notna(),
        value_columns,
    ]
    line_point_wind = (
        sample_time_grid.merge(
            sample_metadata,
            on="target_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            sample_values,
            on=["target_id", "valid_time_utc"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            line_geometry[
                [LINE_ID, "endpoint_distance_km", "line_bearing_deg"]
            ],
            on=LINE_ID,
            how="left",
            validate="many_to_one",
        )
    )

    direction_difference = np.radians(
        line_point_wind["wind_direction_deg_from"]
        - line_point_wind["line_bearing_deg"]
    )
    line_point_wind["crosswind_factor"] = np.abs(
        np.sin(direction_difference)
    )
    line_point_wind["crosswind_gust_q90_ms"] = (
        line_point_wind["wind_gust_q90_ms"]
        * line_point_wind["crosswind_factor"]
    )
    line_point_wind["crosswind_gust_ms"] = (
        line_point_wind["wind_gust_ms"]
        * line_point_wind["crosswind_factor"]
    )
    line_point_wind["crosswind_gust_q10_ms"] = (
        line_point_wind["wind_gust_q10_ms"]
        * line_point_wind["crosswind_factor"]
    )

    valid_point = line_point_wind[["point_type_id", "point_id"]].notna().all(
        axis=1
    )
    line_point_wind["forecast_point_key"] = pd.Series(
        pd.NA, index=line_point_wind.index, dtype="string"
    )
    line_point_wind.loc[valid_point, "forecast_point_key"] = (
        line_point_wind.loc[valid_point, "point_type_id"]
        .astype("int64")
        .astype("string")
        + ":"
        + line_point_wind.loc[valid_point, "point_id"]
        .astype("int64")
        .astype("string")
    )

    line_wind_envelope = (
        line_point_wind.dropna(subset=["run_time_utc", "valid_time_utc"])
        .groupby(
            [LINE_ID, "run_time_utc", "valid_time_utc"],
            as_index=False,
        )
        .agg(
            available_samples=("wind_gust_q90_ms", "count"),
            unique_forecast_points=("forecast_point_key", "nunique"),
            mean_speed_q10_ms=("wind_speed_q10_ms", "mean"),
            mean_speed_ms=("wind_speed_ms", "mean"),
            mean_speed_q90_ms=("wind_speed_q90_ms", "mean"),
            maximum_speed_q10_ms=("wind_speed_q10_ms", "max"),
            maximum_speed_ms=("wind_speed_ms", "max"),
            maximum_speed_q90_ms=("wind_speed_q90_ms", "max"),
            mean_gust_q10_ms=("wind_gust_q10_ms", "mean"),
            mean_gust_ms=("wind_gust_ms", "mean"),
            mean_gust_q90_ms=("wind_gust_q90_ms", "mean"),
            maximum_gust_q10_ms=("wind_gust_q10_ms", "max"),
            maximum_gust_ms=("wind_gust_ms", "max"),
            maximum_gust_q90_ms=("wind_gust_q90_ms", "max"),
            maximum_crosswind_gust_q10_ms=(
                "crosswind_gust_q10_ms",
                "max",
            ),
            maximum_crosswind_gust_ms=("crosswind_gust_ms", "max"),
            maximum_crosswind_gust_q90_ms=(
                "crosswind_gust_q90_ms",
                "max",
            ),
            maximum_poi_distance_km=("poi_distance_km", "max"),
        )
    )
    return line_point_wind, line_wind_envelope


def choose_forecast_time(
    valid_times: pd.Series,
    requested_time: str | datetime | pd.Timestamp | None = None,
) -> pd.Timestamp:
    times = pd.DatetimeIndex(
        pd.to_datetime(valid_times, utc=True, errors="coerce").dropna().unique()
    ).sort_values()
    if times.empty:
        raise ValueError("No valid forecast timestamps are available")

    if requested_time is None:
        requested = pd.Timestamp.now(tz="UTC")
        future = times[times >= requested]
        return future[0] if len(future) else times[-1]

    requested = pd.Timestamp(requested_time)
    if requested.tzinfo is None:
        requested = requested.tz_localize("UTC")
    else:
        requested = requested.tz_convert("UTC")
    nearest_position = int(np.argmin(np.abs(times.asi8 - requested.value)))
    return times[nearest_position]


def fetch_swiss_boundary_image(
    data_directory: Path,
    *,
    bbox: tuple[float, float, float, float] = SWITZERLAND_BBOX,
    refresh: bool = False,
) -> Path:
    """Cache the official swissBOUNDARIES3D WMS rendering."""

    boundary_path = (
        Path(data_directory) / "map_cache" / "switzerland_boundary.png"
    )
    query = urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": BOUNDARY_LAYER,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": 1400,
            "HEIGHT": 760,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }
    )
    return download_cached(
        f"{WMS_BASE_URL}?{query}",
        boundary_path,
        refresh=refresh,
        timeout=120.0,
    )


def plot_wind_map(
    line_point_wind: pd.DataFrame,
    target_wind: pd.DataFrame,
    data_directory: Path,
    figure_directory: Path,
    *,
    requested_time: str | datetime | pd.Timestamp | None = None,
    colour_variable: str = DEFAULT_LINE_COLOUR_VARIABLE,
    colour_map: str = DEFAULT_COLOUR_MAP,
    bbox: tuple[float, float, float, float] = SWITZERLAND_BBOX,
    refresh_boundary: bool = False,
    maximum_arrows: int = 70,
    show_sample_points: bool = False,
    show: bool = False,
) -> tuple[dict[str, Path], pd.Timestamp]:
    """Plot spatially varying wind values along sampled branch segments."""

    if colour_variable not in LINE_COLOUR_VARIABLES:
        raise ValueError(
            f"Unsupported colour_variable {colour_variable!r}. Choose from "
            f"{sorted(LINE_COLOUR_VARIABLES)}"
        )
    if colour_variable not in line_point_wind.columns:
        raise ValueError(f"{colour_variable} is absent from line_point_wind")

    selected_time = choose_forecast_time(
        line_point_wind["valid_time_utc"], requested_time
    )
    sample_wind = line_point_wind.loc[
        line_point_wind["valid_time_utc"].eq(selected_time)
    ].copy()
    if sample_wind.empty:
        raise ValueError(f"No line samples are available for {selected_time}")

    segment_coordinates: list[np.ndarray] = []
    segment_values: list[float] = []
    for _, group in sample_wind.groupby(LINE_ID, sort=False):
        ordered = group.sort_values("sample_index")
        coordinates = ordered[["lon", "lat"]].to_numpy(float)
        values = pd.to_numeric(
            ordered[colour_variable], errors="coerce"
        ).to_numpy(float)
        if len(coordinates) < 2:
            continue
        segment_coordinates.extend(
            np.stack((coordinates[:-1], coordinates[1:]), axis=1)
        )
        adjacent_values = np.column_stack((values[:-1], values[1:]))
        counts = np.isfinite(adjacent_values).sum(axis=1)
        sums = np.nansum(adjacent_values, axis=1)
        segment_values.extend(
            np.divide(
                sums,
                counts,
                out=np.full(len(counts), np.nan, dtype=float),
                where=counts == 2,
            )
        )

    if not segment_coordinates:
        raise ValueError("No line segments could be constructed")
    segments = np.asarray(segment_coordinates, dtype=float)
    line_values = np.asarray(segment_values, dtype=float)
    valid_line_values = np.isfinite(line_values)

    bus_wind = target_wind.loc[
        target_wind["target_type"].eq("bus")
        & target_wind["valid_time_utc"].eq(selected_time)
        & target_wind["poi_match_ok"]
    ].copy()

    boundary_path = fetch_swiss_boundary_image(
        data_directory,
        bbox=bbox,
        refresh=refresh_boundary,
    )
    boundary_image = plt.imread(boundary_path)

    figure, axis = plt.subplots(figsize=(13.5, 7.6), dpi=150)
    axis.set_facecolor("#f7f8fa")
    axis.imshow(
        boundary_image,
        extent=(bbox[0], bbox[2], bbox[1], bbox[3]),
        origin="upper",
        zorder=0,
    )

    bus_values = pd.to_numeric(
        bus_wind.get(colour_variable, pd.Series(dtype=float)),
        errors="coerce",
    ).to_numpy(float)
    finite_values = np.concatenate(
        [line_values[valid_line_values], bus_values[np.isfinite(bus_values)]]
    )

    if finite_values.size:
        lower = float(np.nanmin(finite_values))
        upper = float(np.nanmax(finite_values))
        if np.isclose(lower, upper):
            lower = max(0.0, lower - 0.5)
            upper += 0.5
        normalizer = Normalize(vmin=lower, vmax=upper)
    else:
        normalizer = Normalize(vmin=0.0, vmax=1.0)

    if (~valid_line_values).any():
        missing_collection = LineCollection(
            segments[~valid_line_values],
            colors="#aab2bd",
            linewidths=0.8,
            alpha=0.45,
            zorder=1,
        )
        axis.add_collection(missing_collection)

    if valid_line_values.any():
        line_collection = LineCollection(
            segments[valid_line_values],
            array=line_values[valid_line_values],
            cmap=colour_map,
            norm=normalizer,
            linewidths=1.7,
            alpha=0.9,
            zorder=2,
        )
        axis.add_collection(line_collection)
        colour_source = line_collection
    else:
        colour_source = plt.cm.ScalarMappable(
            norm=normalizer, cmap=colour_map
        )

    if show_sample_points:
        valid_samples = sample_wind[colour_variable].notna()
        axis.scatter(
            sample_wind.loc[valid_samples, "lon"],
            sample_wind.loc[valid_samples, "lat"],
            c=sample_wind.loc[valid_samples, colour_variable],
            cmap=colour_map,
            norm=normalizer,
            s=5,
            linewidth=0,
            alpha=0.55,
            zorder=3,
        )

    if not bus_wind.empty:
        axis.scatter(
            bus_wind["lon"],
            bus_wind["lat"],
            c=bus_values,
            cmap=colour_map,
            norm=normalizer,
            s=18,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=3,
        )

        arrows = (
            bus_wind.dropna(
                subset=[
                    "wind_direction_deg_from",
                    "wind_speed_ms",
                    "lon",
                    "lat",
                ]
            )
            .sort_values(["point_type_id", "point_id", "target_id"])
            .drop_duplicates(["point_type_id", "point_id"])
        )
        if len(arrows) > maximum_arrows:
            positions = np.linspace(
                0, len(arrows) - 1, maximum_arrows, dtype=int
            )
            arrows = arrows.iloc[positions]

        direction_radians = np.radians(
            arrows["wind_direction_deg_from"].to_numpy(float)
        )
        # Meteorological direction specifies where the wind comes from.
        eastward = -np.sin(direction_radians)
        northward = -np.cos(direction_radians)
        axis.quiver(
            arrows["lon"],
            arrows["lat"],
            eastward,
            northward,
            angles="xy",
            scale_units="xy",
            scale=18,
            width=0.0022,
            headwidth=3.5,
            headlength=4.5,
            color="#17202a",
            alpha=0.68,
            zorder=4,
        )

    colour_bar = figure.colorbar(
        colour_source,
        ax=axis,
        fraction=0.032,
        pad=0.02,
    )
    colour_bar.set_label(
        "Adjacent-sample mean — " + LINE_COLOUR_VARIABLES[colour_variable]
    )

    axis.set_xlim(bbox[0], bbox[2])
    axis.set_ylim(bbox[1], bbox[3])
    axis.set_aspect(1 / np.cos(np.radians(np.mean([bbox[1], bbox[3]]))))
    axis.set_xlabel("Longitude [°E]")
    axis.set_ylabel("Latitude [°N]")
    axis.grid(color="#d5d8dc", linewidth=0.45, alpha=0.45)

    local_time = selected_time.tz_convert(LOCAL_TZ)
    samples_per_line = int(
        sample_wind.groupby(LINE_ID)["sample_index"].nunique().median()
    )
    axis.set_title(
        "MeteoSwiss wind forecast and Swiss transmission network\n"
        f"Valid {local_time:%Y-%m-%d %H:%M %Z} — {samples_per_line} "
        "geodesic samples per branch; colour varies by segment",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    figure.text(
        0.01,
        0.01,
        "Forecast source: MeteoSwiss Open Data. Boundary: swisstopo "
        "swissBOUNDARIES3D via geo.admin.ch WMS.",
        fontsize=7.5,
        color="#566573",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))

    figure_directory = Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)
    valid_token = selected_time.strftime("%Y%m%dT%H%MZ")
    map_stem = f"wind_map_switzerland_{colour_variable}_{valid_token}"
    map_paths = {
        "png": figure_directory / f"{map_stem}.png",
        "pdf": figure_directory / f"{map_stem}.pdf",
    }
    figure.savefig(map_paths["png"], dpi=200, bbox_inches="tight")
    figure.savefig(map_paths["pdf"], bbox_inches="tight")
    if show:
        plt.show()
    return map_paths, selected_time


# %% End-to-end pipeline
def write_outputs(
    data_directory: Path,
    frames: Mapping[str, pd.DataFrame],
    *,
    run_token: str,
    map_time: pd.Timestamp,
    max_poi_distance_km: float,
    samples_per_line: int,
    horizon_hours: int | None,
    line_colour_variable: str,
) -> dict[str, Path]:
    output_directory = Path(data_directory) / "processed" / run_token
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    for filename, dataframe in frames.items():
        path = output_directory / filename
        dataframe.to_csv(path, index=False)
        output_paths[filename] = path

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_run_utc": pd.to_datetime(
            run_token, format="%Y%m%d%H%M", utc=True
        ).isoformat(),
        "map_valid_time_utc": map_time.isoformat(),
        "maximum_poi_distance_km": max_poi_distance_km,
        "samples_per_line": samples_per_line,
        "forecast_horizon_hours": horizon_hours,
        "line_colour_variable": line_colour_variable,
        "wind_parameters": WIND_PARAMETERS,
        "forecast_source": "MeteoSwiss Open Data",
        "forecast_product_type": (
            "main non-quantile forecasts with pointwise Q10 and Q90 marginal "
            "quantiles; no ensemble-member scenarios or full PDF"
        ),
        "boundary_source": "swisstopo swissBOUNDARIES3D",
        "methodological_note": (
            "Branch summary maxima are maxima of pointwise quantiles over "
            "sampled locations. They are not quantiles of the spatial "
            "maximum along a branch because spatial dependence is absent."
        ),
    }
    metadata_path = output_directory / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_paths[metadata_path.name] = metadata_path
    return output_paths


def run_pipeline(
    bus_file: Path,
    branch_file: Path,
    data_directory: Path,
    figure_directory: Path,
    *,
    samples_per_line: int = DEFAULT_SAMPLES_PER_LINE,
    horizon_hours: int | None = DEFAULT_HORIZON_HOURS,
    max_poi_distance_km: float = 10.0,
    allowed_point_types: Iterable[int] = (1, 2, 3),
    active_branches_only: bool = True,
    exclude_transformers: bool = True,
    run_token: str | None = None,
    map_valid_time: str | datetime | pd.Timestamp | None = None,
    line_colour_variable: str = DEFAULT_LINE_COLOUR_VARIABLE,
    colour_map: str = DEFAULT_COLOUR_MAP,
    refresh: bool = False,
    show_sample_points: bool = False,
    show_map: bool = False,
) -> dict[str, Any]:
    """Execute the complete network-to-wind workflow."""

    data_directory = Path(data_directory)
    figure_directory = Path(figure_directory)
    network = prepare_network_targets(
        bus_file,
        branch_file,
        samples_per_line=samples_per_line,
        active_branches_only=active_branches_only,
        exclude_transformers=exclude_transformers,
    )
    points = load_meteoswiss_points(
        data_directory,
        refresh=refresh,
    )
    target_mapping = attach_nearest_poi(
        network["targets"],
        points,
        max_distance_km=max_poi_distance_km,
        allowed_point_types=allowed_point_types,
    )
    resolved_run, wind_forecast, target_wind = retrieve_wind_forecast(
        target_mapping,
        data_directory,
        run_token=run_token,
        horizon_hours=horizon_hours,
        refresh=refresh,
    )
    line_point_wind, line_wind_envelope = build_line_wind_exposure(
        network["line_geometry"], network["line_samples"], target_wind
    )
    map_paths, selected_map_time = plot_wind_map(
        line_point_wind,
        target_wind,
        data_directory,
        figure_directory,
        requested_time=map_valid_time,
        colour_variable=line_colour_variable,
        colour_map=colour_map,
        refresh_boundary=refresh,
        show_sample_points=show_sample_points,
        show=show_map,
    )
    if not show_map:
        plt.close("all")

    frames = {
        "target_to_meteoswiss_point.csv": target_mapping,
        "wind_forecast_at_selected_points.csv": wind_forecast,
        "wind_forecast_at_targets.csv": target_wind,
        "wind_forecast_at_line_samples.csv": line_point_wind,
        "line_wind_quantile_summary.csv": line_wind_envelope,
        "line_geometry.csv": network["line_geometry"],
        "line_sample_geometry.csv": network["line_samples"],
    }
    output_paths = write_outputs(
        data_directory,
        frames,
        run_token=resolved_run,
        map_time=selected_map_time,
        max_poi_distance_km=max_poi_distance_km,
        samples_per_line=samples_per_line,
        horizon_hours=horizon_hours,
        line_colour_variable=line_colour_variable,
    )
    for file_format, map_path in map_paths.items():
        output_paths[f"wind_map_{file_format}"] = map_path

    return {
        **network,
        "meteoswiss_points": points,
        "target_mapping": target_mapping,
        "wind_forecast": wind_forecast,
        "target_wind": target_wind,
        "line_point_wind": line_point_wind,
        "line_wind_envelope": line_wind_envelope,
        "run_token": resolved_run,
        "map_valid_time": selected_map_time,
        "map_path": map_paths["png"],
        "map_paths": map_paths,
        "data_directory": data_directory,
        "figure_directory": figure_directory,
        "output_paths": output_paths,
    }


# %% Command-line interface
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map MeteoSwiss local wind forecasts to Swiss2025 buses and "
            "transmission branches."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--bus-file", type=Path)
    parser.add_argument("--branch-file", type=Path)
    parser.add_argument(
        "--data-dir",
        "--output-dir",
        dest="data_dir",
        type=Path,
        help=(
            "Weather-data root (legacy alias: --output-dir). Default: "
            "data/weather/meteoswiss/local_forecast"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help=(
            "Map-output root. Default: "
            "outputs/weather/meteoswiss/local_forecast"
        ),
    )
    parser.add_argument(
        "--samples-per-line",
        type=int,
        default=DEFAULT_SAMPLES_PER_LINE,
        help="Total geodesic samples per branch, including both endpoints",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=DEFAULT_HORIZON_HOURS,
        help="Future lead hours retained; use 0 for the complete forecast",
    )
    parser.add_argument("--max-poi-distance-km", type=float, default=10.0)
    parser.add_argument(
        "--point-types",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1, 2, 3),
        help="Allowed MeteoSwiss point types (1 station, 2 postcode, 3 mountain)",
    )
    parser.add_argument(
        "--run-token",
        help="Optional fixed forecast run in YYYYMMDDHHMM format",
    )
    parser.add_argument(
        "--map-valid-time",
        help="Optional ISO timestamp; the nearest available time is plotted",
    )
    parser.add_argument(
        "--line-colour-variable",
        choices=tuple(LINE_COLOUR_VARIABLES),
        default=DEFAULT_LINE_COLOUR_VARIABLE,
        help="Wind statistic used to colour each sampled line segment",
    )
    parser.add_argument(
        "--colour-map",
        default=DEFAULT_COLOUR_MAP,
        help="Matplotlib sequential colour map, for example plasma or viridis",
    )
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload metadata, forecast CSVs and the boundary image",
    )
    parser.add_argument("--show-sample-points", action="store_true")
    parser.add_argument("--show-map", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    if arguments.project_root is not None:
        project_root = arguments.project_root.expanduser().resolve()
    else:
        try:
            starts = [Path.cwd()]
            if arguments.bus_file is not None:
                starts.append(arguments.bus_file.parent)
            if arguments.branch_file is not None:
                starts.append(arguments.branch_file.parent)
            project_root = discover_project_root(starts)
        except FileNotFoundError:
            if arguments.bus_file is None or arguments.branch_file is None:
                raise
            project_root = Path.cwd().resolve()

    if arguments.bus_file is None or arguments.branch_file is None:
        default_network_directory = (
            project_root
            / "data"
            / "powersystems"
            / "Swiss2025"
            / "network_csvs"
        )
        bus_file = arguments.bus_file or default_network_directory / "bus.csv"
        branch_file = (
            arguments.branch_file or default_network_directory / "branch.csv"
        )
    else:
        bus_file = arguments.bus_file
        branch_file = arguments.branch_file

    data_directory = arguments.data_dir or (
        project_root / "data" / "weather" / "meteoswiss" / "local_forecast"
    )
    figure_directory = arguments.figure_dir or (
        project_root
        / "outputs"
        / "weather"
        / "meteoswiss"
        / "local_forecast"
    )
    if arguments.horizon_hours < 0:
        raise ValueError("--horizon-hours must be non-negative")
    horizon_hours = (
        None if arguments.horizon_hours == 0 else arguments.horizon_hours
    )

    result = run_pipeline(
        bus_file,
        branch_file,
        data_directory,
        figure_directory,
        samples_per_line=arguments.samples_per_line,
        horizon_hours=horizon_hours,
        max_poi_distance_km=arguments.max_poi_distance_km,
        allowed_point_types=arguments.point_types,
        active_branches_only=not arguments.include_inactive,
        exclude_transformers=not arguments.include_transformers,
        run_token=arguments.run_token,
        map_valid_time=arguments.map_valid_time,
        line_colour_variable=arguments.line_colour_variable,
        colour_map=arguments.colour_map,
        refresh=arguments.refresh,
        show_sample_points=arguments.show_sample_points,
        show_map=arguments.show_map,
    )

    matched = int(result["target_mapping"]["poi_match_ok"].sum())
    total = len(result["target_mapping"])
    matched_line_samples = result["target_mapping"].loc[
        result["target_mapping"]["target_type"].eq("line_sample")
        & result["target_mapping"]["poi_match_ok"]
    ]
    unique_line_forecast_points = matched_line_samples[
        ["point_type_id", "point_id"]
    ].drop_duplicates()
    print(f"Forecast run: {result['run_token']}")
    print(f"Matched weather targets: {matched}/{total}")
    print(f"Line samples: {arguments.samples_per_line} per retained branch")
    print(
        "Unique MeteoSwiss points used by line samples: "
        f"{len(unique_line_forecast_points)}"
    )
    print(f"Forecast horizon: {horizon_hours or 'complete product'}")
    print(f"Map valid time: {result['map_valid_time']}")
    print(f"Weather data: {Path(data_directory).resolve()}")
    print(f"Maps: {Path(figure_directory).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
