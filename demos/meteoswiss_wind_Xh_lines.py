#!/usr/bin/env python3
"""Plot the next six hours of MeteoSwiss wind forecasts on grid lines.

This is a self-contained PROPER utility. It:

1. reads a MATPOWER-style ``bus.csv`` and ``branch.csv``;
2. retains active non-transformer branches by default;
3. creates equally spaced geodesic samples along every retained branch;
4. maps every sample to the nearest MeteoSwiss local-forecast point;
5. downloads one selected wind statistic for lead hours 1--6;
6. colours the branch segments using the categorical MeteoSwiss wind legend;
7. saves the mapped data and a 2 x 3 forecast-profile figure.

The default field is ``gust`` because the requested MeteoSwiss page selects
"Gust peak (10 m above ground)". Use ``--field speed`` for hourly-mean wind
speed or one of the Q10/Q90 alternatives listed by ``--help``.

The colour classes and RGB values were read from the live MeteoSwiss legend:
https://www.meteoswiss.admin.ch/services-and-publications/applications/wind.html

The plotted data are MeteoSwiss *local point forecasts* mapped to the network.
They therefore use the same legend but do not reproduce the gridded MeteoSwiss
web-map layer pixel by pixel.

Requirements
------------
    python -m pip install numpy pandas matplotlib

Typical execution from the PROPER repository root
-------------------------------------------------
    python meteoswiss_wind_Xh_lines.py --project-root .

Conservative Q90 gust map
-------------------------
    python meteoswiss_wind_Xh_lines.py \
        --project-root . \
        --field gust-q90 \
        --samples-per-line 20
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------
STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION_ID = "ch.meteoschweiz.ogd-local-forecasting"
COLLECTION_ROOT = f"https://data.geo.admin.ch/{COLLECTION_ID}"
POINT_METADATA_URL = (
    f"{COLLECTION_ROOT}/ogd-local-forecasting_meta_point.csv"
)

WMS_BASE_URL = "https://wms.geo.admin.ch/"
BOUNDARY_LAYER = "ch.swisstopo.swissboundaries3d-land-flaeche.fill"
SWITZERLAND_BBOX = (5.70, 45.70, 10.70, 47.90)
LOCAL_TZ = ZoneInfo("Europe/Zurich")
USER_AGENT = "PROPER-MeteoSwiss-wind-6h/1.0"

BUS_ID = "bus_i"
LINE_ID = "line_id"
FROM_BUS = "fbus"
TO_BUS = "tbus"

DEFAULT_HOURS = 32
DEFAULT_SAMPLES_PER_LINE = 5
DEFAULT_MAX_POI_DISTANCE_KM = 10.0

# MeteoSwiss local-forecast parameters. Wind speeds are supplied in km/h.
FIELD_CONFIG = {
    "gust": {
        "parameter": "fu3010h1",
        "column": "wind_gust_kmh",
        "label": "Hourly maximum gust (main forecast)",
    },
    "gust-q10": {
        "parameter": "fu3q10h1",
        "column": "wind_gust_q10_kmh",
        "label": "Hourly maximum gust (Q10)",
    },
    "gust-q90": {
        "parameter": "fu3q90h1",
        "column": "wind_gust_q90_kmh",
        "label": "Hourly maximum gust (Q90)",
    },
    "speed": {
        "parameter": "fu3010h0",
        "column": "wind_speed_kmh",
        "label": "Hourly-mean wind speed (main forecast)",
    },
    "speed-q10": {
        "parameter": "fu3q10h0",
        "column": "wind_speed_q10_kmh",
        "label": "Hourly-mean wind speed (Q10)",
    },
    "speed-q90": {
        "parameter": "fu3q90h0",
        "column": "wind_speed_q90_kmh",
        "label": "Hourly-mean wind speed (Q90)",
    },
}

# Exact colour boxes exposed by the MeteoSwiss gust legend on 2026-09-15.
# Entries are ordered from the lowest to the highest class.
METEOSWISS_CLASS_UPPER_KMH = np.array(
    [12, 24, 36, 48, 60, 72, 84, 96, 108],
    dtype=float,
)
METEOSWISS_CLASS_COLOURS = (
    "#cccccc",  # 0--12
    "#59cc00",  # 12--24
    "#90cc00",  # 24--36
    "#c7cc00",  # 36--48
    "#cc9a00",  # 48--60
    "#cc2c00",  # 60--72
    "#cc000c",  # 72--84
    "#cc0043",  # 84--96
    "#cc007a",  # 96--108
    "#cc00b1",  # over 108
)
METEOSWISS_CLASS_LABELS = (
    "0 to 12 km/h",
    "12 to 24 km/h",
    "24 to 36 km/h",
    "36 to 48 km/h",
    "48 to 60 km/h",
    "60 to 72 km/h",
    "72 to 84 km/h",
    "84 to 96 km/h",
    "96 to 108 km/h",
    "over 108 km/h",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    source_name: str,
) -> None:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def normalize_bus_ids(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        invalid = values.loc[numeric.isna()].head(10).tolist()
        raise ValueError(f"{name} contains invalid bus IDs: {invalid}")
    if not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{name} contains non-integer bus IDs")
    return numeric.round().astype("int64")


def request_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    attempts: int = 4,
) -> bytes:
    """Retrieve bytes with bounded exponential retry."""

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unable to retrieve {url}: {last_error}")


def fetch_json(url: str) -> dict[str, Any] | None:
    try:
        return json.loads(request_bytes(url, timeout=90.0).decode("utf-8"))
    except RuntimeError as error:
        if "HTTP Error 404" in str(error):
            return None
        raise


def download_cached(
    url: str,
    destination: Path,
    *,
    refresh: bool = False,
    timeout: float = 180.0,
) -> Path:
    """Download a non-empty file atomically, or reuse its cache."""

    destination = Path(destination)
    if (
        not refresh
        and destination.is_file()
        and destination.stat().st_size > 0
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def discover_project_root(start: Path) -> Path:
    start = Path(start).expanduser().resolve()
    candidates = (start, *start.parents)
    relative_network = (
        Path("data") / "powersystems" / "Swiss2025" / "network_csvs"
    )
    for candidate in candidates:
        if (
            (candidate / relative_network / "bus.csv").is_file()
            and (candidate / relative_network / "branch.csv").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate data/powersystems/Swiss2025/network_csvs. "
        "Pass --project-root, or pass --bus-file and --branch-file."
    )


# ---------------------------------------------------------------------------
# Network geometry
# ---------------------------------------------------------------------------
def geodesic_sample_points(
    line_geometry: pd.DataFrame,
    samples_per_line: int,
) -> pd.DataFrame:
    """Equally spaced great-circle samples, including both endpoints."""

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
    distinct = angular_distance > 1e-12
    if distinct.any():
        omega = angular_distance[distinct, None]
        denominator = np.sin(omega)
        start_weight = (
            np.sin((1.0 - fractions[None, :]) * omega) / denominator
        )
        end_weight = np.sin(fractions[None, :] * omega) / denominator
        sample_vectors[distinct] = (
            start_weight[:, :, None] * start_vectors[distinct, None, :]
            + end_weight[:, :, None] * end_vectors[distinct, None, :]
        )
    if (~distinct).any():
        sample_vectors[~distinct] = start_vectors[~distinct, None, :]

    sample_vectors /= np.linalg.norm(sample_vectors, axis=2, keepdims=True)
    latitudes = np.degrees(np.arcsin(sample_vectors[:, :, 2]))
    longitudes = np.degrees(
        np.arctan2(sample_vectors[:, :, 1], sample_vectors[:, :, 0])
    )

    line_ids = np.repeat(
        line_geometry[LINE_ID].astype(str).to_numpy(), samples_per_line
    )
    sample_indices = np.tile(
        np.arange(samples_per_line), len(line_geometry)
    )
    sample_fractions = np.tile(fractions, len(line_geometry))
    width = max(2, len(str(samples_per_line - 1)))
    target_ids = [
        f"line:{line_id}:sample_{sample_index:0{width}d}"
        for line_id, sample_index in zip(line_ids, sample_indices)
    ]

    return pd.DataFrame(
        {
            LINE_ID: line_ids,
            "target_id": target_ids,
            "sample_index": sample_indices,
            "sample_fraction": sample_fractions,
            "lat": latitudes.reshape(-1),
            "lon": longitudes.reshape(-1),
        }
    )


def prepare_line_samples(
    bus_file: Path,
    branch_file: Path,
    *,
    samples_per_line: int,
    include_inactive: bool,
    include_transformers: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    buses = pd.read_csv(bus_file)
    branches = pd.read_csv(branch_file).reset_index(drop=True)
    require_columns(buses, [BUS_ID, "lat", "lon"], bus_file.name)
    require_columns(
        branches,
        [FROM_BUS, TO_BUS, "status", "ratio"],
        branch_file.name,
    )

    buses[BUS_ID] = normalize_bus_ids(buses[BUS_ID], BUS_ID)
    branches[FROM_BUS] = normalize_bus_ids(branches[FROM_BUS], FROM_BUS)
    branches[TO_BUS] = normalize_bus_ids(branches[TO_BUS], TO_BUS)
    buses[["lat", "lon"]] = buses[["lat", "lon"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if buses[["lat", "lon"]].isna().any().any():
        raise ValueError("bus.csv contains invalid coordinates")
    if buses[BUS_ID].duplicated().any():
        raise ValueError("bus_i must be unique")

    branches["source_row"] = np.arange(len(branches), dtype=int)
    if LINE_ID not in branches.columns:
        branches[LINE_ID] = branches["source_row"].map(
            lambda row: f"branch_{row:06d}"
        )
    branches[LINE_ID] = branches[LINE_ID].astype(str)
    if branches[LINE_ID].duplicated().any():
        raise ValueError("line_id must be unique")

    keep = pd.Series(True, index=branches.index)
    if not include_inactive:
        keep &= pd.to_numeric(branches["status"], errors="coerce").eq(1)
    if not include_transformers:
        tap_ratio = pd.to_numeric(
            branches["ratio"], errors="coerce"
        ).fillna(0.0)
        keep &= np.isclose(tap_ratio, 0.0)
    branches = branches.loc[keep].copy()
    if branches.empty:
        raise ValueError("No branches remain after network filtering")

    bus_coordinates = buses.set_index(BUS_ID)[["lat", "lon"]]
    available_buses = set(bus_coordinates.index)
    referenced_buses = set(branches[FROM_BUS]) | set(branches[TO_BUS])
    missing_buses = sorted(referenced_buses - available_buses)
    if missing_buses:
        raise ValueError(
            f"branch.csv references absent bus IDs: {missing_buses[:10]}"
        )

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
    line_samples = geodesic_sample_points(
        line_geometry,
        samples_per_line=samples_per_line,
    )
    return line_geometry, line_samples


# ---------------------------------------------------------------------------
# MeteoSwiss point matching
# ---------------------------------------------------------------------------
def load_meteoswiss_points(
    data_directory: Path,
    *,
    refresh: bool,
) -> pd.DataFrame:
    metadata_path = (
        data_directory
        / "metadata"
        / "ogd-local-forecasting_meta_point.csv"
    )
    download_cached(
        POINT_METADATA_URL,
        metadata_path,
        refresh=refresh,
    )
    points = pd.read_csv(metadata_path, sep=";", encoding="latin-1")
    columns = [
        "point_id",
        "point_type_id",
        "point_name",
        "point_height_masl",
        "point_coordinates_wgs84_lat",
        "point_coordinates_wgs84_lon",
    ]
    require_columns(points, columns, metadata_path.name)
    numeric_columns = [
        "point_id",
        "point_type_id",
        "point_height_masl",
        "point_coordinates_wgs84_lat",
        "point_coordinates_wgs84_lon",
    ]
    points[numeric_columns] = points[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
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

    keys = ["point_type_id", "point_id"]
    duplicated = points.duplicated(keys, keep=False)
    if duplicated.any():
        comparison = [
            "point_name",
            "point_height_masl",
            "point_coordinates_wgs84_lat",
            "point_coordinates_wgs84_lon",
        ]
        conflicting = (
            points.loc[duplicated]
            .groupby(keys, dropna=False)[comparison]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicting.any():
            raise ValueError(
                "Conflicting MeteoSwiss metadata for point keys: "
                f"{conflicting[conflicting].index.tolist()[:10]}"
            )
        points = points.drop_duplicates(keys, keep="first")
    return points


def unit_vectors(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    latitudes = np.radians(np.asarray(latitudes, dtype=float))
    longitudes = np.radians(np.asarray(longitudes, dtype=float))
    return np.column_stack(
        (
            np.cos(latitudes) * np.cos(longitudes),
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
        )
    )


def attach_nearest_forecast_point(
    samples: pd.DataFrame,
    points: pd.DataFrame,
    *,
    max_distance_km: float,
    point_types: Iterable[int],
) -> pd.DataFrame:
    allowed = {int(point_type) for point_type in point_types}
    candidates = points.loc[points["point_type_id"].isin(allowed)].copy()
    if candidates.empty:
        raise ValueError("No eligible MeteoSwiss forecast points remain")

    candidate_vectors = unit_vectors(
        candidates["point_coordinates_wgs84_lat"].to_numpy(float),
        candidates["point_coordinates_wgs84_lon"].to_numpy(float),
    )
    target_vectors = unit_vectors(
        samples["lat"].to_numpy(float),
        samples["lon"].to_numpy(float),
    )

    nearest_positions = np.empty(len(samples), dtype=int)
    nearest_distances = np.empty(len(samples), dtype=float)
    earth_radius_km = 6371.0088
    batch_size = 1_000
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        similarities = target_vectors[start:stop] @ candidate_vectors.T
        positions = np.argmax(similarities, axis=1)
        nearest_positions[start:stop] = positions
        best = similarities[np.arange(stop - start), positions]
        nearest_distances[start:stop] = earth_radius_km * np.arccos(
            np.clip(best, -1.0, 1.0)
        )

    nearest = candidates.iloc[nearest_positions].reset_index(drop=True)
    matches = pd.DataFrame(
        {
            "target_id": samples["target_id"].to_numpy(),
            "point_type_id": nearest["point_type_id"].to_numpy("int64"),
            "point_id": nearest["point_id"].to_numpy("int64"),
            "poi_name": nearest["point_name"].to_numpy(),
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
    return samples.merge(
        matches,
        on="target_id",
        how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------------
# Forecast discovery and retrieval
# ---------------------------------------------------------------------------
def wind_assets_from_item(
    item: dict[str, Any],
    parameter: str,
) -> dict[str, str]:
    runs: dict[str, str] = {}
    pattern = re.compile(r"\.(\d{12})\.([^.]+)\.csv$", re.IGNORECASE)
    for asset_key, asset in item.get("assets", {}).items():
        match = pattern.search(asset_key)
        if not match:
            continue
        run_token, asset_parameter = match.groups()
        if (
            asset_parameter.lower() == parameter.lower()
            and asset.get("href")
        ):
            runs[run_token] = asset["href"]
    return runs


def find_forecast_run(
    parameter: str,
    *,
    run_token: str | None,
    lookback_days: int = 3,
) -> tuple[str, str]:
    if run_token is not None:
        if not re.fullmatch(r"\d{12}", run_token):
            raise ValueError("run-token must use YYYYMMDDHHMM")
        dates = [datetime.strptime(run_token[:8], "%Y%m%d").date()]
    else:
        dates = sorted(
            {
                reference.date() - timedelta(days=offset)
                for reference in (
                    datetime.now(timezone.utc),
                    datetime.now(LOCAL_TZ),
                )
                for offset in range(lookback_days)
            },
            reverse=True,
        )

    all_runs: dict[str, str] = {}
    for production_date in dates:
        item_id = production_date.strftime("%Y%m%d") + "-ch"
        item_url = (
            f"{STAC_BASE_URL}/collections/{COLLECTION_ID}/items/{item_id}"
        )
        item = fetch_json(item_url)
        if item is None:
            continue
        runs = wind_assets_from_item(item, parameter)
        all_runs.update(runs)
        if run_token is None and runs:
            latest = max(runs)
            return latest, runs[latest]

    if run_token is not None and run_token in all_runs:
        return run_token, all_runs[run_token]
    if run_token is not None:
        raise RuntimeError(
            f"Forecast run {run_token} does not contain {parameter}"
        )
    raise RuntimeError(f"No recent MeteoSwiss run contains {parameter}")


def read_selected_forecast(
    csv_path: Path,
    parameter: str,
    selected_points: pd.DataFrame,
    *,
    run_token: str,
    hours: int,
) -> pd.DataFrame:
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
            f"Could not identify one timestamp column in {csv_path.name}"
        )
    time_column = time_columns[0]
    required = ["point_id", "point_type_id", time_column, parameter]
    require_columns(pd.DataFrame(columns=header), required, csv_path.name)

    wanted = selected_points[["point_id", "point_type_id"]].copy()
    wanted[["point_id", "point_type_id"]] = wanted[
        ["point_id", "point_type_id"]
    ].astype("int64")
    wanted = wanted.drop_duplicates()

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin-1",
        usecols=required,
        chunksize=200_000,
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
            pieces.append(selected)
    if not pieces:
        raise RuntimeError("No selected forecast points occur in the CSV file")

    forecast = pd.concat(pieces, ignore_index=True)
    forecast["valid_time_utc"] = pd.to_datetime(
        forecast[time_column].astype("string"),
        format="%Y%m%d%H%M",
        utc=True,
        errors="coerce",
    )
    forecast[parameter] = pd.to_numeric(forecast[parameter], errors="coerce")
    forecast = forecast.dropna(subset=["valid_time_utc"])
    forecast["run_time_utc"] = pd.to_datetime(
        run_token,
        format="%Y%m%d%H%M",
        utc=True,
    )
    forecast["forecast_lead_hours"] = (
        forecast["valid_time_utc"] - forecast["run_time_utc"]
    ).dt.total_seconds() / 3600.0
    forecast = forecast.loc[
        forecast["forecast_lead_hours"].gt(0)
        & forecast["forecast_lead_hours"].le(hours)
    ].copy()

    keys = ["point_id", "point_type_id", "valid_time_utc"]
    duplicated = forecast.duplicated(keys, keep=False)
    if duplicated.any():
        conflicts = (
            forecast.loc[duplicated]
            .groupby(keys, dropna=False)[parameter]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicts.any():
            raise ValueError("Conflicting duplicate rows in forecast CSV")
        forecast = forecast.drop_duplicates(keys, keep="first")

    available_hours = sorted(
        forecast["forecast_lead_hours"].dropna().unique().tolist()
    )
    expected_hours = list(range(1, hours + 1))
    if available_hours != expected_hours:
        raise RuntimeError(
            f"Expected lead hours {expected_hours}, obtained {available_hours}"
        )
    return forecast[
        keys
        + ["run_time_utc", "forecast_lead_hours", parameter]
    ]


def build_line_wind_table(
    sample_mapping: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    parameter: str,
    output_column: str,
) -> pd.DataFrame:
    forecast_times = forecast[
        ["valid_time_utc", "run_time_utc", "forecast_lead_hours"]
    ].drop_duplicates()
    time_grid = sample_mapping.merge(forecast_times, how="cross")
    values = forecast[
        ["point_id", "point_type_id", "valid_time_utc", parameter]
    ].rename(columns={parameter: output_column})
    line_wind = time_grid.merge(
        values,
        on=["point_id", "point_type_id", "valid_time_utc"],
        how="left",
        validate="many_to_one",
    )
    line_wind.loc[~line_wind["poi_match_ok"], output_column] = np.nan
    line_wind[output_column.replace("_kmh", "_ms")] = (
        line_wind[output_column] / 3.6
    )
    return line_wind


# ---------------------------------------------------------------------------
# MeteoSwiss-class plotting
# ---------------------------------------------------------------------------
def wind_class_indices(values_kmh: np.ndarray) -> np.ndarray:
    """Return MeteoSwiss class indices; 108 belongs to the 96--108 class."""

    values = np.asarray(values_kmh, dtype=float)
    return np.searchsorted(
        METEOSWISS_CLASS_UPPER_KMH,
        values,
        side="left",
    ).clip(0, len(METEOSWISS_CLASS_COLOURS) - 1)


def fetch_swiss_boundary(
    data_directory: Path,
    *,
    refresh: bool,
) -> Path:
    boundary_path = (
        data_directory / "map_cache" / "switzerland_boundary.png"
    )
    query = urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": BOUNDARY_LAYER,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(str(value) for value in SWITZERLAND_BBOX),
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


def construct_segments(
    hourly_data: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[np.ndarray] = []
    values: list[float] = []
    for _, group in hourly_data.groupby(LINE_ID, sort=False):
        group = group.sort_values("sample_index")
        line_coordinates = group[["lon", "lat"]].to_numpy(float)
        line_values = pd.to_numeric(
            group[value_column], errors="coerce"
        ).to_numpy(float)
        if len(group) < 2:
            continue
        coordinates.extend(
            np.stack(
                (line_coordinates[:-1], line_coordinates[1:]),
                axis=1,
            )
        )
        adjacent_values = np.column_stack(
            (line_values[:-1], line_values[1:])
        )
        valid_pairs = np.isfinite(adjacent_values).all(axis=1)
        segment_values = np.full(len(adjacent_values), np.nan)
        segment_values[valid_pairs] = adjacent_values[valid_pairs].mean(axis=1)
        values.extend(segment_values.tolist())
    if not coordinates:
        raise ValueError("No line segments could be constructed")
    return np.asarray(coordinates), np.asarray(values)


def plot_six_hour_profile(
    line_wind: pd.DataFrame,
    *,
    value_column: str,
    field_label: str,
    run_token: str,
    samples_per_line: int,
    boundary_path: Path | None,
    figure_directory: Path,
    show: bool,
) -> dict[str, Path]:
    times = pd.DatetimeIndex(
        pd.to_datetime(
            line_wind["valid_time_utc"], utc=True, errors="coerce"
        ).dropna().unique()
    ).sort_values()
    if len(times) != DEFAULT_HOURS:
        raise ValueError(
            f"The six-hour figure requires 6 timestamps; found {len(times)}"
        )

    background = None
    if boundary_path is not None:
        background = plt.imread(boundary_path)

    figure, axes = plt.subplots(
        4,
        int(DEFAULT_HOURS/4),
        figsize=(17.0, 9.2),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    lon_min, lat_min, lon_max, lat_max = SWITZERLAND_BBOX

    for axis, valid_time in zip(axes, times):
        axis.set_facecolor("#f7f8fa")
        if background is not None:
            axis.imshow(
                background,
                extent=(lon_min, lon_max, lat_min, lat_max),
                origin="upper",
                zorder=0,
            )

        hourly = line_wind.loc[
            line_wind["valid_time_utc"].eq(valid_time)
        ]
        segments, segment_values = construct_segments(hourly, value_column)
        finite = np.isfinite(segment_values)

        # Keep the full network visible where forecast matching failed.
        if (~finite).any():
            axis.add_collection(
                LineCollection(
                    segments[~finite],
                    colors="#808080",
                    linewidths=0.65,
                    alpha=0.32,
                    zorder=1,
                )
            )
        if finite.any():
            classes = wind_class_indices(segment_values[finite])
            colours = [METEOSWISS_CLASS_COLOURS[index] for index in classes]
            axis.add_collection(
                LineCollection(
                    segments[finite],
                    colors=colours,
                    linewidths=2.0,
                    alpha=0.96,
                    zorder=2,
                )
            )

        local_time = valid_time.tz_convert(LOCAL_TZ)
        lead_hour = int(
            hourly["forecast_lead_hours"].dropna().iloc[0]
        )
        axis.set_title(
            f"+{lead_hour} h | {local_time:%a %d %b, %H:%M %Z}",
            loc="left",
            fontsize=10,
            fontweight="semibold",
        )
        axis.set_xlim(lon_min, lon_max)
        axis.set_ylim(lat_min, lat_max)
        axis.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
        axis.grid(color="#d5d8dc", linewidth=0.4, alpha=0.35)

    for axis in axes[3:]:
        axis.set_xlabel("Longitude [°E]")
    for axis in axes[::3]:
        axis.set_ylabel("Latitude [°N]")

    # MeteoSwiss lists the legend from high to low intensity.
    legend_handles = [
        Patch(
            facecolor=colour,
            edgecolor="#666666" if index == 0 else "none",
            linewidth=0.4,
            label=label,
        )
        for index, (colour, label) in enumerate(
            zip(METEOSWISS_CLASS_COLOURS, METEOSWISS_CLASS_LABELS)
        )
    ][::-1]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
        title="MeteoSwiss wind-speed classes",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9.5,
    )
    run_time = pd.to_datetime(
        run_token, format="%Y%m%d%H%M", utc=True
    ).tz_convert(LOCAL_TZ)
    figure.suptitle(
        "Six-hour wind profile on Swiss transmission lines\n"
        f"{field_label} [km/h] — forecast issued "
        f"{run_time:%Y-%m-%d %H:%M %Z}",
        fontsize=15,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.01,
        0.008,
        "Forecast source: MeteoSwiss Open Data. Boundary: swisstopo "
        f"swissBOUNDARIES3D. {samples_per_line} geodesic samples per line; "
        "segment value = mean of adjacent samples.",
        fontsize=7.5,
        color="#566573",
    )
    # Fixed margins are more reliable than tight_layout for geographic axes:
    # their non-unit aspect ratio otherwise compresses the inter-row spacing.
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.19,
        top=0.86,
        wspace=0.08,
        hspace=0.32,
    )

    figure_directory.mkdir(parents=True, exist_ok=True)
    field_token = value_column.removesuffix("_kmh")
    stem = f"{field_token}_profile_next_{DEFAULT_HOURS}_{run_token}"
    paths = {
        "png": figure_directory / f"{stem}.png",
        "pdf": figure_directory / f"{stem}.pdf",
    }
    figure.savefig(paths["png"], dpi=220, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# End-to-end execution
# ---------------------------------------------------------------------------
def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.project_root is not None:
        project_root = arguments.project_root.expanduser().resolve()
    else:
        try:
            project_root = discover_project_root(Path.cwd())
        except FileNotFoundError:
            if arguments.bus_file is None or arguments.branch_file is None:
                raise
            project_root = Path.cwd().resolve()

    network_directory = (
        project_root
        / "data"
        / "powersystems"
        / "Swiss2025"
        / "network_csvs"
    )
    bus_file = (
        arguments.bus_file.expanduser().resolve()
        if arguments.bus_file is not None
        else network_directory / "bus.csv"
    )
    branch_file = (
        arguments.branch_file.expanduser().resolve()
        if arguments.branch_file is not None
        else network_directory / "branch.csv"
    )
    data_directory = (
        arguments.data_dir.expanduser().resolve()
        if arguments.data_dir is not None
        else project_root / "data" / "weather" / "meteoswiss" / "local_forecast"
    )
    figure_directory = (
        arguments.figure_dir.expanduser().resolve()
        if arguments.figure_dir is not None
        else project_root
        / "outputs"
        / "weather"
        / "meteoswiss"
        / "local_forecast"
    )

    field = FIELD_CONFIG[arguments.field]
    parameter = field["parameter"]
    output_column = field["column"]

    print("1/6 Reading and sampling network branches")
    line_geometry, line_samples = prepare_line_samples(
        bus_file,
        branch_file,
        samples_per_line=arguments.samples_per_line,
        include_inactive=arguments.include_inactive,
        include_transformers=arguments.include_transformers,
    )

    print("2/6 Loading MeteoSwiss point metadata")
    points = load_meteoswiss_points(
        data_directory,
        refresh=arguments.refresh,
    )
    sample_mapping = attach_nearest_forecast_point(
        line_samples,
        points,
        max_distance_km=arguments.max_poi_distance_km,
        point_types=arguments.point_types,
    )

    matched = int(sample_mapping["poi_match_ok"].sum())
    print(
        f"    Matched {matched}/{len(sample_mapping)} samples within "
        f"{arguments.max_poi_distance_km:g} km"
    )
    if matched == 0:
        raise RuntimeError("No line sample has an acceptable point match")

    print("3/6 Finding the newest forecast run")
    run_token, forecast_url = find_forecast_run(
        parameter,
        run_token=arguments.run_token,
    )
    print(f"    Forecast run: {run_token} UTC")

    raw_file = (
        data_directory
        / "raw"
        / run_token
        / Path(urlparse(forecast_url).path).name
    )
    print("4/6 Downloading or loading the requested wind parameter")
    download_cached(
        forecast_url,
        raw_file,
        refresh=arguments.refresh,
    )
    selected_points = sample_mapping.loc[
        sample_mapping["poi_match_ok"],
        ["point_id", "point_type_id"],
    ].drop_duplicates()
    forecast = read_selected_forecast(
        raw_file,
        parameter,
        selected_points,
        run_token=run_token,
        hours=DEFAULT_HOURS,
    )
    line_wind = build_line_wind_table(
        sample_mapping,
        forecast,
        parameter=parameter,
        output_column=output_column,
    )

    processed_directory = data_directory / "processed" / run_token
    processed_directory.mkdir(parents=True, exist_ok=True)
    mapping_path = processed_directory / "line_sample_to_meteoswiss_point.csv"
    data_path = (
        processed_directory
        / f"{output_column.removesuffix('_kmh')}_on_lines_next_6h.csv"
    )
    metadata_path = processed_directory / "wind_6h_run_metadata.json"
    sample_mapping.to_csv(mapping_path, index=False)
    line_wind.to_csv(data_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "forecast_run_utc": pd.to_datetime(
                    run_token, format="%Y%m%d%H%M", utc=True
                ).isoformat(),
                "field": arguments.field,
                "parameter": parameter,
                "value_column": output_column,
                "forecast_hours": DEFAULT_HOURS,
                "samples_per_line": arguments.samples_per_line,
                "number_of_retained_lines": len(line_geometry),
                "maximum_poi_distance_km": arguments.max_poi_distance_km,
                "matched_samples": matched,
                "total_samples": len(sample_mapping),
                "colour_classes_kmh": list(METEOSWISS_CLASS_LABELS),
                "colour_hex_low_to_high": list(METEOSWISS_CLASS_COLOURS),
                "forecast_source": "MeteoSwiss Open Data local forecast",
                "method_note": (
                    "The colours reproduce the MeteoSwiss categorical legend; "
                    "the values are local point forecasts mapped to line samples."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("5/6 Loading the Switzerland boundary")
    try:
        boundary_path = fetch_swiss_boundary(
            data_directory,
            refresh=arguments.refresh,
        )
    except Exception as error:
        boundary_path = None
        print(f"    Boundary unavailable; plotting without it: {error}")

    print("6/6 Plotting the six-hour line profile")
    figure_paths = plot_six_hour_profile(
        line_wind,
        value_column=output_column,
        field_label=field["label"],
        run_token=run_token,
        samples_per_line=arguments.samples_per_line,
        boundary_path=boundary_path,
        figure_directory=figure_directory,
        show=arguments.show,
    )

    outputs = {
        "mapping_csv": mapping_path,
        "line_wind_csv": data_path,
        "metadata_json": metadata_path,
        "figure_png": figure_paths["png"],
        "figure_pdf": figure_paths["pdf"],
    }
    print("\nOutputs")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot six future hourly MeteoSwiss wind maps on Swiss2025 lines "
            "using the official categorical colour legend."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--bus-file", type=Path)
    parser.add_argument("--branch-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--field",
        choices=tuple(FIELD_CONFIG),
        default="gust-q90",
        help=(
            "Forecast statistic to colour. Default: gust, matching the "
            "linked MeteoSwiss gust layer."
        ),
    )
    parser.add_argument(
        "--samples-per-line",
        type=int,
        default=DEFAULT_SAMPLES_PER_LINE,
    )
    parser.add_argument(
        "--max-poi-distance-km",
        type=float,
        default=DEFAULT_MAX_POI_DISTANCE_KM,
    )
    parser.add_argument(
        "--point-types",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1, 2, 3),
        help="1=station, 2=postal code, 3=mountain POI",
    )
    parser.add_argument(
        "--run-token",
        help="Optional fixed MeteoSwiss run in YYYYMMDDHHMM format",
    )
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
