"""
Plot Swissgrid 2025 pandapower/MATPOWER-style network on a Swiss map.

Expected local structure
------------------------
case_swissgrid2025.py
network_csvs/
    bus.csv       # columns: bus_i, lat, lon, baseKV, ...
    branch.csv    # columns: fbus, tbus, status, ...
    busName.csv   # optional: bus_i, Name, busCode, busLocation
    gen.csv       # optional: bus, status, plant_type, ...

The plotting routine mainly uses the CSV files because they contain the
geographic coordinates and branch endpoints directly. The pandapower case
file is optional and is used only to infer the default folder when convenient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _as_path(path: str | Path | None) -> Path | None:
    return None if path is None else Path(path).expanduser().resolve()


def _infer_csv_dir(
    network_csv_dir: str | Path | None = None,
    case_py_path: str | Path | None = None,
) -> Path:
    """Resolve the folder containing bus.csv and branch.csv."""
    if network_csv_dir is not None:
        csv_dir = _as_path(network_csv_dir)
    elif case_py_path is not None:
        csv_dir = _as_path(case_py_path).parent / "network_csvs"
    else:
        csv_dir = Path(__file__).resolve().parent / "network_csvs"

    if csv_dir is None or not csv_dir.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_dir}")

    required = ["bus.csv", "branch.csv"]
    missing = [name for name in required if not (csv_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files in {csv_dir}: {missing}")

    return csv_dir


def load_swissgrid_csv_network(
    network_csv_dir: str | Path | None = None,
    case_py_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load Swissgrid network CSVs produced from System_swissgrid2025.xlsx.

    Returns
    -------
    bus, branch, bus_names, gen
        DataFrames. ``bus_names`` and ``gen`` are optional and may be None.
    """
    csv_dir = _infer_csv_dir(network_csv_dir, case_py_path)

    bus = pd.read_csv(csv_dir / "bus.csv")
    branch = pd.read_csv(csv_dir / "branch.csv")

    bus_names = None
    if (csv_dir / "busName.csv").exists():
        bus_names = pd.read_csv(csv_dir / "busName.csv")
        bus = bus.merge(bus_names, on="bus_i", how="left")

    gen = None
    if (csv_dir / "gen.csv").exists():
        gen = pd.read_csv(csv_dir / "gen.csv")

    for col in ["bus_i", "lat", "lon", "baseKV"]:
        if col not in bus.columns:
            raise ValueError(f"bus.csv must contain column {col!r}")
    for col in ["fbus", "tbus"]:
        if col not in branch.columns:
            raise ValueError(f"branch.csv must contain column {col!r}")

    bus["bus_i"] = bus["bus_i"].astype(int)
    branch["fbus"] = branch["fbus"].astype(int)
    branch["tbus"] = branch["tbus"].astype(int)
    return bus, branch, bus_names, gen


def _prepare_branch_segments(
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    *,
    only_in_service: bool = True,
    voltage_rule: str = "max",
) -> pd.DataFrame:
    """Attach endpoint coordinates and voltage class to each branch."""
    if only_in_service and "status" in branch.columns:
        branch = branch.loc[branch["status"].astype(float) != 0].copy()
    else:
        branch = branch.copy()

    bus_lookup = bus.set_index("bus_i")[["lat", "lon", "baseKV"]]

    out = branch.merge(
        bus_lookup.rename(columns={"lat": "lat_from", "lon": "lon_from", "baseKV": "kv_from"}),
        left_on="fbus",
        right_index=True,
        how="left",
    ).merge(
        bus_lookup.rename(columns={"lat": "lat_to", "lon": "lon_to", "baseKV": "kv_to"}),
        left_on="tbus",
        right_index=True,
        how="left",
    )

    out = out.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to"])

    if voltage_rule == "min":
        out["voltage_kv"] = np.minimum(out["kv_from"].astype(float), out["kv_to"].astype(float))
    elif voltage_rule == "from":
        out["voltage_kv"] = out["kv_from"].astype(float)
    elif voltage_rule == "to":
        out["voltage_kv"] = out["kv_to"].astype(float)
    elif voltage_rule == "max":
        out["voltage_kv"] = np.maximum(out["kv_from"].astype(float), out["kv_to"].astype(float))
    else:
        raise ValueError("voltage_rule must be one of: 'max', 'min', 'from', 'to'")

    return out


def _plot_swiss_boundary(
    ax,
    *,
    swiss_boundary_path: str | Path | None,
    bbox: tuple[float, float, float, float],
    boundary_facecolor: str = "#f8f8f8",
    boundary_edgecolor: str = "0.45",
    boundary_alpha: float = 1.0,
) -> bool:
    """
    Plot Swiss boundary when possible.

    Priority:
    1. User-provided GeoPackage/Shapefile/GeoJSON.
    2. GeoPandas built-in Natural Earth dataset, if available in the local install.
    3. Cartopy Natural Earth, if available locally or allowed by the environment.
    4. No boundary; the function still plots the grid with a Swiss bounding box.
    """
    # 1) User-provided boundary file.
    if swiss_boundary_path is not None:
        try:
            import geopandas as gpd

            boundary = gpd.read_file(swiss_boundary_path).to_crs(epsg=4326)
            boundary.plot(
                ax=ax,
                facecolor=boundary_facecolor,
                edgecolor=boundary_edgecolor,
                linewidth=0.9,
                alpha=boundary_alpha,
                zorder=0,
            )
            return True
        except Exception as exc:
            warnings.warn(f"Could not read Swiss boundary file {swiss_boundary_path!r}: {exc}")

    # 2) Old GeoPandas Natural Earth dataset, if still available.
    try:
        import geopandas as gpd

        path = gpd.datasets.get_path("naturalearth_lowres")  # unavailable in recent GeoPandas
        world = gpd.read_file(path).to_crs(epsg=4326)
        name_col = "name" if "name" in world.columns else "NAME"
        switzerland = world.loc[world[name_col].str.lower() == "switzerland"]
        if not switzerland.empty:
            switzerland.plot(
                ax=ax,
                facecolor=boundary_facecolor,
                edgecolor=boundary_edgecolor,
                linewidth=0.9,
                alpha=boundary_alpha,
                zorder=0,
            )
            return True
    except Exception:
        pass

    # 3) Cartopy Natural Earth fallback.
    try:
        import geopandas as gpd
        import cartopy.io.shapereader as shpreader

        shp = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
        world = gpd.read_file(shp).to_crs(epsg=4326)
        possible_cols = ["ADMIN", "NAME", "NAME_LONG", "SOVEREIGNT"]
        mask = False
        for col in possible_cols:
            if col in world.columns:
                mask = mask | world[col].astype(str).str.lower().eq("switzerland")
        switzerland = world.loc[mask]
        if not switzerland.empty:
            switzerland.plot(
                ax=ax,
                facecolor=boundary_facecolor,
                edgecolor=boundary_edgecolor,
                linewidth=0.9,
                alpha=boundary_alpha,
                zorder=0,
            )
            return True
    except Exception:
        pass

    # 4) Last-resort fallback: no polygon, but keep Swiss-like extent.
    lon_min, lon_max, lat_min, lat_max = bbox
    ax.add_patch(
        plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor="0.75",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
    )
    return False


def plot_swissgrid_map(
    network_csv_dir: str | Path | None = None,
    *,
    case_py_path: str | Path | None = None,
    swiss_boundary_path: str | Path | None = None,
    save_path: str | Path | None = None,
    ax=None,
    figsize: tuple[float, float] = (9.0, 7.5),
    dpi: int = 220,
    title: str = "Swissgrid 2025 transmission network",
    show_buses: bool = True,
    show_generators: bool = True,
    show_bus_labels: bool = False,
    label_column: str = "Name",
    max_labels: int = 80,
    only_in_service: bool = True,
    voltage_rule: str = "max",
    voltage_colors: Mapping[float, str] | None = None,
    bus_color: str = "black",
    generator_color: str = "tab:green",
    branch_alpha: float = 0.85,
    boundary_margin_deg: float = 0.25,
    return_data: bool = False,
):
    """
    Plot the Swissgrid network over a Swiss map/background.

    Parameters
    ----------
    network_csv_dir:
        Folder containing ``bus.csv`` and ``branch.csv``. In your case:
        ``.../PROPER/data/powersystems/Swiss2025/network_csvs``.
    case_py_path:
        Optional path to ``case_swissgrid2025.py``. If ``network_csv_dir`` is not
        given, the function assumes the CSVs are in ``case_py_path.parent / 'network_csvs'``.
    swiss_boundary_path:
        Optional local Swiss boundary file, e.g. a GeoJSON, Shapefile, or GeoPackage.
        If omitted, the function tries local Natural Earth sources and otherwise plots
        the grid with a bounding box.
    save_path:
        Optional output path. Supports png, pdf, svg, etc.
    ax:
        Optional matplotlib axis. If omitted, a new figure and axis are created.
    show_buses, show_generators, show_bus_labels:
        Toggle network annotations.
    voltage_rule:
        How to classify branch voltage if endpoint voltages differ. One of
        ``'max'``, ``'min'``, ``'from'``, ``'to'``.
    return_data:
        If True, return ``(fig, ax, bus, branch_segments)``. Otherwise return ``(fig, ax)``.

    Returns
    -------
    (fig, ax) or (fig, ax, bus, branch_segments)
    """
    bus, branch, _, gen = load_swissgrid_csv_network(network_csv_dir, case_py_path)
    branch_segments = _prepare_branch_segments(
        bus,
        branch,
        only_in_service=only_in_service,
        voltage_rule=voltage_rule,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    valid_bus = bus.dropna(subset=["lat", "lon"]).copy()

    lon_min = float(valid_bus["lon"].min()) - boundary_margin_deg
    lon_max = float(valid_bus["lon"].max()) + boundary_margin_deg
    lat_min = float(valid_bus["lat"].min()) - boundary_margin_deg
    lat_max = float(valid_bus["lat"].max()) + boundary_margin_deg
    bbox = (lon_min, lon_max, lat_min, lat_max)

    has_boundary = _plot_swiss_boundary(
        ax,
        swiss_boundary_path=swiss_boundary_path,
        bbox=bbox,
    )

    if voltage_colors is None:
        voltage_colors = {
            380.0: "#cc3311",
            220.0: "#0077bb",
            150.0: "#ee7733",
            110.0: "#33aa55",
        }

    def width_for_kv(kv: float) -> float:
        if kv >= 380:
            return 1.6
        if kv >= 220:
            return 1.15
        if kv >= 150:
            return 0.9
        return 0.65

    # Plot branches grouped by voltage class.
    used_labels = set()
    for kv in sorted(branch_segments["voltage_kv"].dropna().unique(), reverse=True):
        group = branch_segments.loc[np.isclose(branch_segments["voltage_kv"], kv)]
        segments = [
            [(row.lon_from, row.lat_from), (row.lon_to, row.lat_to)]
            for row in group.itertuples(index=False)
        ]
        if not segments:
            continue
        color = voltage_colors.get(float(kv), "0.45")
        label = f"{int(kv)} kV" if kv not in used_labels else None
        used_labels.add(kv)
        lc = LineCollection(
            segments,
            colors=color,
            linewidths=width_for_kv(float(kv)),
            alpha=branch_alpha,
            zorder=2,
            label=label,
        )
        ax.add_collection(lc)

    # Plot buses.
    if show_buses:
        sizes = valid_bus["baseKV"].astype(float).map(lambda kv: 18 if kv >= 380 else 11 if kv >= 220 else 8)
        ax.scatter(
            valid_bus["lon"],
            valid_bus["lat"],
            s=sizes,
            c=bus_color,
            alpha=0.78,
            linewidths=0,
            zorder=3,
            label="Buses",
        )

    # Plot generator buses if gen.csv is available.
    if show_generators and gen is not None and not gen.empty and "bus" in gen.columns:
        gen_active = gen.copy()
        if "status" in gen_active.columns:
            gen_active = gen_active.loc[gen_active["status"].astype(float) != 0]
        gen_bus_ids = sorted(set(gen_active["bus"].astype(int)))
        gen_bus = valid_bus.loc[valid_bus["bus_i"].isin(gen_bus_ids)]
        if not gen_bus.empty:
            ax.scatter(
                gen_bus["lon"],
                gen_bus["lat"],
                marker="^",
                s=34,
                c=generator_color,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
                label="Generator buses",
            )

    # Optional labels. Keep limited to avoid unreadable plots.
    if show_bus_labels:
        label_data = valid_bus.copy()
        if len(label_data) > max_labels:
            label_data = label_data.sort_values("baseKV", ascending=False).head(max_labels)
            warnings.warn(f"Showing only the {max_labels} highest-voltage bus labels.")
        if label_column not in label_data.columns:
            label_column = "bus_i"
        for row in label_data.itertuples(index=False):
            label = str(getattr(row, label_column))
            ax.text(
                getattr(row, "lon") + 0.015,
                getattr(row, "lat") + 0.01,
                label,
                fontsize=6.5,
                color="0.15",
                zorder=5,
            )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    if not has_boundary:
        ax.text(
            0.99,
            0.01,
            "Swiss boundary not loaded; grid shown in lon/lat extent.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.35",
        )

    if swiss_boundary_path is not None:
        add_swiss_boundary_on_top(
            ax,
            swiss_boundary_path=swiss_boundary_path,
            edgecolor="black",
            linewidth=1.2,
            zorder=20,
        )

    fig.tight_layout()

    if save_path is not None:
        save_path = _as_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if return_data:
        return fig, ax, bus, branch_segments
    return fig, ax



def add_swiss_boundary_on_top(
    ax,
    swiss_boundary_path,
    *,
    source_crs_if_missing="EPSG:2056",
    target_crs="EPSG:4326",
    edgecolor="black",
    linewidth=1.2,
    alpha=1.0,
    zorder=20,
):
    """
    Add Swiss boundary on top of an existing lon/lat network plot.

    Parameters
    ----------
    swiss_boundary_path:
        Path to a Swiss boundary file: .shp, .geojson, .gpkg, etc.
        For swisstopo swissBOUNDARIES3D, pass the .shp file.
    source_crs_if_missing:
        CRS used if the boundary file has no CRS metadata.
        Swiss LV95 is EPSG:2056.
    """
    import geopandas as gpd

    boundary = gpd.read_file(swiss_boundary_path)

    if boundary.crs is None:
        boundary = boundary.set_crs(source_crs_if_missing)

    boundary = boundary.to_crs(target_crs)

    # Plot only the outline, not the filled polygon.
    boundary.boundary.plot(
        ax=ax,
        color=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    return ax


if __name__ == "__main__":
    # Example for the user's Windows repository layout.

    case_path = r"/data/powersystems/Swiss2025/case_swissgrid2025.py"
    csv_dir = r"/data/powersystems/Swiss2025/network_csvs"

    swiss_boundary_path = r"C:\Users\roberto.rocchetta\Documents\GitHub\PROPER\data\maps\swissBOUNDARIES3D\swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp"
    fig, ax = plot_swissgrid_map(
        network_csv_dir=csv_dir,
        case_py_path=case_path,
        swiss_boundary_path=swiss_boundary_path,
        save_path=r"/data/powersystems/Swiss2025/swissgrid2025_network_map.png",
        show_bus_labels=False,
    )
    plt.show()
