"""
Plot the Swissgrid 2025 pandapower/MATPOWER-style network on a Swiss map.

Expected repository layout
--------------------------
PROPER/
    data/
        powersystems/
            Swiss2025/
                case_swissgrid2025.py
                network_csvs/
                    bus.csv
                    branch.csv
                    busName.csv   # optional
                    gen.csv       # optional

The script uses the CSV files for plotting because they contain the bus
coordinates and branch endpoints directly. The pandapower case file is not
required for the plot, but its path can be passed to infer the CSV folder.

Swiss map handling
------------------
No Swiss shapefile is required by default. If Cartopy is installed, the script
uses Natural Earth country boundaries and draws Switzerland behind and on top
of the network. Cartopy downloads and caches the Natural Earth dataset the
first time it is used.

Optional dependencies for automatic Swiss boundary:
    pip install cartopy shapely

Optional dependency for local shapefile/GeoJSON support:
    pip install geopandas pyogrio shapely
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------------------------------------------------------------
# Default Windows paths for the PROPER repository layout.
# Change only these two constants if your local folder is different.
# -----------------------------------------------------------------------------
DEFAULT_SYSTEM_DIR = Path(r"/data/powersystems/Swiss2025")
DEFAULT_CASE_PATH = DEFAULT_SYSTEM_DIR / "case_swissgrid2025.py"
DEFAULT_CSV_DIR = DEFAULT_SYSTEM_DIR / "network_csvs"
DEFAULT_OUTPUT_PATH = DEFAULT_SYSTEM_DIR / "swissgrid2025_network_map.png"


# -----------------------------------------------------------------------------
# Path and data loading utilities
# -----------------------------------------------------------------------------
def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


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
        csv_dir = DEFAULT_CSV_DIR

    if csv_dir is None or not csv_dir.exists():
        raise FileNotFoundError(
            f"CSV folder not found: {csv_dir}\n"
            "Expected a folder containing at least bus.csv and branch.csv."
        )

    missing = [name for name in ("bus.csv", "branch.csv") if not (csv_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files in {csv_dir}: {missing}")

    return csv_dir


def load_swissgrid_csv_network(
    network_csv_dir: str | Path | None = None,
    case_py_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load the Swissgrid 2025 network CSV files.

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
        if "bus_i" in bus_names.columns:
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

    bus = bus.copy()
    branch = branch.copy()

    bus["bus_i"] = bus["bus_i"].astype(int)
    bus["lat"] = pd.to_numeric(bus["lat"], errors="coerce")
    bus["lon"] = pd.to_numeric(bus["lon"], errors="coerce")
    bus["baseKV"] = pd.to_numeric(bus["baseKV"], errors="coerce")

    branch["fbus"] = branch["fbus"].astype(int)
    branch["tbus"] = branch["tbus"].astype(int)

    return bus, branch, bus_names, gen


# -----------------------------------------------------------------------------
# Swiss boundary helpers
# -----------------------------------------------------------------------------
def _iter_polygon_exteriors(geometry):
    """Yield exterior x/y arrays from a Polygon or MultiPolygon geometry."""
    if geometry is None:
        return

    geom_type = getattr(geometry, "geom_type", None)

    if geom_type == "Polygon":
        x, y = geometry.exterior.xy
        yield x, y
    elif geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            x, y = poly.exterior.xy
            yield x, y
    elif geom_type in {"LineString", "LinearRing"}:
        x, y = geometry.xy
        yield x, y
    elif geom_type == "MultiLineString":
        for line in geometry.geoms:
            x, y = line.xy
            yield x, y


def _find_switzerland_naturalearth_geometry(resolution: str = "10m"):
    """
    Return the Switzerland geometry from Natural Earth using Cartopy.

    Requires:
        pip install cartopy shapely
    """
    import cartopy.io.shapereader as shpreader

    shp_path = shpreader.natural_earth(
        resolution=resolution,
        category="cultural",
        name="admin_0_countries",
    )

    reader = shpreader.Reader(shp_path)

    for record in reader.records():
        attrs = record.attributes
        candidates = {
            str(attrs.get("ADMIN", "")),
            str(attrs.get("NAME", "")),
            str(attrs.get("NAME_LONG", "")),
            str(attrs.get("SOVEREIGNT", "")),
            str(attrs.get("ISO_A3", "")),
            str(attrs.get("ADM0_A3", "")),
        }
        if "Switzerland" in candidates or "CHE" in candidates:
            return record.geometry

    raise RuntimeError("Could not find Switzerland in the Natural Earth countries dataset.")


def add_swiss_map_background(
    ax,
    *,
    swiss_boundary_path: str | Path | None = None,
    naturalearth_resolution: str = "10m",
    facecolor: str = "#f7f7f7",
    edgecolor: str = "0.65",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    zorder: int = 0,
) -> bool:
    """
    Add Switzerland as a light background polygon.

    Priority:
    1. Local boundary file, if ``swiss_boundary_path`` is supplied.
    2. Natural Earth via Cartopy, requiring no local Swiss shapefile.

    Returns
    -------
    bool
        True if a Swiss boundary was plotted, False otherwise.
    """
    # Local shapefile/GeoJSON/GeoPackage path, optional.
    if swiss_boundary_path is not None:
        swiss_boundary_path = _as_path(swiss_boundary_path)
        if swiss_boundary_path is None or not swiss_boundary_path.exists():
            raise FileNotFoundError(
                f"Swiss boundary file not found:\n{swiss_boundary_path}\n\n"
                "Either pass a real .shp/.geojson/.gpkg path, or omit "
                "swiss_boundary_path to use the automatic Natural Earth fallback."
            )
        try:
            import geopandas as gpd

            boundary = gpd.read_file(swiss_boundary_path)
            if boundary.crs is None:
                # Most Swiss federal geodata are LV95. Adjust if your file differs.
                boundary = boundary.set_crs("EPSG:2056")
            boundary = boundary.to_crs("EPSG:4326")
            boundary.plot(
                ax=ax,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
            return True
        except Exception as exc:
            warnings.warn(f"Could not plot local Swiss boundary file: {exc}")
            return False

    # Automatic Natural Earth fallback.
    try:
        geom = _find_switzerland_naturalearth_geometry(naturalearth_resolution)
        for x, y in _iter_polygon_exteriors(geom):
            ax.fill(
                x,
                y,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        return True
    except Exception as exc:
        warnings.warn(
            "Could not load the automatic Switzerland boundary from Natural Earth. "
            "Install cartopy with `pip install cartopy shapely`, or pass a local boundary file. "
            f"Original error: {exc}"
        )
        return False


def add_swiss_boundary_on_top(
    ax,
    *,
    swiss_boundary_path: str | Path | None = None,
    naturalearth_resolution: str = "10m",
    edgecolor: str = "black",
    linewidth: float = 1.4,
    alpha: float = 1.0,
    zorder: int = 30,
) -> bool:
    """
    Draw the Switzerland boundary on top of the network.

    No local shapefile is needed when Cartopy is installed.
    """
    if swiss_boundary_path is not None:
        swiss_boundary_path = _as_path(swiss_boundary_path)
        if swiss_boundary_path is None or not swiss_boundary_path.exists():
            raise FileNotFoundError(
                f"Swiss boundary file not found:\n{swiss_boundary_path}\n\n"
                "Either pass a real .shp/.geojson/.gpkg path, or omit "
                "swiss_boundary_path to use Natural Earth."
            )
        try:
            import geopandas as gpd

            boundary = gpd.read_file(swiss_boundary_path)
            if boundary.crs is None:
                boundary = boundary.set_crs("EPSG:2056")
            boundary = boundary.to_crs("EPSG:4326")
            boundary.boundary.plot(
                ax=ax,
                color=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
            return True
        except Exception as exc:
            warnings.warn(f"Could not plot local Swiss boundary outline: {exc}")
            return False

    try:
        geom = _find_switzerland_naturalearth_geometry(naturalearth_resolution)
        for x, y in _iter_polygon_exteriors(geom):
            ax.plot(
                x,
                y,
                color=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        return True
    except Exception as exc:
        warnings.warn(
            "Could not draw Switzerland boundary from Natural Earth. "
            "Install cartopy with `pip install cartopy shapely`, or pass a local boundary file. "
            f"Original error: {exc}"
        )
        return False


# -----------------------------------------------------------------------------
# Network plotting helpers
# -----------------------------------------------------------------------------
""" Helper functions for result maps

The following plotting functions use the pandapower network geodata.
 The case builder attaches longitude/latitude coordinates to `net.bus_geodata`, 
 so the plots remain consistent with the topology map.
 """

def _prepare_branch_segments(
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    *,
    only_in_service: bool = True,
    voltage_rule: str = "max",
) -> pd.DataFrame:
    """Attach endpoint coordinates and voltage class to each branch."""
    if only_in_service and "status" in branch.columns:
        branch = branch.loc[pd.to_numeric(branch["status"], errors="coerce").fillna(1) != 0].copy()
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

    out = out.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to"]).copy()

    kv_from = pd.to_numeric(out["kv_from"], errors="coerce")
    kv_to = pd.to_numeric(out["kv_to"], errors="coerce")

    if voltage_rule == "min":
        out["voltage_kv"] = np.minimum(kv_from, kv_to)
    elif voltage_rule == "from":
        out["voltage_kv"] = kv_from
    elif voltage_rule == "to":
        out["voltage_kv"] = kv_to
    elif voltage_rule == "max":
        out["voltage_kv"] = np.maximum(kv_from, kv_to)
    else:
        raise ValueError("voltage_rule must be one of: 'max', 'min', 'from', 'to'")

    return out


def _branch_width_for_kv(kv: float) -> float:
    if kv >= 380:
        return 1.7
    if kv >= 220:
        return 1.2
    if kv >= 150:
        return 0.9
    return 0.65


def _default_voltage_colors() -> dict[float, str]:
    return {
        380.0: "#cc3311",
        220.0: "#0077bb",
        150.0: "#ee7733",
        110.0: "#33aa55",
    }


def plot_swissgrid2025_map(
    network_csv_dir: str | Path | None = None,
    *,
    case_py_path: str | Path | None = None,
    swiss_boundary_path: str | Path | None = None,
    save_path: str | Path | None = None,
    ax=None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 220,
    title: str = "Swissgrid 2025 Power Transmission Grid",
    show_swiss_map: bool = True,
    show_swiss_boundary_on_top: bool = True,
    naturalearth_resolution: str = "10m",
    show_buses: bool = True,
    show_load_arrows: bool = True,
    load_arrow_color: str = "blue",
    load_arrow_min_size: float = 0.1,
    load_arrow_max_size: float = 0.4,
    load_arrow_alpha: float = 0.75,
    show_generators: bool = True,
    show_bus_labels: bool = False,
    label_column: str = "Name",
    max_labels: int = 10,
    only_in_service: bool = True,
    voltage_rule: str = "max",
    voltage_colors: Mapping[float, str] | None = None,
    bus_color: str = "black",
    generator_color: str = "tab:green",
    branch_alpha: float = 0.86,
    boundary_margin_deg: float = 0.25,
    return_data: bool = False,
):
    """
    Plot Swissgrid 2025 network branches and buses over a Switzerland map.

    Parameters
    ----------
    network_csv_dir:
        Folder containing ``bus.csv`` and ``branch.csv``.
        Default: ``.../PROPER/data/powersystems/Swiss2025/network_csvs``.
    case_py_path:
        Optional path to ``case_swissgrid2025.py``. If ``network_csv_dir`` is not
        given, the CSV folder is inferred as ``case_py_path.parent / 'network_csvs'``.
    swiss_boundary_path:
        Optional local Swiss boundary file. If omitted, Natural Earth through
        Cartopy is used automatically. Therefore you usually do not need this.
    save_path:
        Optional output path. Supports png, pdf, svg, etc.
    show_swiss_map:
        Draw a light Switzerland polygon behind the network.
    show_swiss_boundary_on_top:
        Draw the Swiss border again above lines and buses.
    naturalearth_resolution:
        ``'10m'`` gives a clean boundary. ``'50m'`` is faster/lighter.
    return_data:
        If True, return ``(fig, ax, bus, branch_segments)``.

    Returns
    -------
    (fig, ax) or (fig, ax, bus, branch_segments)
    """
    bus, branch, BUS_NAMES, gen = load_swissgrid_csv_network(network_csv_dir, case_py_path)
    branch_segments = _prepare_branch_segments(
        bus,
        branch,
        only_in_service=only_in_service,
        voltage_rule=voltage_rule,
    )

    valid_bus = bus.dropna(subset=["lat", "lon"]).copy()
    if valid_bus.empty:
        raise ValueError("No valid bus coordinates found. Check bus.csv columns lat/lon.")

    lon_min = float(valid_bus["lon"].min()) - boundary_margin_deg
    lon_max = float(valid_bus["lon"].max()) + boundary_margin_deg
    lat_min = float(valid_bus["lat"].min()) - boundary_margin_deg
    lat_max = float(valid_bus["lat"].max()) + boundary_margin_deg

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    has_boundary = False
    if show_swiss_map:
        has_boundary = add_swiss_map_background(
            ax,
            swiss_boundary_path=swiss_boundary_path,
            naturalearth_resolution=naturalearth_resolution,
            facecolor="#f7f7f7",
            edgecolor="0.68",
            linewidth=0.8,
            alpha=1.0,
            zorder=0,
        )

    if voltage_colors is None:
        voltage_colors = _default_voltage_colors()

    # Branches grouped by voltage class.
    used_labels: set[float] = set()
    for kv in sorted(branch_segments["voltage_kv"].dropna().unique(), reverse=True):
        group = branch_segments.loc[np.isclose(branch_segments["voltage_kv"], kv)]
        segments = [
            [(row.lon_from, row.lat_from), (row.lon_to, row.lat_to)]
            for row in group.itertuples(index=False)
        ]
        if not segments:
            continue

        color = voltage_colors.get(float(kv), "0.45")
        label = f"{int(round(kv))} kV" if float(kv) not in used_labels else None
        used_labels.add(float(kv))

        lc = LineCollection(
            segments,
            colors=color,
            linewidths=_branch_width_for_kv(float(kv)),
            alpha=branch_alpha,
            zorder=2,
            label=label,
        )
        ax.add_collection(lc)

    # Buses.
    if show_buses:
        sizes = valid_bus["baseKV"].astype(float).map(
            lambda kv: 40 if kv >= 380 else 30 if kv >= 220 else 20
        )
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
    # ------------------------------------------------------------------
    # Plot active load arrows based on bus["Pd"]
    if show_load_arrows and "Pd" in valid_bus.columns:
        pd_values = valid_bus["Pd"].astype(float).values

        # Keep only buses with non-zero active power demand/injection
        mask = np.isfinite(pd_values) & (np.abs(pd_values) > 1e-6)

        if mask.any():
            load_bus = valid_bus.loc[mask].copy()
            pd_nonzero = pd_values[mask]

            # Arrow magnitude scaled by |Pd|
            abs_pd = np.abs(pd_nonzero)
            max_abs_pd = abs_pd.max()

            if max_abs_pd > 0:
                arrow_lengths = (
                    load_arrow_min_size
                    + (abs_pd / max_abs_pd)
                    * (load_arrow_max_size - load_arrow_min_size)
                )
            else:
                arrow_lengths = np.full_like(abs_pd, load_arrow_min_size)

            # Convention:
            # Pd > 0  -> load/consumption  -> arrow down
            # Pd < 0  -> net injection     -> arrow up
            dx = np.zeros_like(pd_nonzero)
            dy = np.where(pd_nonzero >= 0, -arrow_lengths, arrow_lengths)

            ax.quiver(
                load_bus["lon"].astype(float).values,
                load_bus["lat"].astype(float).values,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0022,
                headwidth=3.5,
                headlength=4.5,
                headaxislength=4.0,
                color=load_arrow_color,
                alpha=load_arrow_alpha,
                zorder=4,
                label="Pd arrows",
            )
    # Generator buses.
    if show_generators and gen is not None and not gen.empty and "bus" in gen.columns:
        gen_active = gen.copy()
        if "status" in gen_active.columns:
            gen_active = gen_active.loc[
                pd.to_numeric(gen_active["status"], errors="coerce").fillna(1) != 0
            ]
        gen_bus_ids = sorted(set(gen_active["bus"].astype(int)))
        gen_bus = valid_bus.loc[valid_bus["bus_i"].isin(gen_bus_ids)]
        if not gen_bus.empty:
            ax.scatter(
                gen_bus["lon"],
                gen_bus["lat"],
                marker="^",
                s=36,
                c=generator_color,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
                label="Generator buses",
            )

    # Optional bus labels.
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
                getattr(row, "lat") + 0.010,
                label,
                fontsize=6.5,
                color="0.15",
                zorder=5,
            )

    # Swiss outline on top of the network.
    if show_swiss_boundary_on_top:
        has_top_boundary = add_swiss_boundary_on_top(
            ax,
            swiss_boundary_path=swiss_boundary_path,
            naturalearth_resolution=naturalearth_resolution,
            edgecolor="black",
            linewidth=1.35,
            alpha=1.0,
            zorder=30,
        )
        has_boundary = has_boundary or has_top_boundary

    # If no map could be loaded, use the network extent and inform the user on the figure.
    if not has_boundary:
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
        ax.text(
            0.99,
            0.01,
            "Swiss boundary not loaded. Install cartopy or pass a local boundary file.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.35",
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    fig.tight_layout()

    if save_path is not None:
        save_path = _as_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if return_data:
        return fig, ax, bus, branch_segments
    return fig, ax



def _get_bus_coordinates(net):
    """Return bus coordinates as a DataFrame with columns x=lon, y=lat."""
    if not hasattr(net, "bus_geodata") or net.bus_geodata is None or net.bus_geodata.empty:
        raise ValueError(
            "The pandapower network has no bus_geodata. "
            "Check that case_swissgrid2025.py attaches metadata.json correctly."
        )

    xy = net.bus_geodata.copy()
    required = {"x", "y"}
    if not required.issubset(xy.columns):
        raise ValueError(f"net.bus_geodata must contain {required}; found {xy.columns.tolist()}")

    xy["x"] = pd.to_numeric(xy["x"], errors="coerce")
    xy["y"] = pd.to_numeric(xy["y"], errors="coerce")
    xy = xy.dropna(subset=["x", "y"])
    return xy


def _set_map_extent(ax, xy, margin_deg=0.25):
    ax.set_xlim(float(xy["x"].min()) - margin_deg, float(xy["x"].max()) + margin_deg)
    ax.set_ylim(float(xy["y"].min()) - margin_deg, float(xy["y"].max()) + margin_deg)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, linewidth=0.35, alpha=0.25)


def _line_segments_from_net(net, xy):
    """Build line segments from net.line and net.bus_geodata."""
    segments = []
    valid_line_indices = []

    for idx, row in net.line.iterrows():
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        if fb not in xy.index or tb not in xy.index:
            continue
        segments.append([(xy.loc[fb, "x"], xy.loc[fb, "y"]), (xy.loc[tb, "x"], xy.loc[tb, "y"])])
        valid_line_indices.append(idx)

    return segments, valid_line_indices


def _plot_swiss_context(ax):
    """Add Swiss background and boundary if Cartopy is available."""
    add_swiss_map_background(ax, facecolor="#f7f7f7", edgecolor="0.75", linewidth=0.7, zorder=0)
    add_swiss_boundary_on_top(ax, edgecolor="black", linewidth=1.25, zorder=30)




def plot_line_loading_map(
    net,
    *,
    save_path=None,
    title=None,
    pf_mode="AC",
    cmap="jet",
    line_width=2.2,
    margin_deg=0.25,
    min_alpha=0.1,
    max_alpha=1.0,
    alpha_reference=80.0,
    alpha_power=1.2,
):
    """
    Plot line loading or active power flow on the Swissgrid 2025 map.

    Lines with low loading are made transparent. Lines close to or above
    `alpha_reference` are almost fully opaque.

    Parameters
    ----------
    min_alpha:
        Transparency for almost unloaded lines.
    max_alpha:
        Transparency for highly loaded lines.
    alpha_reference:
        Loading percentage at which lines become fully visible.
        For example, 80 means lines at >=80% loading get alpha=max_alpha.
    alpha_power:
        Controls contrast. Larger values make low-loaded lines more transparent.
    """
    xy = _get_bus_coordinates(net)
    segments, line_indices = _line_segments_from_net(net, xy)

    if not segments:
        raise ValueError("No valid line segments could be built from net.line and net.bus_geodata.")

    if "loading_percent" in net.res_line.columns:
        values = pd.to_numeric(
            net.res_line.loc[line_indices, "loading_percent"],
            errors="coerce",
        )
        label = "Line loading [%]"
        default_title = f"Swissgrid 2025 line loading after {pf_mode} power flow"
        alpha_scale = alpha_reference

    elif "p_from_mw" in net.res_line.columns:
        values = pd.to_numeric(
            net.res_line.loc[line_indices, "p_from_mw"],
            errors="coerce",
        ).abs()
        label = "Absolute active power flow [MW]"
        default_title = f"Swissgrid 2025 line active power flow after {pf_mode} power flow"
        alpha_scale = max(float(values.quantile(0.90)), 1.0)

    else:
        raise ValueError("net.res_line has neither 'loading_percent' nor 'p_from_mw'.")

    values = values.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError("No finite line-flow values available for plotting.")

    vmax = max(float(np.nanmax(finite_values)), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    # ------------------------------------------------------------
    # Build per-line RGBA colors with loading-dependent transparency
    # ------------------------------------------------------------
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(np.nan_to_num(values, nan=0.0)))

    alpha_raw = np.nan_to_num(values, nan=0.0) / max(alpha_scale, 1e-9)
    alpha_raw = np.clip(alpha_raw, 0.0, 1.0)
    alpha_values = min_alpha + (max_alpha - min_alpha) * (alpha_raw ** alpha_power)

    # Hide non-finite lines almost completely
    alpha_values[~np.isfinite(values)] = 0.0

    colors[:, 3] = alpha_values

    fig, ax = plt.subplots(figsize=(9.2, 7.6))

    _plot_swiss_context(ax)

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=line_width,
        zorder=3,
    )
    ax.add_collection(lc)

    ax.scatter(
        xy["x"],
        xy["y"],
        s=7,
        c="black",
        alpha=0.45,
        linewidths=0,
        zorder=4,
    )

    _set_map_extent(ax, xy, margin_deg=margin_deg)

    ax.set_title(title or default_title)

    # Colorbar still reflects the loading/flow values, even though alpha is custom.
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array(values)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(label)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_voltage_magnitude_map(
    net,
    *,
    pf_mode = None,
    save_path=None,
    title=None,
    cmap="coolwarm",
    margin_deg=0.25,
):
    xy = _get_bus_coordinates(net)
    vm = pd.to_numeric(net.res_bus["vm_pu"], errors="coerce")

    plot_df = xy.join(vm.rename("vm_pu"), how="inner").dropna(subset=["vm_pu"])
    if plot_df.empty:
        raise ValueError("No finite bus-voltage values available for plotting.")

    segments, _ = _line_segments_from_net(net, xy)

    vmin = min(0.95, float(plot_df["vm_pu"].min()))
    vmax = max(1.05, float(plot_df["vm_pu"].max()))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    _plot_swiss_context(ax)

    if segments:
        lc = LineCollection(segments, colors="0.72", linewidths=0.75, alpha=0.95, zorder=2)
        ax.add_collection(lc)

    sc = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=plot_df["vm_pu"],
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.99,
        zorder=4,
    )

    _set_map_extent(ax, xy, margin_deg=margin_deg)


    if pf_mode is None:
        pf_mode = 'AC'
    ax.set_title(title or f"Swissgrid 2025 bus-voltage magnitudes after {pf_mode} power flow")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Voltage magnitude [p.u.]")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")




    return fig, ax



# Backward-compatible alias if another script imported the old name.
plot_swissgrid_map = plot_swissgrid2025_map


if __name__ == "__main__":
    # Adjust these paths only if your local repository location differs.
    case_path = DEFAULT_CASE_PATH
    csv_dir = DEFAULT_CSV_DIR
    save_path = DEFAULT_OUTPUT_PATH

    fig, ax = plot_swissgrid2025_map(
        network_csv_dir=csv_dir,
        case_py_path=case_path,
        save_path=save_path,
        show_bus_labels=False,
        show_swiss_map=True,
        show_swiss_boundary_on_top=True,
    )
    print(f"Saved map to: {save_path}")
    plt.show()
