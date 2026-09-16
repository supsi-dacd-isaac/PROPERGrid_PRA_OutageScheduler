"""Visualisation utilities for the monolithic deterministic and CVaR schedulers.

All figures are saved under

    outputs/schedule_results/monolitic_scheduler/figures

relative to the repository root. The plotting functions are non-blocking by
default and return the generated file paths.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# Keep the requested folder spelling for backward compatibility.
DEFAULT_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "schedule_results"
    / "monolitic_scheduler"
    / "figures"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _prepare_output_dir(output_dir: str | Path | None) -> Path:
    path = DEFAULT_FIGURE_DIR if output_dir is None else Path(output_dir)
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(
    fig: Figure,
    output_dir: Path,
    stem: str,
    *,
    show: bool,
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)

    if show:
        plt.show()
    plt.close(fig)
    return paths


def _unwrap_results(results: Any) -> Mapping[str, Any]:
    """Accept either a result dictionary or a tuple/list containing one."""

    if isinstance(results, Mapping):
        return results
    if isinstance(results, (tuple, list)) and results and isinstance(results[0], Mapping):
        return results[0]
    raise TypeError(
        "Expected a result dictionary or a non-empty tuple/list whose first "
        "element is a result dictionary."
    )


def _clean_time_label(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("step_"):
        suffix = value.removeprefix("step_")
        try:
            return int(suffix)
        except ValueError:
            return value
    return value


def _copy_frame(value: Any, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, pd.Series):
        frame = value.to_frame().T
    else:
        frame = pd.DataFrame(value)

    frame.columns = [_clean_time_label(column) for column in frame.columns]
    frame.index = frame.index.map(str)
    return frame.apply(pd.to_numeric, errors="coerce")


def _series_over_time(value: Any) -> pd.Series:
    """Convert a Series/DataFrame to one aggregate time series."""

    if value is None:
        return pd.Series(dtype=float)

    if isinstance(value, pd.Series):
        series = value.copy()
        series.index = [_clean_time_label(index) for index in series.index]
        return pd.to_numeric(series, errors="coerce")

    frame = _copy_frame(value, "time_series")
    if frame.empty:
        return pd.Series(dtype=float)

    if frame.shape[0] == 1:
        return frame.iloc[0]
    if frame.shape[1] == 1:
        series = frame.iloc[:, 0]
        series.index = [_clean_time_label(index) for index in series.index]
        return series

    # Standard scheduler convention: rows are assets/contingencies and columns
    # are time steps.
    return frame.sum(axis=0, min_count=1)


def _ordered_union(left: Iterable[Any], right: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in [*left, *right]:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _align_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    fill_value: float = np.nan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _ordered_union(left.index, right.index)
    columns = _ordered_union(left.columns, right.columns)
    return (
        left.reindex(index=rows, columns=columns, fill_value=fill_value),
        right.reindex(index=rows, columns=columns, fill_value=fill_value),
    )


def _align_series(
    left: pd.Series,
    right: pd.Series,
    *,
    fill_value: float = np.nan,
) -> tuple[pd.Series, pd.Series]:
    index = _ordered_union(left.index, right.index)
    return (
        left.reindex(index, fill_value=fill_value),
        right.reindex(index, fill_value=fill_value),
    )


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _ecdf(values: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(_finite(values))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _add_global_legend(fig: Figure, axes: Sequence[Axes], *, ncol: int = 2) -> None:
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=ncol,
            frameon=False,
        )


def _time_axis(axis: Axes) -> None:
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.grid(True, alpha=0.3)


def _outage_start(row: pd.Series) -> float:
    active = pd.to_numeric(row, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    positions = np.flatnonzero(active)
    return float(positions[0]) if positions.size else np.nan


def _normalise_named_rows(frame: pd.DataFrame, preferred_names: Sequence[str]) -> pd.DataFrame:
    available = [name for name in preferred_names if name in frame.index]
    if available:
        return frame.loc[available]
    return frame


# ---------------------------------------------------------------------------
# Individual result visualisation
# ---------------------------------------------------------------------------
def visualize_results(
    results_dictionary: Mapping[str, Any] | Sequence[Any],
    names: Mapping[str, Sequence[str]],
    *,
    output_dir: str | Path | None = None,
    prefix: str = "scheduler",
    show: bool = False,
    dpi: int = 220,
    formats: Sequence[str] = ("png", "pdf"),
    max_contingencies: int = 10,
    max_lines: int = 15,
) -> list[Path]:
    """Create and save compact diagnostics for one scheduler result."""

    output_path = _prepare_output_dir(output_dir)
    results = _unwrap_results(results_dictionary)

    outage_names = list(map(str, names.get("outages", [])))
    generator_names = list(map(str, names.get("generators", [])))
    line_names = list(map(str, names.get("lines", [])))

    schedule = _normalise_named_rows(
        _copy_frame(results.get("X_OutageSchedule"), "X_OutageSchedule"),
        outage_names,
    )
    generation = _normalise_named_rows(
        _copy_frame(results.get("PowerGenerated"), "PowerGenerated"),
        generator_names,
    )
    line_flows = _normalise_named_rows(
        _copy_frame(results.get("Line_Flows"), "Line_Flows"),
        line_names,
    )
    contingency_curtailment = _copy_frame(
        results.get("WC_CURTAIL_CON"),
        "WC_CURTAIL_CON",
    )
    total_curtailment = _series_over_time(results.get("WC_CURTAIL"))

    generated: list[Path] = []

    # 1. Compact maintenance overview.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    if not schedule.empty:
        image = axes[0, 0].imshow(
            schedule.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, 0].set_title("Outage schedule")
        axes[0, 0].set_xlabel("Time step")
        axes[0, 0].set_ylabel("Outage")
        axes[0, 0].set_yticks(np.arange(len(schedule.index)))
        axes[0, 0].set_yticklabels(schedule.index, fontsize=8)
        fig.colorbar(image, ax=axes[0, 0], label="Active status")

        active_tasks = schedule.sum(axis=0)
        axes[0, 1].plot(active_tasks.index, active_tasks.to_numpy(), marker="o")
        axes[0, 1].set_title("Simultaneous planned outages")
        axes[0, 1].set_xlabel("Time step")
        axes[0, 1].set_ylabel("Number of active outages")
        _time_axis(axes[0, 1])

        durations = schedule.sum(axis=1).sort_values(ascending=False)
        axes[1, 0].bar(np.arange(len(durations)), durations.to_numpy())
        axes[1, 0].set_title("Scheduled outage durations")
        axes[1, 0].set_xlabel("Outage")
        axes[1, 0].set_ylabel("Duration [steps]")
        axes[1, 0].set_xticks(np.arange(len(durations)))
        axes[1, 0].set_xticklabels(durations.index, rotation=45, ha="right")
        axes[1, 0].grid(True, axis="y", alpha=0.3)
    else:
        for axis in axes[0, :]:
            axis.text(0.5, 0.5, "No outage schedule available", ha="center", va="center")
            axis.axis("off")
        axes[1, 0].axis("off")

    if not total_curtailment.empty:
        axes[1, 1].plot(
            total_curtailment.index,
            total_curtailment.to_numpy(dtype=float),
            marker="x",
        )
        axes[1, 1].set_title("Total load curtailment")
        axes[1, 1].set_xlabel("Time step")
        axes[1, 1].set_ylabel("Curtailment [MW]")
        _time_axis(axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "No curtailment data available", ha="center", va="center")
        axes[1, 1].axis("off")

    fig.suptitle("Monolithic scheduler: operational overview", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    generated += _save_figure(
        fig,
        output_path,
        f"{prefix}_01_operational_overview",
        show=show,
        dpi=dpi,
        formats=formats,
    )

    # 2. Contingency curtailment in one compact figure.
    active_contingencies = contingency_curtailment.loc[
        (contingency_curtailment.fillna(0.0).abs() > 1e-10).any(axis=1)
    ]
    if not active_contingencies.empty:
        ranking = active_contingencies.abs().sum(axis=1).sort_values(ascending=False)
        selected = active_contingencies.loc[ranking.head(max_contingencies).index]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for contingency, row in selected.iterrows():
            axes[0].plot(row.index, row.to_numpy(dtype=float), label=contingency)
        axes[0].set_title(f"Top {len(selected)} contingency curtailments")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Load shed [MW]")
        _time_axis(axes[0])
        axes[0].legend(fontsize=7, ncol=2)

        image = axes[1].imshow(
            selected.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
        )
        axes[1].set_title("Contingency–time curtailment map")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Contingency")
        axes[1].set_yticks(np.arange(len(selected.index)))
        axes[1].set_yticklabels(selected.index, fontsize=7)
        fig.colorbar(image, ax=axes[1], label="Load shed [MW]")

        fig.tight_layout()
        generated += _save_figure(
            fig,
            output_path,
            f"{prefix}_02_contingency_curtailment",
            show=show,
            dpi=dpi,
            formats=formats,
        )

    # 3. Generation diagnostics consolidated in one figure.
    if not generation.empty:
        total_generation = generation.sum(axis=0)
        generator_means = generation.mean(axis=1).sort_values(ascending=False)
        generator_data = [
            _finite(generation.loc[generator])
            for generator in generator_means.index
        ]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(
            total_generation.index,
            total_generation.to_numpy(dtype=float),
            marker="o",
            markersize=3,
        )
        axes[0].set_title("Total generation over time")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Generation [MW]")
        _time_axis(axes[0])

        axes[1].boxplot(
            generator_data,
            tick_labels=generator_means.index,
            showfliers=False,
        )
        axes[1].set_title("Generation distribution by unit")
        axes[1].set_ylabel("Generation [MW]")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        generated += _save_figure(
            fig,
            output_path,
            f"{prefix}_03_generation_summary",
            show=show,
            dpi=dpi,
            formats=formats,
        )

    # 4. Critical line-flow diagnostics consolidated in one figure.
    if not line_flows.empty:
        line_score = line_flows.abs().max(axis=1).sort_values(ascending=False)
        selected_names = line_score.head(max_lines).index
        selected = line_flows.loc[selected_names]

        n_columns = 3
        n_rows = ceil(len(selected) / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(15, max(3.0, 2.6 * n_rows)),
            squeeze=False,
        )

        for axis, (line_name, row) in zip(axes.flat, selected.iterrows()):
            axis.plot(row.index, row.to_numpy(dtype=float))
            axis.axhline(0.0, linewidth=0.8)
            axis.set_title(line_name)
            axis.set_ylabel("Flow [MW]")
            _time_axis(axis)

        for axis in axes.flat[len(selected):]:
            axis.axis("off")

        fig.suptitle("Lines with the largest absolute flows", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        generated += _save_figure(
            fig,
            output_path,
            f"{prefix}_04_critical_line_flows",
            show=show,
            dpi=dpi,
            formats=formats,
        )

    return generated


# ---------------------------------------------------------------------------
# Compact CVaR distribution figure
# ---------------------------------------------------------------------------
def plot_risk_pdf_cdf(
    CVARisk: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "cvar",
    show: bool = False,
    dpi: int = 220,
    formats: Sequence[str] = ("png", "pdf"),
    highlighted_buses: int = 5,
) -> list[Path]:
    """Save one compact figure containing the principal CVaR diagnostics."""

    output_path = _prepare_output_dir(output_dir)
    risk = _copy_frame(CVARisk, "CVARisk")
    if risk.empty:
        raise ValueError("CVARisk is empty.")

    total_over_time = risk.sum(axis=0, min_count=1)
    all_values = _finite(risk.to_numpy())
    if all_values.size == 0:
        raise ValueError("CVARisk contains no finite values.")

    bus_mean = risk.mean(axis=1)
    bus_q95 = risk.quantile(0.95, axis=1)
    top_buses = bus_q95.sort_values(ascending=False).head(highlighted_buses).index

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) Aggregate temporal evolution.
    axes[0, 0].plot(
        total_over_time.index,
        total_over_time.to_numpy(dtype=float),
        marker="o",
        markersize=3,
    )
    axes[0, 0].set_title("Aggregate CVaR over time")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Total CVaR")
    _time_axis(axes[0, 0])

    # (b) Pooled empirical density.
    axes[0, 1].hist(all_values, bins=30, density=True, alpha=0.75)
    axes[0, 1].axvline(np.mean(all_values), linestyle="--", label="Mean")
    axes[0, 1].axvline(np.quantile(all_values, 0.95), linestyle=":", label="95th percentile")
    axes[0, 1].set_title("Pooled nodal CVaR distribution")
    axes[0, 1].set_xlabel("CVaR")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # (c) All nodal ECDFs in one axis; only the most critical buses are labelled.
    for bus, row in risk.iterrows():
        x, y = _ecdf(row)
        if x.size == 0:
            continue
        if bus in top_buses:
            axes[1, 0].step(x, y, where="post", linewidth=1.8, label=bus)
        else:
            axes[1, 0].step(x, y, where="post", linewidth=0.7, alpha=0.18)
    axes[1, 0].set_title("Nodal empirical CDFs")
    axes[1, 0].set_xlabel("CVaR")
    axes[1, 0].set_ylabel("Empirical probability")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(title="Highest nodal q95", fontsize=8)

    # (d) Bus-level mean versus upper-tail risk.
    axes[1, 1].scatter(bus_mean, bus_q95, alpha=0.75)
    for bus in top_buses:
        axes[1, 1].annotate(
            bus,
            (float(bus_mean.loc[bus]), float(bus_q95.loc[bus])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_title("Mean and 95th-percentile nodal CVaR")
    axes[1, 1].set_xlabel("Mean CVaR")
    axes[1, 1].set_ylabel("95th-percentile CVaR")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Compact CVaR diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(
        fig,
        output_path,
        f"{prefix}_risk_pdf_cdf_compact",
        show=show,
        dpi=dpi,
        formats=formats,
    )


# ---------------------------------------------------------------------------
# Deterministic versus CVaR comparison
# ---------------------------------------------------------------------------
def plot_comparison_CVAR_DET_SCOS(
    dic_res_cvar: Mapping[str, Any] | Sequence[Any],
    dic_res_det: Mapping[str, Any] | Sequence[Any],
    DATA: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
    dpi: int = 220,
    formats: Sequence[str] = ("png", "pdf"),
    max_lines: int = 12,
) -> list[Path]:
    """Create compact deterministic-versus-CVaR scheduler comparisons."""

    output_path = _prepare_output_dir(output_dir)
    cvar_results = _unwrap_results(dic_res_cvar)
    det_results = _unwrap_results(dic_res_det)

    outage_names = list(map(str, DATA["outages"]["names"]))
    line_names = [f"line_{line}" for line in range(DATA["num_branches"])]

    det_schedule = _normalise_named_rows(
        _copy_frame(det_results.get("X_OutageSchedule"), "det.X_OutageSchedule"),
        outage_names,
    )
    cvar_schedule = _normalise_named_rows(
        _copy_frame(cvar_results.get("X_OutageSchedule"), "cvar.X_OutageSchedule"),
        outage_names,
    )
    det_schedule, cvar_schedule = _align_frames(
        det_schedule,
        cvar_schedule,
        fill_value=0.0,
    )

    det_generation = _copy_frame(det_results.get("PowerGenerated"), "det.PowerGenerated")
    cvar_generation = _copy_frame(cvar_results.get("PowerGenerated"), "cvar.PowerGenerated")
    det_generation, cvar_generation = _align_frames(
        det_generation,
        cvar_generation,
        fill_value=0.0,
    )

    det_flows = _normalise_named_rows(
        _copy_frame(det_results.get("Line_Flows"), "det.Line_Flows"),
        line_names,
    )
    cvar_flows = _normalise_named_rows(
        _copy_frame(cvar_results.get("Line_Flows"), "cvar.Line_Flows"),
        line_names,
    )
    det_flows, cvar_flows = _align_frames(det_flows, cvar_flows)

    det_curtailment = _series_over_time(det_results.get("WC_CURTAIL"))
    cvar_curtailment = _series_over_time(cvar_results.get("WC_CURTAIL"))
    det_curtailment, cvar_curtailment = _align_series(
        det_curtailment,
        cvar_curtailment,
        fill_value=0.0,
    )

    generated: list[Path] = []

    # 1. Schedule comparison: deterministic, CVaR, difference, active tasks.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    common_vmax = max(
        1.0,
        float(np.nanmax(det_schedule.to_numpy(dtype=float)))
        if not det_schedule.empty
        else 1.0,
        float(np.nanmax(cvar_schedule.to_numpy(dtype=float)))
        if not cvar_schedule.empty
        else 1.0,
    )

    det_image = axes[0, 0].imshow(
        det_schedule.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=common_vmax,
    )
    axes[0, 0].set_title("Deterministic outage schedule")
    axes[0, 0].set_ylabel("Outage")
    axes[0, 0].set_yticks(np.arange(len(det_schedule.index)))
    axes[0, 0].set_yticklabels(det_schedule.index, fontsize=8)
    fig.colorbar(det_image, ax=axes[0, 0], label="Active status")

    cvar_image = axes[0, 1].imshow(
        cvar_schedule.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=common_vmax,
    )
    axes[0, 1].set_title("CVaR outage schedule")
    axes[0, 1].set_yticks(np.arange(len(cvar_schedule.index)))
    axes[0, 1].set_yticklabels(cvar_schedule.index, fontsize=8)
    fig.colorbar(cvar_image, ax=axes[0, 1], label="Active status")

    schedule_delta = cvar_schedule - det_schedule
    delta_limit = max(1.0, float(np.nanmax(np.abs(schedule_delta.to_numpy(dtype=float)))))
    delta_image = axes[1, 0].imshow(
        schedule_delta.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=-delta_limit,
        vmax=delta_limit,
        cmap="coolwarm",
    )
    axes[1, 0].set_title("Schedule difference: CVaR − deterministic")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Outage")
    axes[1, 0].set_yticks(np.arange(len(schedule_delta.index)))
    axes[1, 0].set_yticklabels(schedule_delta.index, fontsize=8)
    fig.colorbar(delta_image, ax=axes[1, 0], label="Status difference")

    axes[1, 1].plot(
        det_schedule.columns,
        det_schedule.sum(axis=0).to_numpy(dtype=float),
        linestyle="-",
        label="Deterministic",
    )
    axes[1, 1].plot(
        cvar_schedule.columns,
        cvar_schedule.sum(axis=0).to_numpy(dtype=float),
        linestyle="--",
        label="CVaR",
    )
    axes[1, 1].set_title("Simultaneous planned outages")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Number of active outages")
    _time_axis(axes[1, 1])
    axes[1, 1].legend()

    fig.suptitle("Outage-schedule comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    generated += _save_figure(
        fig,
        output_path,
        "comparison_01_outage_schedules",
        show=show,
        dpi=dpi,
        formats=formats,
    )

    # 2. New comparison: outage start shifts and duration differences.
    starts_det = det_schedule.apply(_outage_start, axis=1)
    starts_cvar = cvar_schedule.apply(_outage_start, axis=1)
    durations_det = det_schedule.sum(axis=1)
    durations_cvar = cvar_schedule.sum(axis=1)

    shift_table = pd.DataFrame(
        {
            "deterministic_start": starts_det,
            "cvar_start": starts_cvar,
            "start_shift_cvar_minus_det": starts_cvar - starts_det,
            "deterministic_duration": durations_det,
            "cvar_duration": durations_cvar,
            "duration_difference_cvar_minus_det": durations_cvar - durations_det,
        }
    )
    shift_table.to_csv(output_path / "comparison_outage_start_shifts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    positions = np.arange(len(shift_table))
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        shift_table["deterministic_start"],
        width=width,
        label="Deterministic",
    )
    axes[0].bar(
        positions + width / 2,
        shift_table["cvar_start"],
        width=width,
        label="CVaR",
    )
    axes[0].set_title("Outage start periods")
    axes[0].set_ylabel("Start step")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(shift_table.index, rotation=45, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        positions,
        shift_table["start_shift_cvar_minus_det"].fillna(0.0),
    )
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_title("Start shift: CVaR − deterministic")
    axes[1].set_ylabel("Shift [steps]")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(shift_table.index, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    generated += _save_figure(
        fig,
        output_path,
        "comparison_02_outage_start_shifts",
        show=show,
        dpi=dpi,
        formats=formats,
    )

    # 3. New comparison: curtailment time series, cumulative consequence,
    # empirical CDFs and pointwise difference in one compact figure.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(
        det_curtailment.index,
        det_curtailment.to_numpy(dtype=float),
        linestyle="-",
        label="Deterministic",
    )
    axes[0, 0].plot(
        cvar_curtailment.index,
        cvar_curtailment.to_numpy(dtype=float),
        linestyle="--",
        label="CVaR",
    )
    axes[0, 0].set_title("Total load curtailment")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Curtailment [MW]")
    _time_axis(axes[0, 0])
    axes[0, 0].legend()

    axes[0, 1].plot(
        det_curtailment.index,
        det_curtailment.fillna(0.0).cumsum(),
        linestyle="-",
        label="Deterministic",
    )
    axes[0, 1].plot(
        cvar_curtailment.index,
        cvar_curtailment.fillna(0.0).cumsum(),
        linestyle="--",
        label="CVaR",
    )
    axes[0, 1].set_title("Cumulative curtailment")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Cumulative MW-step")
    _time_axis(axes[0, 1])
    axes[0, 1].legend()

    for series, label, style in (
        (det_curtailment, "Deterministic", "-"),
        (cvar_curtailment, "CVaR", "--"),
    ):
        x, y = _ecdf(series)
        axes[1, 0].step(x, y, where="post", linestyle=style, label=label)
    axes[1, 0].set_title("Empirical CDF of time-step curtailment")
    axes[1, 0].set_xlabel("Curtailment [MW]")
    axes[1, 0].set_ylabel("Empirical probability")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    curtailment_delta = cvar_curtailment - det_curtailment
    axes[1, 1].bar(curtailment_delta.index, curtailment_delta.to_numpy(dtype=float))
    axes[1, 1].axhline(0.0, linewidth=0.8)
    axes[1, 1].set_title("Curtailment difference: CVaR − deterministic")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Difference [MW]")
    _time_axis(axes[1, 1])

    fig.suptitle("Curtailment comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    generated += _save_figure(
        fig,
        output_path,
        "comparison_03_curtailment",
        show=show,
        dpi=dpi,
        formats=formats,
    )

    # 4. New comparison: generation totals and generator-level changes.
    if not det_generation.empty or not cvar_generation.empty:
        total_det = det_generation.sum(axis=0)
        total_cvar = cvar_generation.sum(axis=0)
        total_det, total_cvar = _align_series(total_det, total_cvar, fill_value=0.0)

        mean_det = det_generation.mean(axis=1)
        mean_cvar = cvar_generation.mean(axis=1)
        mean_det, mean_cvar = _align_series(mean_det, mean_cvar, fill_value=0.0)
        generator_delta = (mean_cvar - mean_det).sort_values(
            key=lambda series: series.abs(),
            ascending=False,
        )

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(total_det.index, total_det, linestyle="-", label="Deterministic")
        axes[0].plot(total_cvar.index, total_cvar, linestyle="--", label="CVaR")
        axes[0].set_title("Total generation")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Generation [MW]")
        _time_axis(axes[0])
        axes[0].legend()

        selected_generator_delta = generator_delta.head(15)
        axes[1].bar(
            np.arange(len(selected_generator_delta)),
            selected_generator_delta.to_numpy(dtype=float),
        )
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_title("Largest mean generator changes: CVaR − deterministic")
        axes[1].set_ylabel("Mean generation difference [MW]")
        axes[1].set_xticks(np.arange(len(selected_generator_delta)))
        axes[1].set_xticklabels(
            selected_generator_delta.index,
            rotation=45,
            ha="right",
        )
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        generated += _save_figure(
            fig,
            output_path,
            "comparison_04_generation",
            show=show,
            dpi=dpi,
            formats=formats,
        )

    # 5. Improved line-flow comparison: select lines by actual discrepancy,
    # rather than plotting the first 15 lines.
    if not det_flows.empty or not cvar_flows.empty:
        flow_difference = cvar_flows - det_flows
        line_score = pd.concat(
            [
                det_flows.abs().max(axis=1).rename("det_max"),
                cvar_flows.abs().max(axis=1).rename("cvar_max"),
                flow_difference.abs().max(axis=1).rename("difference_max"),
            ],
            axis=1,
        ).max(axis=1)
        selected_lines = line_score.sort_values(ascending=False).head(max_lines).index

        n_columns = 3
        n_rows = ceil(len(selected_lines) / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(15, max(3.2, 2.8 * n_rows)),
            squeeze=False,
        )

        for axis, line_name in zip(axes.flat, selected_lines):
            axis.plot(
                det_flows.columns,
                det_flows.loc[line_name].to_numpy(dtype=float),
                linestyle="-",
                label="Deterministic",
            )
            axis.plot(
                cvar_flows.columns,
                cvar_flows.loc[line_name].to_numpy(dtype=float),
                linestyle="--",
                label="CVaR",
            )
            axis.axhline(0.0, linewidth=0.7)
            axis.set_title(line_name)
            axis.set_ylabel("Flow [MW]")
            _time_axis(axis)

        for axis in axes.flat[len(selected_lines):]:
            axis.axis("off")

        _add_global_legend(fig, list(axes.flat), ncol=2)
        fig.suptitle(
            "Line flows with the largest loading or formulation difference",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        generated += _save_figure(
            fig,
            output_path,
            "comparison_05_critical_line_flows",
            show=show,
            dpi=dpi,
            formats=formats,
        )

    # 6. Compact numerical summary for the two schedules.
    metrics = pd.DataFrame(
        {
            "Deterministic": {
                "Scheduled active steps": float(det_schedule.to_numpy().sum()),
                "Total curtailment [MW-step]": float(det_curtailment.fillna(0.0).sum()),
                "Peak curtailment [MW]": float(det_curtailment.max(skipna=True)),
                "Mean total generation [MW]": float(det_generation.sum(axis=0).mean()),
                "Maximum absolute line flow [MW]": float(
                    np.nanmax(np.abs(det_flows.to_numpy(dtype=float)))
                )
                if not det_flows.empty
                else np.nan,
            },
            "CVaR": {
                "Scheduled active steps": float(cvar_schedule.to_numpy().sum()),
                "Total curtailment [MW-step]": float(cvar_curtailment.fillna(0.0).sum()),
                "Peak curtailment [MW]": float(cvar_curtailment.max(skipna=True)),
                "Mean total generation [MW]": float(cvar_generation.sum(axis=0).mean()),
                "Maximum absolute line flow [MW]": float(
                    np.nanmax(np.abs(cvar_flows.to_numpy(dtype=float)))
                )
                if not cvar_flows.empty
                else np.nan,
            },
        }
    )
    metrics["CVaR minus deterministic"] = metrics["CVaR"] - metrics["Deterministic"]
    metrics.to_csv(output_path / "comparison_summary_metrics.csv")

    # Plot relative differences only, avoiding a misleading mixed-unit grouped bar.
    denominator = metrics["Deterministic"].abs().replace(0.0, np.nan)
    relative_change = 100.0 * (
        metrics["CVaR"] - metrics["Deterministic"]
    ) / denominator

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(np.arange(len(relative_change)), relative_change.fillna(0.0))
    axis.axhline(0.0, linewidth=0.8)
    axis.set_title("Relative change from deterministic to CVaR schedule")
    axis.set_ylabel("Relative change [%]")
    axis.set_xticks(np.arange(len(relative_change)))
    axis.set_xticklabels(relative_change.index, rotation=35, ha="right")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    generated += _save_figure(
        fig,
        output_path,
        "comparison_06_relative_metrics",
        show=show,
        dpi=dpi,
        formats=formats,
    )

    return generated
