"""Compatibility exports for the compact comparison visualisation module."""

from .comparison_visualization import (
    plot_comparison_suite,
    plot_operational_comparison,
    plot_schedule_comparison,
    plot_tail_contingency_contributions,
    plot_tail_risk_comparison,
)

__all__ = [
    "plot_comparison_suite",
    "plot_operational_comparison",
    "plot_schedule_comparison",
    "plot_tail_contingency_contributions",
    "plot_tail_risk_comparison",
]
