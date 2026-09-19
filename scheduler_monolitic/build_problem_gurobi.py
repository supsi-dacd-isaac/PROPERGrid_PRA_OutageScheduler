"""Deprecated compatibility module.

The former collection of duplicated variable and constraint builders has been
replaced by :class:`scheduler_monolitic.monolithic_scos.MonolithicSCOS`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "build_problem_gurobi is deprecated; import MonolithicSCOS from scheduler_monolitic.monolithic_scos.",
    DeprecationWarning,
    stacklevel=2,
)

from .monolithic_scos import MonolithicSCOS, solve_scos

__all__ = ["MonolithicSCOS", "solve_scos"]
