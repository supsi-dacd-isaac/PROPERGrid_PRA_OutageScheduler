"""Deprecated legacy module.

Constraint construction is now centralised in ``monolithic_scos.py``. The old
implementation remains under ``optimizers/legacy`` for reproducibility.
"""

from .monolithic_scos import contingency_asset

__all__ = ["contingency_asset"]
