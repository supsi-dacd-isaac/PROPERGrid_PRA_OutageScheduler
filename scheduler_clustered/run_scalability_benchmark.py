"""Backward-compatible entry point for the three-axis scalability study."""
from __future__ import annotations

from scheduler_clustered.scalability_study import main_scalability


def main(argv=None) -> int:
    return main_scalability(argv)


if __name__ == "__main__":
    raise SystemExit(main())
