#!/usr/bin/env python3
"""Thin entry point for the static contingency-PRA comparison."""

from __future__ import annotations

import sys

from run_pra_comparison import main


if __name__ == "__main__":
    main(["--engines", "contingency", *sys.argv[1:]])
