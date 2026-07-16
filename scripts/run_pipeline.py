#!/usr/bin/env python3
"""Convenience runner — `python scripts/run_pipeline.py run-all` from project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_rec.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
