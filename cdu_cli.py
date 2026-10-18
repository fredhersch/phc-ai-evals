#!/usr/bin/env python3
"""
cdu_cli.py - Top-level CLI entrypoint for the a-CDU Lifecycle Harness
"""

import sys
from pathlib import Path

# Add repo root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
