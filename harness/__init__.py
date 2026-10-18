"""
a-CDU Lifecycle Harness
Compiler, static linter, adversarial collision testbed, and packager for Atomic Clinical Decision Units.
"""

from .models import ClinicalDecisionUnit
from .linter import lint_cdu, lint_cdu_library
from .collision_runner import run_collision_test_suite
from .packager import export_to_sqlite, export_to_bundle_json

__version__ = "1.0.0"
