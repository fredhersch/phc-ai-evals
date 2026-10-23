"""
a-CDU Lifecycle Harness
Compiler, static linter, adversarial collision testbed, and packager for Atomic Clinical Decision Units.
"""

from .models import ClinicalDecisionUnit
from .linter import lint_cdu, lint_cdu_library
from .collision_runner import run_collision_test_suite
from .packager import export_to_sqlite, export_to_bundle_json
from .graph import ClinicalKnowledgeGraph, CollisionRecord
from .ontology import DRUG_CLASSES, CLINICAL_HAZARDS, CLINICAL_CONDITIONS

__version__ = "1.1.0"
