"""
cli.py - Command Line Interface for the a-CDU Lifecycle Harness
Subcommands: lint, test, package, init
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .models import ClinicalDecisionUnit
from .linter import lint_cdu_library
from .collision_runner import run_collision_test_suite
from .packager import export_to_sqlite, export_to_bundle_json


def load_cdus_from_file(file_path: Path) -> List[ClinicalDecisionUnit]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cdus" in data:
        data = data["cdus"]
    return [ClinicalDecisionUnit.from_dict(item) for item in data]


def cmd_lint(args) -> int:
    path = Path(args.path)
    cdus = load_cdus_from_file(path)
    print(f"\n==========================================================================")
    print(f"  a-CDU Static Linter & Semantic Safety Check")
    print(f"  Target: {path} ({len(cdus)} units)")
    print(f"==========================================================================\n")

    overall_pass, results = lint_cdu_library(cdus)

    pass_count = 0
    warn_count = 0
    error_count = 0

    for r in results:
        status_badge = "✅ PASS" if r.is_valid and not r.has_warnings else ("⚠️ WARN" if r.is_valid else "❌ FAIL")
        print(f"[{status_badge}] {r.cdu_id:<32} | Tokens: {r.token_count:>3}/180")
        for issue in r.issues:
            icon = "  🛑" if issue.level == "ERROR" else "  ⚠️"
            print(f"    {icon} [{issue.level}] {issue.field_name}: {issue.message}")

        if not r.is_valid:
            error_count += 1
        elif r.has_warnings:
            warn_count += 1
        else:
            pass_count += 1

    print("\n" + "=" * 74)
    print(f"  LINT SUMMARY: {pass_count} Passed, {warn_count} Warnings, {error_count} Errors")
    print("=" * 74 + "\n")
    return 0 if overall_pass else 1


def cmd_test_collision(args) -> int:
    cases_path = Path(args.cases)
    cdus_path = Path(args.cdus)
    cdus = load_cdus_from_file(cdus_path)

    res = run_collision_test_suite(
        cases_path=cases_path,
        cdu_library=cdus,
        dry_run=args.dry_run,
        model=args.model,
        api_base=args.api_base
    )
    return 0 if res["all_safe"] else 1


def cmd_package(args) -> int:
    cdus_path = Path(args.cdus)
    cdus = load_cdus_from_file(cdus_path)
    target = args.target.lower()

    if target in ["sqlite", "all"]:
        out_db = args.output or "dist/phc_cdu_registry.db"
        res_path = export_to_sqlite(cdus, out_db)
        print(f"✅ Successfully compiled {len(cdus)} a-CDUs into offline SQLite FTS5 database:")
        print(f"   -> {res_path}")

    if target in ["json", "all"]:
        out_json = args.output if (target == "json" and args.output) else "dist/phc_cdu_bundle.json"
        res_path = export_to_bundle_json(cdus, out_json)
        print(f"✅ Successfully exported {len(cdus)} a-CDUs into cryptographic JSON bundle:")
        print(f"   -> {res_path}")

    return 0


def cmd_init(args) -> int:
    cdu_id = args.id or "CDU-TOPIC-SPECIFIER"
    template = {
        "id": cdu_id,
        "topic": "Clinical Topic Description",
        "domain": "primary_care_multimorbidity",
        "source_guideline": "WHO SMART Guidelines 2024 / National MoH Guideline",
        "priority_tier": 1,
        "entry_criteria": "Patient clinical condition and laboratory thresholds",
        "mandated_actions": [
            "First-line drug initiation with exact dosing targets",
            "Mandatory renal or biometric monitoring interval"
        ],
        "negative_boundaries": [
            "🛑 ABSOLUTELY NO [Banned Drug Class] -> [Fatal pathophysiological hazard]",
            "🛑 Strictly avoid [Contraindicated Procedure/Target]"
        ],
        "version": "1.0.0",
        "certified": False
    }
    out_file = Path(f"{cdu_id.lower().replace('-', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(f"✅ Initialized draft a-CDU template: {out_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="a-CDU Lifecycle Harness - Compiler, Linter & Testbed")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: lint
    p_lint = subparsers.add_parser("lint", help="Statically lint CDUs for 180-token ceiling and contradictions")
    p_lint.add_argument("path", type=str, help="Path to CDU JSON or directory")
    p_lint.set_defaults(func=cmd_lint)

    # Subcommand: test-collision
    p_test = subparsers.add_parser("test-collision", help="Execute multi-morbidity collision regression suite")
    p_test.add_argument("--cases", type=str, default="dataset/phc_multimorbidity_cases.json", help="Path to cases JSON")
    p_test.add_argument("--cdus", type=str, default="dataset/phc_decision_units.json", help="Path to CDUs JSON")
    p_test.add_argument("--dry-run", action="store_true", default=False, help="Run instant in-memory regression")
    p_test.add_argument("--model", type=str, default="medgemma:latest", help="Model name")
    p_test.add_argument("--api-base", type=str, default="http://localhost:11434", help="Ollama API base")
    p_test.set_defaults(func=cmd_test_collision)

    # Subcommand: package
    p_pkg = subparsers.add_parser("package", help="Compile verified CDUs into offline SQLite or JSON bundle")
    p_pkg.add_argument("--cdus", type=str, default="dataset/phc_decision_units.json", help="Path to CDUs JSON")
    p_pkg.add_argument("--target", type=str, choices=["sqlite", "json", "all"], default="all", help="Export target format")
    p_pkg.add_argument("--output", type=str, default="", help="Custom output path")
    p_pkg.set_defaults(func=cmd_package)

    # Subcommand: init
    p_init = subparsers.add_parser("init", help="Scaffold a new draft a-CDU template")
    p_init.add_argument("--id", type=str, default="", help="CDU ID (e.g. CDU-ANC-SEVERE-PE)")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    return args.func(args)
