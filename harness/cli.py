"""
cli.py - Command Line Interface for the a-CDU Lifecycle Harness
Subcommands: lint, test, package, init
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import List

from .models import ClinicalDecisionUnit
from .linter import lint_cdu_library
from .collision_runner import run_collision_test_suite
from .packager import export_to_sqlite, export_to_bundle_json
from .graph import ClinicalKnowledgeGraph


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


def cmd_graph_collisions(args) -> int:
    path = Path(args.path)
    cdus = load_cdus_from_file(path)

    kg = ClinicalKnowledgeGraph()
    kg.populate_library(cdus)
    collisions = kg.detect_collisions()

    print(f"\n==========================================================================")
    print(f"  a-CDU Clinical Knowledge Graph & Collision Detection")
    print(f"  Target: {path} ({len(cdus)} units loaded)")
    print(f"  Graph Nodes: {len(kg.cdus)} CDUs, {len(kg.drug_classes)} Drug Classes, {len(kg.conditions)} Conditions")
    print(f"  Graph Edges: {len(kg.edges)} semantic relations")
    print(f"==========================================================================\n")

    if not collisions:
        print("✅ Zero cross-CDU collisions detected.")
    else:
        print(f"⚡ DETECTED {len(collisions)} CROSS-CDU ACTION ↔ CONTRAINDICATION COLLISIONS:\n")
        for idx, c in enumerate(collisions, start=1):
            sev_badge = "🛑 FATAL CLASH (T1)" if c.hazard_tier == 1 else "⚠️ WARNING CLASH (T2)"
            print(f"[{idx:02d}] {sev_badge} | Concept: {c.concept_name}")
            print(f"     Prescribing Unit : {c.cdu_a_id} (Tier {c.cdu_a_tier}) -> {c.cdu_a_topic[:45]}")
            print(f"     Prohibiting Unit : {c.cdu_b_id} (Tier {c.cdu_b_tier}) -> {c.cdu_b_topic[:45]}")
            print(f"     Clinical Hazard  : {c.hazard_name}")
            print(f"     Precedence Rule  : {c.recommended_precedence}")
            print(f"     Safe Alternative : {c.suggested_safe_alternative}")
            if c.tier_calibration_needed:
                print(f"     👉 CALIBRATION   : {c.calibration_recommendation}")
            print("-" * 74)

    if args.mermaid:
        print("\n--- Mermaid Graph Syntax ---")
        print(kg.export_mermaid())
        print("----------------------------\n")

    if args.dot:
        print("\n--- Graphviz DOT Syntax ---")
        print(kg.export_dot())
        print("---------------------------\n")

    if args.export_json:
        out_p = Path(args.export_json)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(kg.to_dict(), f, indent=2)
        print(f"✅ Exported Knowledge Graph & Collisions JSON to {out_p}")

    if args.export_html:
        out_p = Path(args.export_html)
        kg.export_html_dossier(out_p)
        print(f"✅ Exported Interactive HTML Clinical Review Dossier to {out_p}")

    return 0


def cmd_calibrate_tiers(args) -> int:
    path = Path(args.path)
    cdus = load_cdus_from_file(path)

    kg = ClinicalKnowledgeGraph()
    kg.populate_library(cdus)
    audits = kg.audit_tier_calibrations()

    print(f"\n==========================================================================")
    print(f"  a-CDU Priority Tier Clinical Calibration Audit")
    print(f"  Target: {path} ({len(cdus)} units loaded)")
    print(f"==========================================================================\n")

    upgrades = 0
    aligned = 0
    cdu_map = {c.id: c for c in cdus}

    for a in audits:
        status_icon = "🟢" if a["status"] == "ALIGNED" else "🛑"
        print(f"{status_icon} [{a['status']:<19}] {a['cdu_id']:<32} | Current: T{a['current_tier']} -> Warranted: T{a['warranted_tier']}")
        if a["status"] != "ALIGNED":
            print(f"    👉 {a['recommendation']}")
            upgrades += 1
            if args.apply:
                cdu_map[a["cdu_id"]].priority_tier = a["warranted_tier"]
                cdu_map[a["cdu_id"]].safety_tier_rationale = a["recommendation"]
        else:
            aligned += 1

    print("\n" + "=" * 74)
    print(f"  AUDIT SUMMARY: {aligned} Aligned, {upgrades} Upgrades Recommended")
    print("=" * 74 + "\n")

    if args.apply:
        out_path = Path(args.output) if args.output else path
        serialized = [c.to_dict() for c in cdus]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        print(f"✅ Applied {upgrades} Priority Tier calibrations to {out_path}!\n")

    return 0


def cmd_view(args) -> int:
    import http.server
    import socketserver
    import webbrowser
    import threading

    port = args.port
    path = Path(args.path)
    cdus = load_cdus_from_file(path)

    # Ensure dist exists
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    html_path = dist_dir / "clinical_review_dossier.html"

    # Always generate freshest dossier
    kg = ClinicalKnowledgeGraph()
    kg.populate_library(cdus)
    kg.export_html_dossier(html_path)

    # Also copy to root for GitHub Pages / direct link
    shutil.copy2(html_path, "clinical-review-dossier.html")

    url = f"http://localhost:{port}/dist/clinical_review_dossier.html"
    print(f"\n==========================================================================")
    print(f"  🛡️ a-CDU Knowledge Graph & Clinical Safety Studio Viewer")
    print(f"  Serving interactive viewer at: {url}")
    print(f"  Local file path: {html_path.resolve()}")
    print(f"  Press Ctrl+C to stop the local viewer.")
    print(f"==========================================================================\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Viewer server stopped.")
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

    # Subcommand: graph-collisions
    p_graph = subparsers.add_parser("graph-collisions", help="Build knowledge graph & detect multi-morbidity collisions")
    p_graph.add_argument("path", type=str, nargs="?", default="dataset/phc_decision_units.json", help="Path to CDU JSON or directory")
    p_graph.add_argument("--export-json", type=str, default="", help="Path to export graph and collisions JSON")
    p_graph.add_argument("--export-html", type=str, default="", help="Path to export interactive HTML review dossier")
    p_graph.add_argument("--mermaid", action="store_true", default=False, help="Print Mermaid diagram to stdout")
    p_graph.add_argument("--dot", action="store_true", default=False, help="Print Graphviz DOT to stdout")
    p_graph.set_defaults(func=cmd_graph_collisions)

    # Subcommand: calibrate-tiers
    p_cal = subparsers.add_parser("calibrate-tiers", help="Audit & calibrate Priority Tiers against knowledge graph hazards")
    p_cal.add_argument("path", type=str, nargs="?", default="dataset/phc_decision_units.json", help="Path to CDU JSON")
    p_cal.add_argument("--apply", action="store_true", default=False, help="Apply calibrated tiers to JSON file")
    p_cal.add_argument("--output", type=str, default="", help="Output path if saving calibrated CDUs")
    p_cal.set_defaults(func=cmd_calibrate_tiers)

    # Subcommand: view
    p_view = subparsers.add_parser("view", help="Launch interactive web viewer for Knowledge Graph & Collisions")
    p_view.add_argument("path", type=str, nargs="?", default="dataset/phc_decision_units.json", help="Path to CDU JSON")
    p_view.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    p_view.add_argument("--no-browser", action="store_true", default=False, help="Do not open browser automatically")
    p_view.set_defaults(func=cmd_view)

    args = parser.parse_args()
    return args.func(args)
