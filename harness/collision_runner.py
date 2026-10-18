"""
collision_runner.py - Automated Adversarial Multi-Morbidity Collision Testbed
Executes collision regression batteries against local or cloud SLMs to guarantee
zero fatal contraindication leaks when new a-CDUs are added.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .models import ClinicalDecisionUnit
import eval_assertions


def run_collision_test_suite(
    cases_path: Path,
    cdu_library: List[ClinicalDecisionUnit],
    dry_run: bool = True,
    model: str = "medgemma:latest",
    api_base: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Executes all multi-morbidity cases against the active CDU library.
    In dry-run mode, uses ground truth assertions to verify harness integrity.
    """
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found at {cases_path}")

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Build lookup map for CDUs
    cdu_map = {c.id: c for c in cdu_library}

    results = []
    total_safety_passes = 0
    total_tier1_passes = 0

    print(f"\n[a-CDU Collision Harness] Executing {len(cases)} Multi-Morbidity Regression Cases...")
    print(f"  Target Model : {model} (Dry-Run: {dry_run})")
    print(f"  Active CDUs  : {len(cdu_library)} loaded\n" + "-" * 70)

    for idx, case in enumerate(cases, start=1):
        case_id = case["id"]
        title = case["title"]

        # Formulate active CDU context from the library
        active_chunks = []
        for cid in case.get("retrieved_cdu_ids", []):
            if cid in cdu_map:
                active_chunks.append(cdu_map[cid].to_prompt_chunk())
            else:
                # Fallback to case provided chunk
                pass

        if not active_chunks and "retrieved_cdu_chunks" in case:
            prompt_cdu_text = case["retrieved_cdu_chunks"]
        else:
            prompt_cdu_text = "\n\n".join(active_chunks)

        if dry_run:
            model_output = case["ground_truth_plan"]
            latency = 0.01
        else:
            # Import runner call logic if live
            import run_phc_evals
            template = run_phc_evals.PROMPT_TEMPLATE.read_text(encoding="utf-8")
            prompt = (
                template
                .replace("{{patient_markdown}}", case["patient_summary_markdown"])
                .replace("{{retrieved_cdu_chunks}}", prompt_cdu_text)
            )
            t0 = time.time()
            model_output = run_phc_evals.call_ollama(prompt, model, api_base)
            latency = round(time.time() - t0, 2)

        # Run Tier-1 Assertions
        eval_res = eval_assertions.evaluate_tier1(model_output, case)
        if eval_res["fatal_contraindication_free"]:
            total_safety_passes += 1
        if eval_res["overall_tier1_pass"]:
            total_tier1_passes += 1

        status_sym = "✅ PASS" if eval_res["overall_tier1_pass"] else "❌ FAIL"
        safety_sym = "🛡️ SAFE" if eval_res["fatal_contraindication_free"] else f"🛑 FATAL: {eval_res['fatal_violations']}"

        print(f"[{idx:02d}/10] {case_id}: {title[:40]}... -> {status_sym} | {safety_sym}")

        results.append({
            "case_id": case_id,
            "title": title,
            "latency": latency,
            "eval": eval_res
        })

    safety_rate = (total_safety_passes / len(cases)) * 100
    pass_rate = (total_tier1_passes / len(cases)) * 100

    print("-" * 70)
    print(f"Collision Suite Results: Safety Adherence = {safety_rate:.1f}% | Tier-1 Pass = {pass_rate:.1f}%\n")

    return {
        "total_cases": len(cases),
        "safety_rate": safety_rate,
        "pass_rate": pass_rate,
        "all_safe": total_safety_passes == len(cases),
        "results": results
    }
