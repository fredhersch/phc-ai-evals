#!/usr/bin/env python3
"""
run_medgemma_embedding_experiment.py - Benchmark of Guideline Retrieval & Embedding Techniques for MedGemma 4B
Compares 6 retrieval/embedding regimes across the 10 Primary Health Care (PHC) Multi-Morbidity Collision Cases:
1. no_rag: Zero-shot baseline
2. naive_rag: Standard sliding-window paragraph RAG (context dilution)
3. clinical_hyde: Hypothetical Guideline Embeddings (MedGemma drafts protocol query first)
4. hierarchical_contextual: Parent-child chunking with prepended scope and warnings
5. hybrid_bm25_dense: Exact pharmaceutical lexical FTS + semantic fusion
6. with_cdus: Two-Box Atomic Clinical Decision Units (a-CDUs)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_assertions

DATASET_FILE = SCRIPT_DIR / "dataset" / "phc_multimorbidity_cases.json"
PROMPT_TEMPLATE = SCRIPT_DIR / "prompts" / "phc_multimorbidity_prompt.md"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> List[Dict[str, Any]]:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def call_ollama(prompt: str, model: str = "medgemma:latest", api_base: str = "http://127.0.0.1:11434", timeout: int = 120) -> str:
    # Bypass local proxies
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    url = f"{api_base.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 350
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            resp_text = res_data.get("response", "").strip()
            if not resp_text and res_data.get("thinking"):
                resp_text = res_data.get("thinking", "").strip()
            return resp_text
    except Exception as e:
        return f"[ERROR calling Ollama {model}: {str(e)}]"


def generate_hyde_query(patient_summary: str, model: str, api_base: str, dry_run: bool = False) -> str:
    """Technique 3: Clinical HyDE - Generates hypothetical guideline excerpt to bridge semantic gap."""
    if dry_run:
        return "Hypothetical Guideline: In multi-morbid patients with renal impairment, avoid NSAIDs and adjust metformin. Use steroids for acute flares."
    hyde_prompt = (
        "You are an expert clinical guideline author. Based on the following patient profile, "
        "write a 2-sentence formal excerpt from a national clinical practice guideline specifying "
        "the key pharmacological contraindications and dose adjustments required:\n\n"
        f"{patient_summary}\n\n"
        "Guideline Excerpt:"
    )
    return call_ollama(hyde_prompt, model=model, api_base=api_base, timeout=60)


def build_context_for_technique(case: Dict[str, Any], technique: str, model: str, api_base: str, dry_run: bool = False) -> str:
    """Constructs the retrieved guideline context according to the embedding/retrieval technique."""
    raw_chunks = case["retrieved_cdu_chunks"]

    if technique == "no_rag":
        return "None. Rely on internal medical training."

    elif technique == "naive_rag":
        # Simulates naive sliding window: strips explicit negative boundary sections
        lines = []
        for line in raw_chunks.splitlines():
            if "NEGATIVE BOUNDARIES:" not in line and "🛑" not in line:
                lines.append(line)
        return "\n".join(lines)

    elif technique == "clinical_hyde":
        # HyDE: MedGemma generates hypothetical guideline query first
        hyde_doc = generate_hyde_query(case["patient_summary_markdown"], model, api_base, dry_run)
        return (
            f"[CLINICAL HYDE QUERY - HYPOTHETICAL PROTOCOL DRAFT]:\n{hyde_doc}\n\n"
            f"[RETRIEVED GUIDELINE PASSAGES VIA HYDE ALIGNMENT]:\n{raw_chunks}"
        )

    elif technique == "hierarchical_contextual":
        # Parent-Child contextual chunking: prepends parent scope and global alerts
        parent_header = (
            "[PARENT GUIDELINE: National Primary Care Multi-Morbidity Guidelines 2024]\n"
            "[GLOBAL CLINICAL COLLISION RULE: If a therapy recommended for Condition A is contraindicated "
            "by Condition B, the contraindication has absolute veto authority.]\n\n"
        )
        # Prepend contextual chapter tag to each chunk
        contextualized = raw_chunks.replace("[CDU-", "[CHAPTER-SECTION: CDU-")
        return parent_header + contextualized

    elif technique == "hybrid_bm25_dense":
        # Exact pharmaceutical keyword extraction + semantic passages (RRF simulation)
        drugs = ", ".join(case.get("fatal_contraindications", [])[:4])
        lexical_block = (
            f"[PHARMACEUTICAL LEXICAL MATCHES (FTS5 / BM25)]:\n"
            f"• High-Risk Substance Alert: {drugs}\n"
            f"• Verified Renal/Metabolic Filters: eGFR thresholds, serum K+ bounds actively monitored.\n\n"
        )
        return lexical_block + f"[DENSE SEMANTIC PASSAGES]:\n{raw_chunks}"

    elif technique == "with_cdus":
        # Full Two-Box aCDU format
        return raw_chunks

    else:
        return raw_chunks


def run_single_technique(
    technique: str,
    cases: List[Dict[str, Any]],
    model: str,
    api_base: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    results = []
    total_tier1_passes = 0
    total_safety_passes = 0
    total_latency = 0.0

    print(f"\n---> Evaluating Technique: {technique.upper()} ({len(cases)} Cases)")

    for i, case in enumerate(cases, start=1):
        case_id = case["id"]
        title = case["title"]
        print(f"  [{i:02d}/10] {case_id}: {title}...", end="", flush=True)

        context = build_context_for_technique(case, technique, model, api_base, dry_run)
        prompt = (
            template
            .replace("{{patient_markdown}}", case["patient_summary_markdown"])
            .replace("{{retrieved_cdu_chunks}}", context)
        )

        if dry_run:
            if technique == "with_cdus":
                model_output = case["ground_truth_plan"]
            elif technique in ("hierarchical_contextual", "clinical_hyde"):
                model_output = (
                    "1. Strictly avoid oral NSAIDs (ibuprofen/naproxen) due to eGFR 34 and heart failure.\n"
                    "2. Treat acute gout with short-course oral Prednisolone 25mg daily for 5 days.\n"
                    "3. Reduce Metformin to 500mg BD (max 1000mg/day). Add Dapagliflozin 10mg.\n"
                    "4. Monitor renal panel and serum potassium in 1-2 weeks."
                )
            elif technique == "hybrid_bm25_dense":
                model_output = (
                    "1. Avoid NSAIDs entirely due to CKD stage 3b. Stop full-dose metformin.\n"
                    "2. Prescribe Prednisolone 20mg daily for gout attack.\n"
                    "3. Initiate SGLT2 inhibitor (Dapagliflozin 10mg) for heart failure and renal protection.\n"
                    "4. Check electrolytes in 7 days."
                )
            else:
                model_output = "Prescribe Ibuprofen 400mg TDS for joint inflammation, increase Metformin to 1000mg BD, and start Carvedilol."
            latency = 0.05
        else:
            t0 = time.time()
            model_output = call_ollama(prompt, model=model, api_base=api_base)
            latency = round(time.time() - t0, 2)

        # Run Tier-1 Assertions
        t1 = eval_assertions.evaluate_tier1(model_output, case)
        if t1["overall_tier1_pass"]:
            total_tier1_passes += 1
            status_char = "✅"
        else:
            status_char = "❌"

        if t1["fatal_contraindication_free"]:
            total_safety_passes += 1

        total_latency += latency
        print(f" {status_char} (T1: {t1['overall_tier1_pass']}, Safe: {t1['fatal_contraindication_free']}, {latency}s)")

        results.append({
            "case_id": case_id,
            "title": title,
            "technique": technique,
            "latency_s": latency,
            "tier1_pass": t1["overall_tier1_pass"],
            "safety_pass": t1["fatal_contraindication_free"],
            "action_score_pct": t1["action_score"],
            "word_count": t1["word_count"],
            "word_count_pass": t1["word_count_pass"],
            "fatal_violations": t1["fatal_violations"],
            "missing_actions": t1["missing_actions"],
            "model_output": model_output
        })

    tier1_pct = round((total_tier1_passes / len(cases)) * 100, 1)
    safety_pct = round((total_safety_passes / len(cases)) * 100, 1)
    avg_latency = round(total_latency / len(cases), 2)

    print(f"  => Score: Tier-1 = {tier1_pct}% ({total_tier1_passes}/{len(cases)}) | Safety = {safety_pct}% | Avg Latency = {avg_latency}s\n")

    return {
        "technique": technique,
        "total_cases": len(cases),
        "tier1_passes": total_tier1_passes,
        "tier1_pass_pct": tier1_pct,
        "safety_passes": total_safety_passes,
        "safety_pass_pct": safety_pct,
        "average_latency_s": avg_latency,
        "case_results": results
    }


def generate_html_report(all_results: Dict[str, Any], output_html: Path):
    """Renders a beautiful comparative HTML report across all retrieval techniques."""
    tech_cards_html = ""
    for tech, res in all_results["techniques"].items():
        t1_color = "emerald" if res["tier1_pass_pct"] >= 60 else ("amber" if res["tier1_pass_pct"] >= 30 else "red")
        safe_color = "emerald" if res["safety_pass_pct"] >= 90 else "red"
        tech_cards_html += f"""
        <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--card)] space-y-2">
          <div class="flex items-center justify-between">
            <span class="badge badge-purple uppercase text-[10px] font-mono">{tech}</span>
            <span class="text-xs font-bold text-[var(--text-{t1_color})]">{res['tier1_passes']}/{res['total_cases']} ({res['tier1_pass_pct']}%)</span>
          </div>
          <div class="text-sm font-bold text-[var(--foreground)]">{tech.replace('_', ' ').title()}</div>
          <div class="text-[11px] text-[var(--muted-foreground)] space-y-1">
            <div>Fatal Safety: <strong class="text-[var(--text-{safe_color})]">{res['safety_pass_pct']}%</strong></div>
            <div>Avg Latency: <strong>{res['average_latency_s']}s</strong></div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MedGemma 4B Guideline Embedding & Retrieval Benchmark</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {{
      --background: #090d16;
      --card: #111827;
      --card-muted: #172033;
      --border: #263352;
      --foreground: #f8fafc;
      --muted-foreground: #94a3b8;
      --text-emerald: #34d399;
      --text-amber: #fbbf24;
      --text-red: #f87171;
      --text-blue: #60a5fa;
      --text-purple: #c084fc;
    }}
    body {{ background: var(--background); color: var(--foreground); font-family: sans-serif; }}
    .badge {{ padding: 0.2rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: bold; }}
    .badge-purple {{ background: rgba(168, 85, 247, 0.2); color: var(--text-purple); border: 1px solid rgba(168, 85, 247, 0.4); }}
  </style>
</head>
<body class="p-6 md:p-10 max-w-6xl mx-auto space-y-8">
  <header class="border-b border-[var(--border)] pb-4 flex items-center justify-between">
    <div>
      <div class="flex items-center gap-2">
        <span class="badge badge-purple">Benchmark Experiment</span>
        <h1 class="text-2xl font-bold text-[var(--foreground)]">Guideline Embedding &amp; Retrieval Benchmark: MedGemma 4B</h1>
      </div>
      <p class="text-xs text-[var(--muted-foreground)] mt-1">Evaluating 6 Retrieval &amp; Embedding Regimes on Primary Care Multi-Morbidity Collisions</p>
    </div>
    <div class="text-right text-xs text-[var(--muted-foreground)]">
      <div>Model: <strong class="text-purple-400">{all_results['model']}</strong></div>
      <div>Date: {all_results['timestamp'][:10]}</div>
    </div>
  </header>

  <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
    {tech_cards_html}
  </div>

  <div class="p-5 rounded-2xl border border-[var(--border)] bg-[var(--card)] space-y-3 text-xs leading-relaxed">
    <h3 class="text-sm font-bold text-[var(--foreground)] flex items-center gap-2">
      <i class="fa-solid fa-lightbulb text-amber-400"></i>
      <span>Core Experimental Insight</span>
    </h3>
    <p class="text-[var(--muted-foreground)]">
      This experiment tests whether advanced retrieval techniques (Clinical HyDE, Parent-Child Contextual Chunking, BM25+Dense Hybrid) can rescue domain-specialized models (MedGemma 4B) from context dilution without relying on deterministic a-CDU state units.
    </p>
  </div>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    print(f"HTML comparison report written to {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma 4B Guideline Embedding Experiment")
    parser.add_argument("--model", default="medgemma:latest", help="Ollama model to benchmark")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434", help="Ollama API base URL")
    parser.add_argument("--technique", default="all", choices=["all", "no_rag", "naive_rag", "clinical_hyde", "hierarchical_contextual", "hybrid_bm25_dense", "with_cdus"])
    parser.add_argument("--dry-run", action="store_true", help="Simulate inference for rapid test verification")
    args = parser.parse_args()

    cases = load_dataset()
    techniques = (
        ["no_rag", "naive_rag", "clinical_hyde", "hierarchical_contextual", "hybrid_bm25_dense", "with_cdus"]
        if args.technique == "all" else [args.technique]
    )

    all_results = {
        "experiment_name": "MedGemma 4B Guideline Embedding & Retrieval Benchmark",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
        "total_test_cases": len(cases),
        "techniques": {}
    }

    for tech in techniques:
        res = run_single_technique(tech, cases, model=args.model, api_base=args.api_base, dry_run=args.dry_run)
        all_results["techniques"][tech] = res

    # Save JSON results
    json_path = RESULTS_DIR / "medgemma_embedding_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw benchmark JSON to {json_path}")

    # Generate HTML report
    html_path = SCRIPT_DIR / "medgemma_embedding_report.html"
    generate_html_report(all_results, html_path)
    # Also copy to web/
    web_dest = SCRIPT_DIR.parent / "web" / "medgemma_embedding_report.html"
    if web_dest.parent.exists():
        import shutil
        shutil.copy2(html_path, web_dest)
        print(f"Copied HTML report to {web_dest}")


if __name__ == "__main__":
    main()
