#!/usr/bin/env python3
"""
run_phc_evals.py - Benchmark Runner for Primary Health Care (PHC) Multi-Morbidity Decision Support
Evaluates on-device SLMs (Gemma 2B, 4B, MedGemma, 27B) against a-CDU Composite RAG vs Baselines.
Supports Frontier LLM Judge (Gemini) and Tier-1 Deterministic Clinical Safety Assertions.
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

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_assertions

DATASET_FILE = SCRIPT_DIR / "dataset" / "phc_multimorbidity_cases.json"
PROMPT_TEMPLATE = SCRIPT_DIR / "prompts" / "phc_multimorbidity_prompt.md"
JUDGE_TEMPLATE = SCRIPT_DIR / "judge_prompts" / "phc_llm_judge.md"

# Auto-load .env if present
_env_file = SCRIPT_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def load_dataset() -> List[Dict[str, Any]]:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_FILE}")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def call_ollama(prompt: str, model: str, api_base: str, timeout: int = 180) -> str:
    url = f"{api_base.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 1200,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            resp_text = res_data.get("response", "").strip()
            # If response is empty but thinking exists, use thinking or log
            if not resp_text and res_data.get("thinking"):
                resp_text = res_data.get("thinking", "").strip()
            return resp_text
    except Exception as e:
        return f"[ERROR calling Ollama {model}: {str(e)}]"


def call_cloud_api(prompt: str, model: str, api_base: str, api_key: str, timeout: int = 120) -> str:
    """Invokes any OpenAI-compatible cloud endpoint (Groq, OpenRouter, Together AI, vLLM)."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/fredhersch/phc-ai-evals",
        "User-Agent": "PHC-AI-Evals/1.0"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR calling Cloud API {model}: {str(e)}]"


def call_gemini_judge(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(raw_text)
    except Exception as e:
        return {
            "safety_verdict": "ERROR",
            "clinical_rationale": f"Judge invocation failed: {str(e)}",
            "scores": {"composite_quality_score": 0.0}
        }


def run_benchmark(
    model: str,
    api_base: str,
    cdu_mode: str,
    judge_model: str = "",
    judge_api_key: str = "",
    cloud_api_base: str = "",
    cloud_api_key: str = "",
    dry_run: bool = False,
    output_path: str = ""
) -> Dict[str, Any]:
    cases = load_dataset()
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    judge_template_text = JUDGE_TEMPLATE.read_text(encoding="utf-8") if judge_model else ""
    
    results = []
    total_tier1_passes = 0
    total_safety_passes = 0
    
    print(f"\n==========================================================================")
    print(f"  PHC Multi-Morbidity Decision Support Benchmark (10 Cases)")
    print(f"  Model Under Test : {model} (Mode: {cdu_mode})")
    print(f"  Frontier Judge   : {judge_model or 'None (Tier-1 Assertions Only)'}")
    print(f"  Dry Run Active   : {dry_run}")
    print(f"==========================================================================\n")
    
    for i, case in enumerate(cases, start=1):
        case_id = case["id"]
        title = case["title"]
        print(f"[{i:02d}/10] Evaluating {case_id}: {title}...")
        
        # Prepare context based on cdu_mode
        if cdu_mode == "with_cdus":
            cdu_chunks = case["retrieved_cdu_chunks"]
        elif cdu_mode == "naive_rag":
            # Naive sliding window without negative boundaries or atomic structure
            cdu_chunks = case["retrieved_cdu_chunks"].split("NEGATIVE BOUNDARIES:")[0]
        else: # no_rag
            cdu_chunks = "None. Rely on internal medical training."
            
        prompt = (
            template
            .replace("{{patient_markdown}}", case["patient_summary_markdown"])
            .replace("{{retrieved_cdu_chunks}}", cdu_chunks)
        )
        
        # Generate model response
        if dry_run:
            # In dry-run, use ground truth for with_cdus, or simulated naive answer
            if cdu_mode == "with_cdus":
                model_output = case["ground_truth_plan"]
            else:
                model_output = "Prescribe Ibuprofen 400mg TDS for joint inflammation, increase Metformin to 1000mg BD, and start Carvedilol."
            gen_time = 0.05
        else:
            t0 = time.time()
            if cloud_api_key and cloud_api_base:
                model_output = call_cloud_api(prompt, model, cloud_api_base, cloud_api_key)
            else:
                model_output = call_ollama(prompt, model, api_base)
            gen_time = round(time.time() - t0, 2)
            
        # Tier-1 Assertions
        t1_results = eval_assertions.evaluate_tier1(model_output, case)
        if t1_results["overall_tier1_pass"]:
            total_tier1_passes += 1
        if t1_results["fatal_contraindication_free"]:
            total_safety_passes += 1
            
        # Tier-2 Judge Evaluation
        judge_res = None
        if judge_model and judge_api_key and not dry_run:
            judge_prompt = (
                judge_template_text
                .replace("{{case_id}}", case_id)
                .replace("{{case_title}}", title)
                .replace("{{patient_summary}}", case["patient_summary_markdown"])
                .replace("{{clinical_tension}}", case["clinical_tension"])
                .replace("{{ground_truth_plan}}", case["ground_truth_plan"])
                .replace("{{candidate_output}}", model_output)
            )
            judge_res = call_gemini_judge(judge_prompt, judge_api_key, model=judge_model)
            
        case_record = {
            "case_id": case_id,
            "title": title,
            "cdu_mode": cdu_mode,
            "latency_seconds": gen_time,
            "model_output": model_output,
            "tier1_eval": t1_results,
            "judge_eval": judge_res
        }
        results.append(case_record)
        
        # Line summary
        status_icon = "✅ PASS" if t1_results["overall_tier1_pass"] else "❌ FAIL"
        safety_icon = "🛡️ SAFE" if t1_results["fatal_contraindication_free"] else f"🛑 FATAL ({', '.join(t1_results['fatal_violations'])})"
        print(f"      -> {status_icon} | {safety_icon} | Action Match: {t1_results['action_score']}% | Words: {t1_results['word_count']}")
        
    print(f"\n==========================================================================")
    print(f"  FINAL SUMMARY: {model} ({cdu_mode})")
    print(f"  Overall Tier-1 Pass Rate : {total_tier1_passes}/10 ({total_tier1_passes * 10}%)")
    print(f"  Fatal Safety Adherence   : {total_safety_passes}/10 ({total_safety_passes * 10}%)")
    print(f"==========================================================================\n")
    
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "cdu_mode": cdu_mode,
        "dry_run": dry_run,
        "total_cases": 10,
        "overall_tier1_passes": total_tier1_passes,
        "fatal_safety_passes": total_safety_passes,
        "tier1_pass_percentage": total_tier1_passes * 10.0,
        "safety_adherence_percentage": total_safety_passes * 10.0,
        "results": results
    }
    
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved full benchmark report to: {out_file}\n")
        
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHC Multi-Morbidity Decision Support Evals")
    parser.add_argument("--model", type=str, default="gemma2:2b-instruct-q4_K_M", help="Model name in Ollama")
    parser.add_argument("--api-base", type=str, default="http://localhost:11434", help="Ollama API base URL")
    parser.add_argument("--cdu-mode", type=str, choices=["with_cdus", "naive_rag", "no_rag"], default="with_cdus", help="RAG conditioning mode")
    parser.add_argument("--cloud-api-base", type=str, default=os.environ.get("CLOUD_API_BASE", ""), help="OpenAI-compatible cloud API base (e.g., https://openrouter.ai/api/v1)")
    parser.add_argument("--cloud-api-key", type=str, default=os.environ.get("CLOUD_API_KEY", ""), help="API key for cloud endpoint")
    parser.add_argument("--judge-model", type=str, default="", help="Frontier Judge model (e.g. gemini-2.0-flash, gemini-1.5-pro)")
    parser.add_argument("--judge-api-key", type=str, default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API Key")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without external model calls")
    parser.add_argument("--output", type=str, default="", help="Path to save output JSON")
    
    args = parser.parse_args()
    
    if not args.output:
        mode_slug = args.cdu_mode
        model_slug = args.model.replace(":", "_").replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = str(SCRIPT_DIR / "results" / f"phc_eval_{model_slug}_{mode_slug}_{timestamp}.json")
        
    run_benchmark(
        model=args.model,
        api_base=args.api_base,
        cdu_mode=args.cdu_mode,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        cloud_api_base=args.cloud_api_base,
        cloud_api_key=args.cloud_api_key,
        dry_run=args.dry_run,
        output_path=args.output
    )
