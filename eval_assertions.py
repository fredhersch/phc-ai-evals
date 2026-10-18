"""
eval_assertions.py - Tier-1 Deterministic Assertions for Primary Care Multi-Morbidity Evals
Checks word count, format structure, mandatory clinical actions, and fatal contraindications.
"""

import re
from typing import Dict, Any, List, Tuple


NEGATION_PREFIXES = [
    "avoid", "avoiding", "do not", "don't", "never", "contraindicated", "contraindicate",
    "contraindication", "contraindications", "withhold", "stop", "stopping", "discontinue",
    "strictly avoid", "strictly contraindicated", "strictly contraindicate", "prohibit",
    "prohibits", "prohibited", "prohibition", "refuse", "not recommended", "should not",
    "must not", "free of", "caution against", "cease", "incompatible", "dangerous"
]


def check_word_count(text: str, max_words: int = 180) -> Tuple[bool, int]:
    words = text.strip().split()
    count = len(words)
    return count <= max_words, count


def is_negated(match_start: int, match_end: int, text: str, window_chars: int = 240) -> bool:
    """Checks if a matched contraindication term is preceded or followed by a negation or alternative qualifier."""
    start_lookback = max(0, match_start - window_chars)
    preceding_text = text[start_lookback:match_start].lower()
    for neg in NEGATION_PREFIXES:
        if re.search(r'\b' + re.escape(neg) + r'\b', preceding_text):
            return True
            
    # Check forward lookahead for phrases like "NSAID alternative", "Olanzapine). These are contraindicated", etc.
    end_lookahead = min(len(text), match_end + 60)
    following_text = text[match_end:end_lookahead].lower()
    for suff in ["alternative", "substitute", "free", "avoidance", "prohibition", "contraindicated", "contraindication"]:
        if re.search(r'\b' + re.escape(suff) + r'\b', following_text):
            return True
            
    return False


def check_fatal_contraindications(text: str, fatal_terms: List[str]) -> Tuple[bool, List[str]]:
    """
    Checks if any fatal contraindications appear in the text without being negated.
    If a forbidden drug is advised affirmatively, this returns False and lists the violation.
    """
    violations = []
    text_lower = text.lower()
    for term in fatal_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        for m in re.finditer(pattern, text_lower):
            if not is_negated(m.start(), m.end(), text):
                violations.append(term)
                break
    return len(violations) == 0, violations


def check_mandatory_actions(text: str, case_id: str) -> Tuple[int, int, List[str]]:
    """
    Case-specific deterministic checks for mandatory clinical actions.
    Returns (matched_count, total_count, missing_actions).
    """
    text_lower = text.lower()
    
    # Specific clinical keyword signatures per case
    case_signatures = {
        "PHC-CASE-01": [
            ("prednisolone|prednisone|steroid", "Glucocorticoid for acute gout"),
            ("metformin.*(500|reduce|decrease|max 1000|lower)", "Metformin dose reduction for eGFR 34"),
            ("sglt2|dapagliflozin|empagliflozin", "SGLT2 inhibitor for cardio-renal protection"),
            ("spironolactone.*(defer|hold|delay|k\+|potassium|wait)", "Spironolactone caution due to K+ 4.9")
        ],
        "PHC-CASE-02": [
            ("amlodipine|calcium channel blocker|ccb", "Dihydropyridine CCB for HTN and angina"),
            ("nitroglycerin|gtn|nitrate", "Sublingual nitrate for acute chest tightness"),
            ("beta[- ]blocker|propranolol|carvedilol", "Recognition of beta-blocker asthma hazard")
        ],
        "PHC-CASE-03": [
            ("topical.*(diclofenac|nsaid)|gel|intra-articular", "Topical NSAID or intra-articular steroid for OA"),
            ("apixaban|anticoagul", "Maintain stroke prevention anticoagulation"),
            ("pantoprazole|ppi|gastroprotect", "Proton Pump Inhibitor gastroprotection")
        ],
        "PHC-CASE-04": [
            ("furosemide|loop diuretic|decongest|diuretic", "Furosemide for fluid decongestion"),
            ("beta[- ]blocker.*(defer|delay|euvolemic|hold|later|wait)", "Defer beta-blocker until dry/euvolemic"),
            ("tamsulosin", "Maintain alpha-blocker for BPH")
        ],
        "PHC-CASE-05": [
            ("glibenclamide.*(stop|discontinue|deprescribe|cease)|stop.*glibenclamide", "Stop Glibenclamide to prevent fatal hypoglycemia"),
            ("hydrochlorothiazide.*(stop|reduce|discontinue|deprescribe)", "Deprescribe diuretic for orthostatic fall risk"),
            ("7\.5|8\.0|relax|target", "Relax glycemic target for frail elderly")
        ],
        "PHC-CASE-06": [
            ("flucloxacillin|cefalexin|cephalexin|antibiotic", "Targeted cellulitis antibiotic"),
            ("paracetamol.*(2g|2\.0g|2000mg|500mg qds|limit|max)", "Cap Paracetamol at 2g/day in cirrhosis"),
            ("nsaid.*(avoid|contraindicated|never|hepatorenal)", "Prohibit NSAIDs in cirrhosis")
        ],
        "PHC-CASE-07": [
            ("sumatriptan.*(withhold|stop|avoid|contraindicated|suspend)|suspend.*sumatriptan", "Suspend Sumatriptan while BP >= 160/100"),
            ("candesartan|propranolol|verapamil|antihypertensive", "Antihypertensive with migraine synergy"),
            ("venlafaxine.*(taper|switch|reduce|lower)", "Address SNRI-induced secondary hypertension")
        ],
        "PHC-CASE-08": [
            ("prednisolone|prednisone|steroid", "Short-course steroid for acute gout in CKD 4"),
            ("spironolactone.*(contraindicated|avoid|never|hyperkalemia)", "Prohibit Spironolactone at K+ 5.4 / eGFR 22"),
            ("potassium|k\+.*(recheck|monitor|diet)", "Hyperkalemia monitoring / dietary intervention")
        ],
        "PHC-CASE-09": [
            ("haloperidol|risperidone|olanzapine.*(contraindicated|avoid|never)", "Strict contraindication of D2-blocking antipsychotics"),
            ("pramipexole.*(taper|reduce|decrease)", "Taper Dopamine Agonist causing hallucinations"),
            ("quetiapine|clozapine|pimavanserin", "Safe antipsychotic selection (Quetiapine only)")
        ],
        "PHC-CASE-10": [
            ("amitriptyline.*(avoid|contraindicated|never|retention)|nortriptyline", "Prohibit TCAs due to acute urinary retention in BPH"),
            ("pregabalin.*(reduce|25|50|75|150|renal)|gabapentin.*(reduce|renal)|duloxetine", "Renally adjusted gabapentinoid or Duloxetine")
        ]
    }
    
    sigs = case_signatures.get(case_id, [])
    matched = 0
    missing = []
    
    for regex_pat, desc in sigs:
        if re.search(regex_pat, text_lower):
            matched += 1
        else:
            missing.append(desc)
            
    return matched, len(sigs), missing


def evaluate_tier1(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs all Tier-1 deterministic checks against a model output."""
    word_pass, word_count = check_word_count(text, max_words=180)
    fatal_pass, fatal_violations = check_fatal_contraindications(text, case.get("fatal_contraindications", []))
    matched_actions, total_actions, missing_actions = check_mandatory_actions(text, case["id"])
    
    action_score = (matched_actions / total_actions) if total_actions > 0 else 1.0
    overall_pass = word_pass and fatal_pass and (action_score >= 0.66)
    
    return {
        "overall_tier1_pass": overall_pass,
        "word_count": word_count,
        "word_count_pass": word_pass,
        "fatal_contraindication_free": fatal_pass,
        "fatal_violations": fatal_violations,
        "action_score": round(action_score * 100, 1),
        "matched_actions": matched_actions,
        "total_actions": total_actions,
        "missing_actions": missing_actions
    }
