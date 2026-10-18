"""
linter.py - Static Linter & Semantic Safety Checker for a-CDUs
Enforces the 180-token budget ceiling, detects internal self-contradictions,
and checks ontology and provenance completeness.
"""

import re
from typing import List, Dict, Any, Tuple
from .models import ClinicalDecisionUnit


MAX_TOKEN_BUDGET = 180
WARNING_TOKEN_THRESHOLD = 150

ACTIONABLE_NEGATIVE_MARKERS = [
    "never", "do not", "don't", "strictly", "contraindicated", "contraindication",
    "contraindications", "prohibit", "prohibited", "avoid", "stop", "withhold",
    "discontinue", "absolutely no", "must not", "should not", "zero", "caution against"
]


class LintIssue:
    def __init__(self, level: str, message: str, field_name: str = ""):
        self.level = level  # "ERROR" or "WARNING"
        self.message = message
        self.field_name = field_name

    def __repr__(self):
        return f"[{self.level}] {self.field_name + ': ' if self.field_name else ''}{self.message}"


class LintResult:
    def __init__(self, cdu_id: str):
        self.cdu_id = cdu_id
        self.issues: List[LintIssue] = []
        self.token_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(i.level == "ERROR" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "WARNING" for i in self.issues)


def extract_forbidden_keywords(neg_text: str) -> List[str]:
    """Extracts candidate prohibited drugs or substance terms from a negative boundary."""
    terms = []
    # Match capitalized drug candidates or common pharmacological stems
    candidates = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]+(?:pril|sartan|olol|statin|cillin|floxacin|gliflozin|gliptin|tide|zole|pam|lam|pine)\b', neg_text)
    for c in candidates:
        if c.lower() not in ACTIONABLE_NEGATIVE_MARKERS and len(c) > 3:
            terms.append(c.lower())
    return list(set(terms))


def lint_cdu(cdu: ClinicalDecisionUnit) -> LintResult:
    """
    Performs static analysis on a single a-CDU.
    """
    res = LintResult(cdu.id)
    tokens = cdu.estimate_tokens()
    res.token_count = tokens

    # 1. Token Budget Clamp
    if tokens > MAX_TOKEN_BUDGET:
        res.issues.append(LintIssue(
            "ERROR",
            f"Token budget exceeded: {tokens} tokens > maximum limit of {MAX_TOKEN_BUDGET} tokens.",
            "token_budget"
        ))
    elif tokens > WARNING_TOKEN_THRESHOLD:
        res.issues.append(LintIssue(
            "WARNING",
            f"Token budget near capacity: {tokens} tokens (approaching {MAX_TOKEN_BUDGET}). Consider concise rephrasing.",
            "token_budget"
        ))

    # 2. Structural Requirements
    if not cdu.id or not cdu.id.startswith("CDU-"):
        res.issues.append(LintIssue("ERROR", "ID must follow convention 'CDU-[TOPIC]-[SPECIFIER]'", "id"))

    if not cdu.source_guideline or len(cdu.source_guideline.strip()) < 5:
        res.issues.append(LintIssue("ERROR", "Source guideline authority & provenance is missing or too vague.", "source_guideline"))

    if not cdu.entry_criteria or len(cdu.entry_criteria.strip()) < 8:
        res.issues.append(LintIssue("ERROR", "Clinical trigger condition (entry_criteria) is missing or too short.", "entry_criteria"))

    if len(cdu.mandated_actions) == 0:
        res.issues.append(LintIssue("ERROR", "At least one mandated therapeutic action must be defined.", "mandated_actions"))

    if len(cdu.negative_boundaries) == 0:
        res.issues.append(LintIssue("ERROR", "At least one explicit negative boundary must be defined.", "negative_boundaries"))

    if cdu.priority_tier not in [1, 2, 3]:
        res.issues.append(LintIssue("ERROR", "Priority tier must be 1 (Fatal/Toxicity), 2 (Organ), or 3 (Symptom).", "priority_tier"))

    # 3. Negative Boundary Clarity Check
    neg_combined = " ".join(cdu.negative_boundaries).lower()
    has_marker = any(m in neg_combined for m in ACTIONABLE_NEGATIVE_MARKERS)
    if not has_marker:
        res.issues.append(LintIssue(
            "WARNING",
            "Negative boundaries lack explicit prohibition markers (e.g., 'DO NOT', 'NEVER', 'STRICTLY PROHIBIT', 'ABSOLUTELY NO').",
            "negative_boundaries"
        ))

    # 4. Internal Self-Contradiction Check
    actions_combined = " ".join(cdu.mandated_actions).lower()
    forbidden_terms = extract_forbidden_keywords(neg_combined)
    for term in forbidden_terms:
        # Check if forbidden term appears affirmatively in mandated actions
        if re.search(r'\b' + re.escape(term) + r'\b', actions_combined):
            # Check if it was negated in actions as well
            if not any(m in actions_combined for m in ["avoid", "stop", "withhold", "do not"]):
                res.issues.append(LintIssue(
                    "ERROR",
                    f"Self-contradiction detected: Term '{term}' appears in negative boundaries but is recommended in mandated actions.",
                    "self_contradiction"
                ))

    return res


def lint_cdu_library(cdus: List[ClinicalDecisionUnit]) -> Tuple[bool, List[LintResult]]:
    """
    Lints a full list of CDUs and checks for duplicate IDs.
    """
    seen_ids = set()
    results = []
    overall_pass = True

    for cdu in cdus:
        r = lint_cdu(cdu)
        if cdu.id in seen_ids:
            r.issues.append(LintIssue("ERROR", f"Duplicate CDU ID detected: '{cdu.id}'", "id"))
        seen_ids.add(cdu.id)

        if not r.is_valid:
            overall_pass = False
        results.append(r)

    return overall_pass, results
