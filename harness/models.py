"""
models.py - Core Domain Data Models for Atomic Clinical Decision Units (a-CDUs)
Formalizes the 5-tuple: c = < I, C, A, N, P >
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import hashlib
import json
import re


@dataclass
class ClinicalDecisionUnit:
    """
    Formal 5-Tuple Atomic Clinical Decision Unit:
      c = < Identifier (I), TriggerCondition (C), MandatoryActions (A), NegativeBoundaries (N), PriorityTier (P) >
    """
    id: str                                  # I: Unique anchor (e.g., CDU-T2D-METFORMIN-RENAL)
    topic: str                               # Descriptive title
    source_guideline: str                    # Guideline authority and provenance
    entry_criteria: str                      # C: Structured clinical trigger condition
    mandated_actions: List[str]              # A: Mandatory clinical choices and dosing targets
    negative_boundaries: List[str]           # N: Hard safety overrides and contraindications
    priority_tier: int = 2                   # P: 1 = Toxic/Fatal override, 2 = Organ protection, 3 = Symptom
    domain: str = "primary_care_multimorbidity"
    verbatim_excerpt: str = ""
    danger_signs: List[str] = field(default_factory=list)
    certified: bool = False
    version: str = "1.0.0"
    approved_by: Optional[str] = None
    signature: Optional[str] = None

    def estimate_tokens(self) -> int:
        """
        Estimates BPE token count for edge models (Gemma/Llama heuristic: ~3.8 chars per token for clinical English).
        Also factors in special tokens and punctuation overhead.
        """
        text = self.to_prompt_chunk()
        # Word and punctuation based estimation
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[^\w\s]', text))
        char_est = len(text) / 3.8
        return int(max(words + (punctuation * 0.5), char_est))

    def to_prompt_chunk(self) -> str:
        """
        Renders the unit into the exact token-budgeted prompt injection format.
        """
        actions_str = " ".join(self.mandated_actions)
        neg_str = " ".join(self.negative_boundaries)
        return (
            f"[{self.id}]\n"
            f"CONDITION: {self.entry_criteria}\n"
            f"GUIDELINE ACTIONS: {actions_str}\n"
            f"NEGATIVE BOUNDARIES: {neg_str}"
        )

    def compute_hash(self) -> str:
        """
        Generates a deterministic SHA256 checksum of the unit's clinical content for integrity checking.
        """
        canonical = json.dumps({
            "id": self.id,
            "entry_criteria": self.entry_criteria,
            "mandated_actions": self.mandated_actions,
            "negative_boundaries": self.negative_boundaries,
            "priority_tier": self.priority_tier,
            "source_guideline": self.source_guideline,
            "version": self.version
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prompt_injection_chunk"] = self.to_prompt_chunk()
        d["content_hash"] = self.compute_hash()
        d["estimated_tokens"] = self.estimate_tokens()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalDecisionUnit":
        return cls(
            id=data.get("id", data.get("cdu_id", "")),
            topic=data.get("topic", ""),
            source_guideline=data.get("source_guideline", data.get("guideline_provenance", "")),
            entry_criteria=data.get("entry_criteria", data.get("condition", "")),
            mandated_actions=list(data.get("mandated_actions", data.get("actions", []))),
            negative_boundaries=list(data.get("negative_boundaries", data.get("prohibitions", []))),
            priority_tier=int(data.get("priority_tier", 2)),
            domain=data.get("domain", "primary_care_multimorbidity"),
            verbatim_excerpt=data.get("verbatim_excerpt", ""),
            danger_signs=list(data.get("danger_signs", [])),
            certified=bool(data.get("certified", False)),
            version=str(data.get("version", "1.0.0")),
            approved_by=data.get("approved_by"),
            signature=data.get("signature")
        )
