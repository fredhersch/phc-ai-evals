"""
graph.py - Clinical Knowledge Graph & Collision Detection Engine
Constructs an ontological graph from Atomic Clinical Decision Units (a-CDUs),
detects multi-morbidity action-prohibition collisions across drug classes and conditions,
and provides tools for clinical review and safety tier calibration.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path

from .models import ClinicalDecisionUnit
from .ontology import (
    DRUG_CLASSES,
    CLINICAL_HAZARDS,
    CLINICAL_CONDITIONS,
    extract_drug_classes_from_text,
    extract_conditions_from_text,
    get_paracetamol_dose_ceiling
)


@dataclass
class CollisionRecord:
    cdu_a_id: str
    cdu_a_topic: str
    cdu_a_tier: int
    cdu_b_id: str
    cdu_b_topic: str
    cdu_b_tier: int
    collision_type: str  # DIRECT_CONTRAINDICATION, CLASS_SUBSUMPTION, DOSE_CEILING, CO_TRIGGER_TENSION
    conflicting_concept: str
    concept_name: str
    hazard_id: Optional[str]
    hazard_name: Optional[str]
    hazard_tier: int
    clinical_tension: str
    patient_archetype: str
    recommended_precedence: str
    suggested_safe_alternative: str
    tier_calibration_needed: bool = False
    calibration_recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalKnowledgeGraph:
    """
    Computable Knowledge Graph connecting a-CDUs, Clinical Indications,
    Interventions, Ontological Drug Classes, and Life-Threatening Hazards.
    """

    def __init__(self):
        # Nodes
        self.cdus: Dict[str, ClinicalDecisionUnit] = {}
        self.conditions: Dict[str, Dict[str, Any]] = {}
        self.drug_classes: Dict[str, Dict[str, Any]] = {}
        self.hazards: Dict[str, Dict[str, Any]] = {}

        # Graph Adjacency Edges
        # edge: (source_id, relation, target_id, metadata)
        self.edges: List[Dict[str, Any]] = []

        # Load ontology foundations
        self._init_ontology_nodes()

    def _init_ontology_nodes(self):
        for cid, data in CLINICAL_CONDITIONS.items():
            self.conditions[cid] = {
                "id": cid,
                "name": data["name"],
                "organ": data["organ"],
                "type": "Condition"
            }

        for did, data in DRUG_CLASSES.items():
            self.drug_classes[did] = {
                "id": did,
                "name": data["name"],
                "atc": data.get("atc"),
                "members": data.get("members", []),
                "type": "DrugClass"
            }

        for hid, hobj in CLINICAL_HAZARDS.items():
            self.hazards[hid] = {
                "id": hid,
                "name": hobj.name,
                "severity_tier": hobj.severity_tier,
                "target_organ": hobj.target_organ,
                "description": hobj.description,
                "type": "Hazard"
            }

            # Link drug class -> hazard
        for did, data in DRUG_CLASSES.items():
            for hid in data.get("hazards", []):
                self.edges.append({
                    "source": did,
                    "relation": "PRECIPITATES",
                    "target": hid,
                    "meta": {"severity_tier": CLINICAL_HAZARDS[hid].severity_tier if hid in CLINICAL_HAZARDS else 2}
                })

    def add_cdu(self, cdu: ClinicalDecisionUnit):
        """Ingests an a-CDU and connects it to the Knowledge Graph."""
        self.cdus[cdu.id] = cdu

        # 1. Connect CDU -> Conditions (Triggers)
        cond_matches = extract_conditions_from_text(f"{cdu.topic} {cdu.entry_criteria}")
        for cond_id, term in cond_matches:
            self.edges.append({
                "source": cdu.id,
                "relation": "TRIGGERS_ON",
                "target": cond_id,
                "meta": {"matched_term": term}
            })

        # 2. Connect CDU -> Mandated Actions (Interventions)
        for act in cdu.mandated_actions:
            act_lower = act.lower()
            # If the mandated action is phrased as a negative avoidance directive (e.g. "Strictly avoid NSAIDs"),
            # skip treating it as a positive prescription mandate.
            if any(m in act_lower for m in ["avoid", "strictly avoid", "do not", "never", "withhold", "stop", "discontinue", "contraindicated", "deprescribe"]):
                continue

            # If action is pure surveillance/monitoring rather than drug initiation, skip prescribing mandate
            if act_lower.startswith("monitor ") or act_lower.startswith("screen ") or act_lower.startswith("audit "):
                continue

            action_matches = extract_drug_classes_from_text(act)
            for class_id, term in action_matches:
                self.edges.append({
                    "source": cdu.id,
                    "relation": "MANDATES",
                    "target": class_id,
                    "meta": {"matched_term": term, "source_text": act}
                })

        # 3. Connect CDU -> Negative Boundaries (Prohibitions & Safety Clamps)
        negs_text = " ".join(cdu.negative_boundaries)
        neg_matches = extract_drug_classes_from_text(negs_text)
        for class_id, term in neg_matches:
            # Associate appropriate hazard
            hazards = DRUG_CLASSES.get(class_id, {}).get("hazards", [])
            primary_hazard = hazards[0] if hazards else None
            hazard_tier = CLINICAL_HAZARDS[primary_hazard].severity_tier if primary_hazard in CLINICAL_HAZARDS else 2

            self.edges.append({
                "source": cdu.id,
                "relation": "PROHIBITS",
                "target": class_id,
                "meta": {
                    "matched_term": term,
                    "source_text": negs_text,
                    "hazard_id": primary_hazard,
                    "hazard_tier": hazard_tier,
                    "cdu_tier": cdu.priority_tier
                }
            })

    def populate_library(self, cdus: List[ClinicalDecisionUnit]):
        """Populates the knowledge graph with a full library of a-CDUs."""
        for cdu in cdus:
            self.add_cdu(cdu)

    def detect_collisions(self) -> List[CollisionRecord]:
        """
        Cross-analyzes all loaded a-CDUs to detect multi-morbidity collisions:
        1. Mandated intervention in CDU_A is strictly prohibited in CDU_B.
        2. Dosing ceiling mismatch (e.g. Paracetamol 3g in OA vs 2.0g ceiling in Cirrhosis).
        3. Common multi-morbidity co-occurrence where clinical goals conflict.
        """
        collisions: List[CollisionRecord] = []
        cdu_ids = list(self.cdus.keys())

        # Map actions and prohibitions per CDU
        cdu_actions: Dict[str, List[Dict[str, Any]]] = {}
        cdu_prohibitions: Dict[str, List[Dict[str, Any]]] = {}
        cdu_conditions: Dict[str, List[str]] = {}

        for cdu_id in cdu_ids:
            cdu_actions[cdu_id] = [e for e in self.edges if e["source"] == cdu_id and e["relation"] == "MANDATES"]
            cdu_prohibitions[cdu_id] = [e for e in self.edges if e["source"] == cdu_id and e["relation"] == "PROHIBITS"]
            cdu_conditions[cdu_id] = [e["target"] for e in self.edges if e["source"] == cdu_id and e["relation"] == "TRIGGERS_ON"]

        # Pairwise comparison
        for i in range(len(cdu_ids)):
            for j in range(len(cdu_ids)):
                if i == j:
                    continue
                cdu_a = self.cdus[cdu_ids[i]]
                cdu_b = self.cdus[cdu_ids[j]]

                # Check: CDU_A mandates an intervention that CDU_B prohibits
                for act_edge in cdu_actions[cdu_a.id]:
                    target_class = act_edge["target"]
                    act_term = act_edge["meta"]["matched_term"]

                    # Look for matching prohibition in CDU_B
                    for prohib_edge in cdu_prohibitions[cdu_b.id]:
                        if prohib_edge["target"] == target_class:
                            hazard_id = prohib_edge["meta"].get("hazard_id")
                            hazard_info = CLINICAL_HAZARDS.get(hazard_id) if hazard_id else None
                            hazard_name = hazard_info.name if hazard_info else "Unspecified Adverse Outcome"
                            hazard_tier = hazard_info.severity_tier if hazard_info else 2

                            # Identify clinical tension & patient archetype
                            archetype, tension, alt = self._generate_collision_context(
                                cdu_a, cdu_b, target_class, act_term, hazard_name
                            )

                            # Determine tier calibration recommendation
                            cal_needed = False
                            cal_rec = None
                            if hazard_tier == 1 and cdu_b.priority_tier > 1:
                                cal_needed = True
                                cal_rec = f"Upgrade {cdu_b.id} to Priority Tier 1 (Fatal Veto) to clamp against {cdu_a.id}."
                            elif cdu_a.priority_tier == 1 and cdu_b.priority_tier == 1:
                                cal_rec = f"High-Stakes Tier 1 vs Tier 1 clash. Explicit clinician precedence rule required."

                            precedence = (
                                f"{cdu_b.id} (Tier {cdu_b.priority_tier}) OVERRIDES {cdu_a.id} (Tier {cdu_a.priority_tier})"
                                if cdu_b.priority_tier <= cdu_a.priority_tier
                                else f"CONFLICT: Lower-tier {cdu_b.id} cannot safely suppress {cdu_a.id} without tier calibration."
                            )

                            rec = CollisionRecord(
                                cdu_a_id=cdu_a.id,
                                cdu_a_topic=cdu_a.topic,
                                cdu_a_tier=cdu_a.priority_tier,
                                cdu_b_id=cdu_b.id,
                                cdu_b_topic=cdu_b.topic,
                                cdu_b_tier=cdu_b.priority_tier,
                                collision_type="CLASS_SUBSUMPTION" if target_class.startswith("CLASS:") else "DIRECT_CONTRAINDICATION",
                                conflicting_concept=target_class,
                                concept_name=DRUG_CLASSES.get(target_class, {}).get("name", target_class),
                                hazard_id=hazard_id,
                                hazard_name=hazard_name,
                                hazard_tier=hazard_tier,
                                clinical_tension=tension,
                                patient_archetype=archetype,
                                recommended_precedence=precedence,
                                suggested_safe_alternative=alt,
                                tier_calibration_needed=cal_needed,
                                calibration_recommendation=cal_rec
                            )
                            collisions.append(rec)

                # Check Dosing Ceiling Collision (e.g., Paracetamol ceiling)
                cdu_a_text = " ".join(cdu_a.mandated_actions)
                cdu_b_text = " ".join(cdu_b.negative_boundaries)
                dose_a = get_paracetamol_dose_ceiling(cdu_a_text)
                dose_b = get_paracetamol_dose_ceiling(cdu_b_text)

                if dose_a is not None and dose_b is not None and dose_a > dose_b:
                    rec = CollisionRecord(
                        cdu_a_id=cdu_a.id,
                        cdu_a_topic=cdu_a.topic,
                        cdu_a_tier=cdu_a.priority_tier,
                        cdu_b_id=cdu_b.id,
                        cdu_b_topic=cdu_b.topic,
                        cdu_b_tier=cdu_b.priority_tier,
                        collision_type="DOSE_CEILING",
                        conflicting_concept="CLASS:PARACETAMOL",
                        concept_name="Paracetamol Dosing Ceiling",
                        hazard_id="HAZARD:HEPATOTOXICITY_OVERDOSE",
                        hazard_name="Dose-Dependent Hepatotoxicity / Exhausted Glutathione",
                        hazard_tier=2,
                        clinical_tension=(
                            f"{cdu_a.id} permits Paracetamol up to {dose_a}g/day, but {cdu_b.id} enforces a strict "
                            f"ceiling of {dose_b}g/day."
                        ),
                        patient_archetype="Patient with Osteoarthritis / Musculoskeletal Pain + Concomitant Liver Cirrhosis",
                        recommended_precedence=f"Enforce lower dose ceiling ({dose_b}g/day maximum) to protect hepatic function.",
                        suggested_safe_alternative=f"Limit Paracetamol to <= {dose_b}g/day; augment with topical therapies or physical therapy.",
                        tier_calibration_needed=(cdu_b.priority_tier > cdu_a.priority_tier),
                        calibration_recommendation="Ensure safety ceiling CDU has equal or higher priority than symptomatic CDU."
                    )
                    collisions.append(rec)

        # Deduplicate identical pairs
        dedup_collisions = []
        seen_keys = set()
        for c in collisions:
            key = (c.cdu_a_id, c.cdu_b_id, c.conflicting_concept, c.collision_type)
            if key not in seen_keys:
                seen_keys.add(key)
                dedup_collisions.append(c)

        return dedup_collisions

    def _generate_collision_context(
        self,
        cdu_a: ClinicalDecisionUnit,
        cdu_b: ClinicalDecisionUnit,
        target_class: str,
        matched_term: str,
        hazard_name: str
    ) -> Tuple[str, str, str]:
        """Generates domain-rich clinical tension and safe alternative recommendations."""
        archetype = f"Co-morbid patient presenting with criteria for both '{cdu_a.topic}' and '{cdu_b.topic}'"
        tension = (
            f"{cdu_a.id} recommends or permits '{matched_term}' ({DRUG_CLASSES.get(target_class, {}).get('name', target_class)}), "
            f"whereas {cdu_b.id} strictly forbids it due to risk of {hazard_name}."
        )

        alt = "Consult verified guideline for non-conflicting second-line pharmacological alternative."

        if target_class == "CLASS:NSAID":
            alt = "Substitute oral NSAIDs with low-dose short-course oral Prednisolone, Paracetamol, or localized Intra-articular injections."
        elif target_class == "CLASS:BETA_BLOCKER_NONSELECTIVE":
            alt = "Substitute non-selective beta-blockers with Dihydropyridine Calcium Channel Blockers (Amlodipine) or Sublingual Nitrates."
        elif target_class == "CLASS:D2_ANTAGONIST_ANTIEMETIC":
            alt = "Strictly avoid Metoclopramide/Prochlorperazine in Parkinson's; use Ondansetron (5-HT3 antagonist) or Domperidone (peripheral only) for nausea."
        elif target_class == "CLASS:D2_ANTAGONIST_ANTIPSYCHOTIC":
            alt = "Use only Quetiapine, Pimavanserin, or low-dose Clozapine for psychosis in Parkinson's disease."
        elif target_class == "CLASS:TCA_ANTICHOLINERGIC":
            alt = "Substitute tricyclics with renally-adjusted Pregabalin or Duloxetine (SNRI) to prevent complete acute urinary retention."
        elif target_class == "CLASS:SULFONYLUREA_LONGACTING":
            alt = "Substitute long-acting sulfonylureas with DPP-4 inhibitors (Linagliptin) and relax HbA1c target to 7.5%-8.5%."
        elif target_class == "CLASS:TRIPTAN":
            alt = "Withhold triptans until blood pressure is controlled (<140/90); use Paracetamol + Metoclopramide or CGRP antagonists."

        return archetype, tension, alt

    def audit_tier_calibrations(self) -> List[Dict[str, Any]]:
        """
        Audits the entire loaded library to detect units whose assigned PriorityTier
        is misaligned with the severity of hazards defined in the knowledge graph.
        """
        recommendations = []

        for cdu_id, cdu in self.cdus.items():
            # Check all prohibitions on this CDU
            prohibs = [e for e in self.edges if e["source"] == cdu_id and e["relation"] == "PROHIBITS"]
            max_hazard_severity = 3
            fatal_hazards_prevented = []

            for p in prohibs:
                hid = p["meta"].get("hazard_id")
                if hid and hid in CLINICAL_HAZARDS:
                    sev = CLINICAL_HAZARDS[hid].severity_tier
                    if sev < max_hazard_severity:
                        max_hazard_severity = sev
                    if sev == 1 and CLINICAL_HAZARDS[hid].name not in fatal_hazards_prevented:
                        fatal_hazards_prevented.append(CLINICAL_HAZARDS[hid].name)

            # Also check text heuristics for fatal markers
            neg_text = " ".join(cdu.negative_boundaries).lower()
            if any(k in neg_text for k in ["fatal", "death", "lethal", "catastrophic", "asphyxiation", "irreversible"]):
                max_hazard_severity = min(max_hazard_severity, 1)

            # Compare current priority tier with warranted tier
            current_tier = cdu.priority_tier
            warranted_tier = max_hazard_severity

            status = "ALIGNED"
            rec_text = "Priority tier is medically aligned with graph hazard severity."

            if warranted_tier == 1 and current_tier > 1:
                status = "UPGRADE_RECOMMENDED"
                rec_text = (
                    f"CRITICAL SAFETY GAP: CDU prevents fatal/irreversible outcomes ({', '.join(fatal_hazards_prevented) or 'Life-threatening toxicity'}) "
                    f"but is currently Tier {current_tier}. It must be upgraded to Tier 1 (Hard Veto Floor) "
                    f"to prevent model-level instruction decay and dangerous overrides."
                )
            elif warranted_tier > current_tier and warranted_tier == 3 and current_tier == 1:
                status = "REVIEW_DOWNGRADE"
                rec_text = f"CDU is tagged Tier 1 but only addresses non-fatal symptomatic boundaries. Consider Tier 2 or 3."

            recommendations.append({
                "cdu_id": cdu.id,
                "topic": cdu.topic,
                "current_tier": current_tier,
                "warranted_tier": warranted_tier,
                "status": status,
                "hazards_prevented": fatal_hazards_prevented,
                "recommendation": rec_text,
                "guideline": cdu.source_guideline
            })

        return recommendations

    def export_summary_markdown(self) -> str:
        """Renders collision detection analysis into clean Markdown."""
        collisions = self.detect_collisions()
        audits = self.audit_tier_calibrations()

        lines = []
        lines.append("# a-CDU Knowledge Graph: Multi-Morbidity Collision & Safety Tier Audit")
        lines.append(f"\n**Total CDUs Analyzed:** {len(self.cdus)} | **Graph Edges:** {len(self.edges)} | **Detected Collisions:** {len(collisions)}\n")

        lines.append("## 1. Detected Multi-Morbidity Action $\\leftrightarrow$ Prohibition Collisions\n")
        if not collisions:
            lines.append("✅ No inter-CDU collisions detected.")
        else:
            lines.append("| Colliding Unit A | Colliding Unit B | Conflicting Concept | Hazard Prevented | Recommended Precedence |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for c in collisions:
                lines.append(
                    f"| **{c.cdu_a_id}** (T{c.cdu_a_tier}) | **{c.cdu_b_id}** (T{c.cdu_b_tier}) | `{c.concept_name}` | {c.hazard_name} | {c.recommended_precedence} |"
                )

        lines.append("\n## 2. Priority Tier Calibration Audit\n")
        lines.append("| CDU ID | Topic | Current Tier | Warranted Tier | Status | Clinical Action |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
        for a in audits:
            badge = "🟢 ALIGNED" if a["status"] == "ALIGNED" else "🛑 UPGRADE NEEDED"
            lines.append(
                f"| `{a['cdu_id']}` | {a['topic'][:35]} | T{a['current_tier']} | **T{a['warranted_tier']}** | {badge} | {a['recommendation'][:90]}... |"
            )

        return "\n".join(lines)

    def export_mermaid(self) -> str:
        """Renders the Knowledge Graph and collisions into Mermaid diagram syntax."""
        lines = ["flowchart TD"]
        lines.append("  %% Styling classes")
        lines.append("  classDef cduNode fill:#e8f0fe,stroke:#1a73e8,stroke-width:2px,color:#174ea6;")
        lines.append("  classDef drugNode fill:#e6f4ea,stroke:#137333,stroke-width:2px,color:#0d652d;")
        lines.append("  classDef hazardNode fill:#fce8e6,stroke:#c5221f,stroke-width:2px,color:#a50e0e;")
        lines.append("  classDef condNode fill:#fef7e0,stroke:#f29900,stroke-width:2px,color:#b06000;")

        # Nodes
        for cid, cdu in self.cdus.items():
            clean_id = cid.replace("-", "_")
            lines.append(f'  {clean_id}["{cid}<br/>Tier {cdu.priority_tier}"]:::cduNode')

        for did, dinfo in self.drug_classes.items():
            clean_id = did.replace(":", "_").replace("-", "_")
            dname = dinfo["name"]
            lines.append(f'  {clean_id}["{dname}"]:::drugNode')

        for hid, hinfo in self.hazards.items():
            clean_id = hid.replace(":", "_").replace("-", "_")
            hname = hinfo["name"]
            hsev = hinfo["severity_tier"]
            lines.append(f'  {clean_id}["⚠️ {hname}<br/>Tier {hsev} Hazard"]:::hazardNode')

        # Edges
        for e in self.edges:
            s = e["source"].replace(":", "_").replace("-", "_")
            t = e["target"].replace(":", "_").replace("-", "_")
            rel = e["relation"]
            if rel == "MANDATES":
                lines.append(f"  {s} == MANDATES ==> {t}")
            elif rel == "PROHIBITS":
                lines.append(f"  {s} -. PROHIBITS (VETO) .-> {t}")
            elif rel == "PRECIPITATES":
                lines.append(f"  {s} -->|PRECIPITATES| {t}")

        return "\n".join(lines)

    def export_dot(self) -> str:
        """Exports the graph to Graphviz DOT format."""
        lines = ["digraph ClinicalKnowledgeGraph {", '  rankdir="LR";', '  node [fontname="Helvetica"];']
        for cid, cdu in self.cdus.items():
            lines.append(f'  "{cid}" [shape="box", style="filled", fillcolor="#e8f0fe", label="{cid}\\nTier {cdu.priority_tier}"];')

        for did, dinfo in self.drug_classes.items():
            dname = dinfo["name"]
            lines.append(f'  "{did}" [shape="ellipse", style="filled", fillcolor="#e6f4ea", label="{dname}"];')

        for hid, hinfo in self.hazards.items():
            hname = hinfo["name"]
            lines.append(f'  "{hid}" [shape="octagon", style="filled", fillcolor="#fce8e6", label="{hname}"];')

        for e in self.edges:
            s = e["source"]
            t = e["target"]
            rel = e["relation"]
            lines.append(f'  "{s}" -> "{t}" [label="{rel}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire graph, collisions, and tier calibration audit to dictionary."""
        return {
            "stats": {
                "total_cdus": len(self.cdus),
                "total_edges": len(self.edges),
                "total_conditions": len(self.conditions),
                "total_drug_classes": len(self.drug_classes),
                "total_hazards": len(self.hazards)
            },
            "collisions": [c.to_dict() for c in self.detect_collisions()],
            "tier_calibrations": self.audit_tier_calibrations(),
            "nodes": {
                "cdus": [c.to_dict() for c in self.cdus.values()],
                "drug_classes": list(self.drug_classes.values()),
                "conditions": list(self.conditions.values()),
                "hazards": list(self.hazards.values())
            },
            "edges": self.edges
        }

    def export_html_dossier(self, output_path: Path):
        """Generates a rich, interactive HTML Clinical Review Studio & Collision Dossier."""
        collisions = self.detect_collisions()
        audits = self.audit_tier_calibrations()
        stats = {
            "cdus": len(self.cdus),
            "edges": len(self.edges),
            "collisions": len(collisions),
            "tier1_clashes": sum(1 for c in collisions if c.cdu_a_tier == 1 or c.cdu_b_tier == 1),
            "upgrades_needed": sum(1 for a in audits if a["status"] == "UPGRADE_RECOMMENDED")
        }

        # Prepare Vis.js nodes & edges
        vis_nodes = []
        vis_edges = []
        node_seen = set()

        # Add CDU nodes
        for cid, cdu in self.cdus.items():
            vis_nodes.append({
                "id": cid,
                "label": f"{cid}\n(T{cdu.priority_tier})",
                "title": f"<b>{cid}</b><br>{cdu.topic}<br><i>Priority Tier: {cdu.priority_tier}</i>",
                "group": "cdu",
                "color": {"background": "#e0f2fe", "border": "#0284c7"},
                "shape": "box",
                "font": {"color": "#0369a1", "size": 13, "face": "monospace", "bold": True}
            })
            node_seen.add(cid)

        # Add Drug Class nodes involved in collisions
        for c in collisions:
            did = c.conflicting_concept
            if did not in node_seen:
                dname = DRUG_CLASSES.get(did, {}).get("name", did)
                vis_nodes.append({
                    "id": did,
                    "label": dname,
                    "title": f"<b>Drug Class:</b> {dname}<br>ID: {did}",
                    "group": "drug",
                    "color": {"background": "#dcfce7", "border": "#16a34a"},
                    "shape": "ellipse",
                    "font": {"color": "#15803d", "size": 12}
                })
                node_seen.add(did)

            # Add Hazard node if present
            if c.hazard_id and c.hazard_id not in node_seen:
                hinfo = CLINICAL_HAZARDS.get(c.hazard_id)
                hname = hinfo.name if hinfo else c.hazard_name
                vis_nodes.append({
                    "id": c.hazard_id,
                    "label": f"⚠️ {hname}",
                    "title": f"<b>Hazard:</b> {hname}<br>Severity: Tier {c.hazard_tier}",
                    "group": "hazard",
                    "color": {"background": "#fee2e2", "border": "#dc2626"},
                    "shape": "hexagon",
                    "font": {"color": "#b91c1c", "size": 12, "bold": True}
                })
                node_seen.add(c.hazard_id)

            # Collision edge between CDU A and CDU B
            vis_edges.append({
                "from": c.cdu_a_id,
                "to": c.cdu_b_id,
                "label": f"⚡ CLASH: {c.concept_name}",
                "color": {"color": "#ef4444", "highlight": "#b91c1c"},
                "dashes": True,
                "arrows": "to",
                "width": 2,
                "font": {"color": "#b91c1c", "size": 11, "strokeWidth": 2, "strokeColor": "#ffffff"}
            })

            # Connect CDU to conflicting concept
            vis_edges.append({
                "from": c.cdu_a_id,
                "to": did,
                "label": "MANDATES",
                "color": {"color": "#0284c7"},
                "arrows": "to",
                "width": 1
            })
            vis_edges.append({
                "from": c.cdu_b_id,
                "to": did,
                "label": "PROHIBITS",
                "color": {"color": "#dc2626"},
                "dashes": True,
                "arrows": "to",
                "width": 2
            })
            if c.hazard_id:
                vis_edges.append({
                    "from": did,
                    "to": c.hazard_id,
                    "label": "PRECIPITATES",
                    "color": {"color": "#b91c1c"},
                    "arrows": "to",
                    "width": 1
                })

        graph_payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges})
        cards_html = []
        for idx, c in enumerate(collisions, start=1):
            badge_class = "danger-badge" if c.hazard_tier == 1 else "warn-badge"
            cal_box = ""
            if c.tier_calibration_needed:
                cal_box = f"""
                <div class="cal-alert">
                    <strong>⚠️ Calibration Action:</strong> {c.calibration_recommendation}
                </div>
                """
            cards_html.append(f"""
            <div class="collision-card">
                <div class="collision-header">
                    <div>
                        <span class="case-num">Collision #{idx:02d}</span>
                        <span class="{badge_class}">Tier {c.hazard_tier} Hazard</span>
                        <span class="type-pill">{c.collision_type}</span>
                    </div>
                    <div class="concept-tag">{c.concept_name}</div>
                </div>
                <div class="units-grid">
                    <div class="unit-box unit-a">
                        <div class="unit-title">CDU A (Prescribing Intent)</div>
                        <div class="unit-id">{c.cdu_a_id} <span class="tier-tag">Tier {c.cdu_a_tier}</span></div>
                        <div class="unit-desc">{c.cdu_a_topic}</div>
                    </div>
                    <div class="versus-icon">⚡ CLASHES WITH ⚡</div>
                    <div class="unit-box unit-b">
                        <div class="unit-title">CDU B (Safety Override / Veto)</div>
                        <div class="unit-id">{c.cdu_b_id} <span class="tier-tag">Tier {c.cdu_b_tier}</span></div>
                        <div class="unit-desc">{c.cdu_b_topic}</div>
                    </div>
                </div>
                <div class="detail-section">
                    <strong>Clinical Tension:</strong>
                    <p>{c.clinical_tension}</p>
                </div>
                <div class="detail-section">
                    <strong>Multi-Morbidity Archetype:</strong>
                    <p>{c.patient_archetype}</p>
                </div>
                <div class="detail-section">
                    <strong>Hazard / Pathophysiological Threat:</strong>
                    <p class="hazard-text">🛑 {c.hazard_name}</p>
                </div>
                <div class="detail-section safe-box">
                    <strong>Recommended Precedence & Safe Alternative:</strong>
                    <p><strong>Precedence:</strong> {c.recommended_precedence}</p>
                    <p><strong>Guideline Safe Pathway:</strong> {c.suggested_safe_alternative}</p>
                </div>
                {cal_box}
            </div>
            """)

        # Render audit rows
        audit_rows = []
        for a in audits:
            status_badge = (
                '<span class="badge-aligned">ALIGNED</span>'
                if a["status"] == "ALIGNED"
                else '<span class="badge-upgrade">UPGRADE TO T1</span>'
            )
            audit_rows.append(f"""
            <tr>
                <td><code>{a['cdu_id']}</code></td>
                <td>{a['topic']}</td>
                <td style="text-align: center;"><span class="tier-pill">T{a['current_tier']}</span></td>
                <td style="text-align: center;"><strong class="warranted-tier">T{a['warranted_tier']}</strong></td>
                <td style="text-align: center;">{status_badge}</td>
                <td style="font-size: 13px; color: #475467;">{a['recommendation']}</td>
            </tr>
            """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>a-CDU Clinical Review Studio: Knowledge Graph & Collision Dossier</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {{
            --primary: #1e40af;
            --primary-light: #eff6ff;
            --danger: #dc2626;
            --danger-light: #fef2f2;
            --warning: #d97706;
            --warning-light: #fffbeb;
            --success: #16a34a;
            --success-light: #f0fdf4;
            --bg: #0f172a;
            --surface: #ffffff;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: #f8fafc; color: var(--text-dark); line-height: 1.6; padding: 32px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ margin-bottom: 24px; background: white; padding: 28px; border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        h1 {{ font-size: 26px; color: #1e293b; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; }}
        p.subtitle {{ color: var(--text-muted); font-size: 15px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .stat-card {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: 700; color: #0f172a; }}
        .stat-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}

        /* Graph Canvas Styles */
        .graph-wrapper {{ background: white; border: 1px solid var(--border); border-radius: 12px; margin-bottom: 28px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
        .graph-header {{ padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: #ffffff; flex-wrap: wrap; gap: 10px; }}
        .graph-title {{ font-weight: 700; font-size: 16px; color: #1e293b; display: flex; align-items: center; gap: 8px; }}
        .graph-controls {{ display: flex; gap: 8px; align-items: center; }}
        .graph-btn {{ background: #f1f5f9; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; }}
        .graph-btn:hover {{ background: #e2e8f0; }}
        #network-canvas {{ width: 100%; height: 500px; background: #fafbfc; }}
        .graph-legend {{ padding: 10px 20px; background: #f8fafc; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap; }}
        .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .dot-cdu {{ background: #0284c7; }}
        .dot-drug {{ background: #16a34a; }}
        .dot-hazard {{ background: #dc2626; border-radius: 2px; }}

        /* Filter Controls */
        .controls-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 12px; }}
        .filter-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .filter-btn {{ background: white; border: 1px solid var(--border); padding: 8px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; cursor: pointer; color: #475467; transition: all 0.15s; }}
        .filter-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
        .search-box {{ padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px; font-size: 13px; width: 260px; background: white; }}
        .search-box:focus {{ outline: none; border-color: var(--primary); }}

        .section-title {{ font-size: 20px; margin: 32px 0 16px 0; color: #1e293b; display: flex; align-items: center; justify-content: space-between; }}
        .collision-card {{ background: white; border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: transform 0.15s ease, border-color 0.15s ease; cursor: pointer; }}
        .collision-card:hover {{ border-color: var(--primary); transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.06); }}
        .collision-card.active-card {{ border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary-light); }}
        .collision-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; }}
        .case-num {{ font-weight: 700; color: #1e293b; margin-right: 8px; }}
        .danger-badge {{ background: var(--danger-light); color: var(--danger); font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }}
        .warn-badge {{ background: var(--warning-light); color: var(--warning); font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }}
        .type-pill {{ background: #f1f5f9; color: #475467; font-size: 11px; font-weight: 600; padding: 4px 8px; border-radius: 6px; margin-left: 6px; }}
        .concept-tag {{ font-family: monospace; font-size: 13px; font-weight: 700; color: var(--primary); background: var(--primary-light); padding: 4px 12px; border-radius: 6px; }}

        .units-grid {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center; margin-bottom: 20px; }}
        .unit-box {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
        .unit-a {{ border-left: 4px solid var(--warning); }}
        .unit-b {{ border-left: 4px solid var(--danger); }}
        .versus-icon {{ font-weight: 800; font-size: 12px; color: #94a3b8; text-align: center; }}
        .unit-title {{ font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin-bottom: 4px; }}
        .unit-id {{ font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }}
        .tier-tag {{ font-size: 11px; font-weight: 700; background: #e2e8f0; color: #334155; padding: 2px 6px; border-radius: 4px; }}
        .unit-desc {{ font-size: 13px; color: #475467; }}

        .detail-section {{ margin-bottom: 12px; font-size: 14px; }}
        .detail-section strong {{ color: #334155; display: block; margin-bottom: 2px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .hazard-text {{ color: var(--danger); font-weight: 600; }}
        .safe-box {{ background: var(--success-light); border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; }}
        .safe-box strong {{ color: var(--success); }}
        .safe-box p {{ color: #166534; font-size: 13.5px; margin-bottom: 4px; }}
        .cal-alert {{ margin-top: 14px; background: var(--danger-light); border: 1px solid #fecaca; border-radius: 6px; padding: 10px; color: #991b1b; font-size: 13.5px; }}

        table.audit-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
        table.audit-table th {{ background: #f8fafc; padding: 12px 16px; text-align: left; font-size: 12px; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
        table.audit-table td {{ padding: 14px 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; vertical-align: middle; }}
        .tier-pill {{ background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 12px; }}
        .warranted-tier {{ color: var(--primary); font-size: 15px; }}
        .badge-aligned {{ background: var(--success-light); color: var(--success); padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 11px; }}
        .badge-upgrade {{ background: var(--danger-light); color: var(--danger); padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 11px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ a-CDU Knowledge Graph & Clinical Safety Studio</h1>
            <p class="subtitle">
                Interactive Multi-Morbidity Collision Visualizer, Ontological Subsumption Graph, and Priority Tier Calibration Studio.
            </p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['cdus']}</div>
                    <div class="stat-label">Active a-CDUs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['edges']}</div>
                    <div class="stat-label">Graph Edges</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--danger);">{stats['collisions']}</div>
                    <div class="stat-label">Detected Collisions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--warning);">{stats['tier1_clashes']}</div>
                    <div class="stat-label">Tier-1 High-Stakes Clashes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #6366f1;">{stats['upgrades_needed']}</div>
                    <div class="stat-label">Recommended Tier Upgrades</div>
                </div>
            </div>
        </header>

        <!-- Interactive Visual Graph -->
        <div class="graph-wrapper">
            <div class="graph-header">
                <div class="graph-title">
                    <span>🕸️ Interactive Multi-Morbidity Collision Graph</span>
                    <span style="font-size: 12px; color: var(--text-muted); font-weight: 400;">(Click any node to focus; drag to inspect force physics)</span>
                </div>
                <div class="graph-controls">
                    <button class="graph-btn" onclick="fitNetwork()">🔍 Reset Zoom</button>
                    <button class="graph-btn" id="physics-btn" onclick="togglePhysics()">⏸️ Toggle Physics</button>
                </div>
            </div>
            <div id="network-canvas"></div>
            <div class="graph-legend">
                <span class="legend-item"><span class="dot dot-cdu"></span> <strong>a-CDU Unit</strong> (Box)</span>
                <span class="legend-item"><span class="dot dot-drug"></span> <strong>Pharmacological Drug Class</strong> (Ellipse)</span>
                <span class="legend-item"><span class="dot dot-hazard"></span> <strong>Clinical Hazard / Hazard Tier</strong> (Hexagon)</span>
                <span class="legend-item" style="color: #ef4444;">⚡ Red Dashed Arrow = Inter-CDU Collision</span>
            </div>
        </div>

        <div class="section-title">
            <span>⚡ Detected Multi-Morbidity Collisions ({len(collisions)})</span>
            <span style="font-size: 13px; color: var(--text-muted);">Pairs where Prescribing Intent directly conflicts with Contraindications</span>
        </div>

        <!-- Filter & Search Controls -->
        <div class="controls-row">
            <div class="filter-group">
                <button class="filter-btn active" onclick="filterCards('ALL', this)">All Collisions ({len(collisions)})</button>
                <button class="filter-btn" onclick="filterCards('TIER1', this)">🛑 Tier 1 Fatal Clashes</button>
                <button class="filter-btn" onclick="filterCards('CEILING', this)">⚖️ Dosing Ceilings</button>
                <button class="filter-btn" onclick="filterCards('NSAID', this)">💊 NSAID Clashes</button>
            </div>
            <input type="text" class="search-box" id="searchInput" placeholder="Search CDUs, drugs, or hazards..." onkeyup="searchCards()">
        </div>

        <div id="collisions-list">
            {"".join(cards_html)}
        </div>

        <div class="section-title">
            <span>📋 Clinical Priority Tier Calibration Audit ({len(audits)} units)</span>
            <span style="font-size: 13px; color: var(--text-muted);">Verification of Tier 1 (Fatal Veto) vs Tier 2 (Organ Protection) vs Tier 3 (Symptom)</span>
        </div>
        <table class="audit-table">
            <thead>
                <tr>
                    <th>CDU ID</th>
                    <th>Clinical Topic</th>
                    <th style="text-align: center;">Current Tier</th>
                    <th style="text-align: center;">Warranted Tier</th>
                    <th style="text-align: center;">Audit Status</th>
                    <th>Safety Rationale & Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {"".join(audit_rows)}
            </tbody>
        </table>
    </div>

    <!-- Graph Rendering Script -->
    <script>
        const graphData = {graph_payload};
        let networkInstance = null;
        let physicsEnabled = true;

        function initGraph() {{
            const container = document.getElementById('network-canvas');
            if (!container || !window.vis) return;

            const data = {{
                nodes: new vis.DataSet(graphData.nodes),
                edges: new vis.DataSet(graphData.edges)
            }};

            const options = {{
                nodes: {{
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    smooth: {{ type: 'cubicBezier', roundness: 0.2 }}
                }},
                physics: {{
                    barnesHut: {{
                        gravitationalConstant: -3500,
                        springLength: 120,
                        springConstant: 0.04
                    }},
                    stabilization: {{ iterations: 120 }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 150,
                    zoomView: true
                }}
            }};

            networkInstance = new vis.Network(container, data, options);

            networkInstance.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    highlightCollisionsByNode(nodeId);
                }}
            }});
        }}

        function fitNetwork() {{
            if (networkInstance) networkInstance.fit({{ animation: true }});
        }}

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            if (networkInstance) {{
                networkInstance.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
            }}
            document.getElementById('physics-btn').innerText = physicsEnabled ? "⏸️ Toggle Physics" : "▶️ Resume Physics";
        }}

        function highlightCollisionsByNode(nodeId) {{
            const cards = document.querySelectorAll('.collision-card');
            cards.forEach(c => {{
                const text = c.innerText;
                if (text.includes(nodeId)) {{
                    c.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                    c.classList.add('active-card');
                    setTimeout(() => c.classList.remove('active-card'), 2000);
                }}
            }});
        }}

        function focusCollisionOnGraph(cduA, cduB) {{
            if (!networkInstance) return;
            networkInstance.selectNodes([cduA, cduB]);
            networkInstance.focus(cduA, {{ scale: 1.2, animation: true }});
        }}

        // Card filtering
        function filterCards(type, btn) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const cards = document.querySelectorAll('.collision-card');
            cards.forEach(c => {{
                let show = true;
                const txt = c.innerText.toUpperCase();
                if (type === 'TIER1') {{
                    show = txt.includes('TIER 1 HAZARD');
                }} else if (type === 'CEILING') {{
                    show = txt.includes('DOSE_CEILING');
                }} else if (type === 'NSAID') {{
                    show = txt.includes('NSAID');
                }}
                c.style.display = show ? 'block' : 'none';
            }});
        }}

        function searchCards() {{
            const q = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.collision-card');
            cards.forEach(c => {{
                const match = c.innerText.toLowerCase().includes(q);
                c.style.display = match ? 'block' : 'none';
            }});
        }}

        // Attach click handlers to collision cards
        document.addEventListener('DOMContentLoaded', () => {{
            initGraph();
            const cards = document.querySelectorAll('.collision-card');
            cards.forEach(c => {{
                const idElA = c.querySelector('.unit-a .unit-id');
                const idElB = c.querySelector('.unit-b .unit-id');
                if (idElA && idElB) {{
                    const idA = idElA.innerText.split(' ')[0].trim();
                    const idB = idElB.innerText.split(' ')[0].trim();
                    c.addEventListener('click', () => focusCollisionOnGraph(idA, idB));
                }}
            }});
        }});
    </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
