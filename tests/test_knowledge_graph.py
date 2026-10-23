"""
test_knowledge_graph.py - Comprehensive Unit Tests for a-CDU Knowledge Graph & Collision Detection Engine
"""

import unittest
import json
from pathlib import Path
import tempfile
import shutil

from harness.models import ClinicalDecisionUnit
from harness.ontology import (
    extract_drug_classes_from_text,
    extract_conditions_from_text,
    get_paracetamol_dose_ceiling,
    DRUG_CLASSES,
    CLINICAL_HAZARDS
)
from harness.graph import ClinicalKnowledgeGraph, CollisionRecord


class TestClinicalOntology(unittest.TestCase):
    """Tests concept normalization and entity extraction from clinical text."""

    def test_extract_nsaid_class_and_members(self):
        text = "Prescribe Ibuprofen 400mg or Naproxen 500mg. Strictly avoid systemic NSAIDs."
        matches = extract_drug_classes_from_text(text)
        class_ids = [m[0] for m in matches]
        self.assertIn("CLASS:NSAID", class_ids)
        terms = [m[1] for m in matches]
        self.assertTrue(any("ibuprofen" in t or "naproxen" in t or "nsaid" in t for t in terms))

    def test_extract_beta_blockers_differentiated(self):
        non_sel_text = "Strictly contraindicate Propranolol, Carvedilol, and Labetalol in asthma."
        matches = extract_drug_classes_from_text(non_sel_text)
        class_ids = [m[0] for m in matches]
        self.assertIn("CLASS:BETA_BLOCKER_NONSELECTIVE", class_ids)

        cardioselective_text = "If necessary, use highly cardioselective Bisoprolol at 1.25mg."
        cs_matches = extract_drug_classes_from_text(cardioselective_text)
        cs_ids = [m[0] for m in cs_matches]
        self.assertIn("CLASS:BETA_BLOCKER_CARDIOSELECTIVE", cs_ids)

    def test_extract_d2_antagonists(self):
        text = "Contraindicated: Haloperidol, Metoclopramide, and Prochlorperazine in Parkinson's."
        matches = extract_drug_classes_from_text(text)
        class_ids = [m[0] for m in matches]
        self.assertIn("CLASS:D2_ANTAGONIST_ANTIPSYCHOTIC", class_ids)
        self.assertIn("CLASS:D2_ANTAGONIST_ANTIEMETIC", class_ids)

    def test_extract_clinical_conditions(self):
        text = "Patient with active asthma, stage 3b CKD, and compensated liver cirrhosis."
        conds = extract_conditions_from_text(text)
        cond_ids = [c[0] for c in conds]
        self.assertIn("COND:ASTHMA", cond_ids)
        self.assertIn("COND:CKD_ADVANCED", cond_ids)
        self.assertIn("COND:CIRRHOSIS", cond_ids)

    def test_paracetamol_dose_ceiling_parser(self):
        text1 = "Regular scheduled Paracetamol up to 3g/day for knee pain."
        self.assertEqual(get_paracetamol_dose_ceiling(text1), 3.0)

        text2 = "Strictly cap Paracetamol at a maximum of 2.0g per 24 hours."
        self.assertEqual(get_paracetamol_dose_ceiling(text2), 2.0)

        text3 = "No analgesics specified."
        self.assertIsNone(get_paracetamol_dose_ceiling(text3))


class TestKnowledgeGraphCollisionEngine(unittest.TestCase):
    """Tests Knowledge Graph assembly, collision detection, and tier calibration."""

    def setUp(self):
        self.kg = ClinicalKnowledgeGraph()

        # Load standard test CDUs
        self.cdu_gout = ClinicalDecisionUnit(
            id="CDU-GOUT-TEST",
            topic="Gout Flare Analgesia",
            source_guideline="ACR Gout 2020",
            entry_criteria="Acute gout attack",
            mandated_actions=["Prescribe short-course oral Prednisolone or Ibuprofen 400mg"],
            negative_boundaries=["Do not delay treatment"],
            priority_tier=2
        )

        self.cdu_ckd = ClinicalDecisionUnit(
            id="CDU-CKD-TEST",
            topic="CKD NSAID Avoidance",
            source_guideline="KDIGO 2024",
            entry_criteria="Stage 3b CKD with eGFR 34 mL/min",
            mandated_actions=["Monitor renal function"],
            negative_boundaries=["🛑 STRICTLY CONTRAINDICATE oral NSAIDs (Ibuprofen, Naproxen) -> acute renal shutdown."],
            priority_tier=2
        )

        self.cdu_cirrhosis = ClinicalDecisionUnit(
            id="CDU-CIRRHOSIS-TEST",
            topic="Cirrhosis Analgesia Ceiling",
            source_guideline="AASLD 2022",
            entry_criteria="Liver cirrhosis with abdominal or joint pain",
            mandated_actions=["Use topical therapies and gentle physical therapy"],
            negative_boundaries=[
                "🛑 NEVER exceed Paracetamol 2.0g/day in cirrhosis.",
                "🛑 NEVER prescribe oral NSAIDs -> fatal hepatorenal syndrome."
            ],
            priority_tier=2
        )

        self.cdu_oa = ClinicalDecisionUnit(
            id="CDU-OA-TEST",
            topic="Osteoarthritis Knee Pain",
            source_guideline="OARSI 2023",
            entry_criteria="Knee osteoarthritis flare",
            mandated_actions=[
                "Recommend regular scheduled Paracetamol up to 3g/day",
                "Consider oral NSAID Ibuprofen if refractory"
            ],
            negative_boundaries=["Do not leave pain untreated"],
            priority_tier=3
        )

    def test_node_and_edge_population(self):
        self.kg.add_cdu(self.cdu_gout)
        self.kg.add_cdu(self.cdu_ckd)

        self.assertIn("CDU-GOUT-TEST", self.kg.cdus)
        self.assertIn("CDU-CKD-TEST", self.kg.cdus)

        # Check mandated action edge
        mandate_edges = [e for e in self.kg.edges if e["source"] == "CDU-GOUT-TEST" and e["relation"] == "MANDATES"]
        self.assertTrue(len(mandate_edges) > 0)
        self.assertEqual(mandate_edges[0]["target"], "CLASS:NSAID")

        # Check prohibition edge
        prohib_edges = [e for e in self.kg.edges if e["source"] == "CDU-CKD-TEST" and e["relation"] == "PROHIBITS"]
        self.assertTrue(len(prohib_edges) > 0)
        self.assertEqual(prohib_edges[0]["target"], "CLASS:NSAID")

    def test_detect_nsaid_collision(self):
        self.kg.add_cdu(self.cdu_gout)
        self.kg.add_cdu(self.cdu_ckd)
        collisions = self.kg.detect_collisions()

        self.assertEqual(len(collisions), 1)
        c = collisions[0]
        self.assertEqual(c.cdu_a_id, "CDU-GOUT-TEST")
        self.assertEqual(c.cdu_b_id, "CDU-CKD-TEST")
        self.assertEqual(c.conflicting_concept, "CLASS:NSAID")
        self.assertIn("Prednisolone", c.suggested_safe_alternative)

    def test_detect_dosing_ceiling_collision(self):
        self.kg.add_cdu(self.cdu_oa)
        self.kg.add_cdu(self.cdu_cirrhosis)
        collisions = self.kg.detect_collisions()

        ceiling_collisions = [c for c in collisions if c.collision_type == "DOSE_CEILING"]
        self.assertEqual(len(ceiling_collisions), 1)
        c = ceiling_collisions[0]
        self.assertEqual(c.conflicting_concept, "CLASS:PARACETAMOL")
        self.assertIn("2.0g/day", c.clinical_tension)
        self.assertIn("3.0g/day", c.clinical_tension)

    def test_tier_calibration_audit(self):
        self.kg.add_cdu(self.cdu_cirrhosis)
        audits = self.kg.audit_tier_calibrations()

        cirrhosis_audit = next(a for a in audits if a["cdu_id"] == "CDU-CIRRHOSIS-TEST")
        # Since cirrhosis protects against fatal hepatorenal syndrome and states 'fatal', warranted tier should be 1
        self.assertEqual(cirrhosis_audit["warranted_tier"], 1)
        self.assertEqual(cirrhosis_audit["status"], "UPGRADE_RECOMMENDED")

    def test_full_dataset_execution(self):
        dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "phc_decision_units.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cdus = [ClinicalDecisionUnit.from_dict(d) for d in data]

        kg = ClinicalKnowledgeGraph()
        kg.populate_library(cdus)
        collisions = kg.detect_collisions()
        audits = kg.audit_tier_calibrations()

        self.assertTrue(len(collisions) >= 8)
        self.assertEqual(len(audits), len(cdus))

        # Check export formats
        mermaid_text = kg.export_mermaid()
        self.assertIn("flowchart TD", mermaid_text)

        dot_text = kg.export_dot()
        self.assertIn("digraph ClinicalKnowledgeGraph", dot_text)

        kg_dict = kg.to_dict()
        self.assertIn("stats", kg_dict)
        self.assertIn("collisions", kg_dict)

        # Test HTML export in temp dir
        temp_dir = tempfile.mkdtemp()
        try:
            html_path = Path(temp_dir) / "test_report.html"
            kg.export_html_dossier(html_path)
            self.assertTrue(html_path.exists())
            content = html_path.read_text(encoding="utf-8")
            self.assertIn("a-CDU Clinical Review Studio", content)
            self.assertIn("Detected Multi-Morbidity Collisions", content)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
