"""
ontology.py - Clinical Ontology & Pharmacological Concept Hierarchy
Provides standardized medical concepts, drug classes, organ system vulnerabilities,
and hazard classifications to ground a-CDUs in a computable knowledge representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import re


@dataclass
class DrugConcept:
    name: str
    class_id: str
    atc_code: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    is_class: bool = False


@dataclass
class HazardConcept:
    id: str
    name: str
    severity_tier: int  # 1 = Fatal / Irreversible Organ Failure, 2 = Serious Organ Impairment, 3 = Symptom Exacerbation
    target_organ: str
    description: str


# Standardized Pharmacological Concept Hierarchy & Controlled Terminology
DRUG_CLASSES: Dict[str, Dict] = {
    "CLASS:NSAID": {
        "name": "Non-Steroidal Anti-Inflammatory Drugs (NSAIDs)",
        "atc": "M01A",
        "synonyms": ["nsaid", "nsaids", "oral nsaid", "systemic nsaid", "oral nsaids", "systemic nsaids", "cox inhibitor"],
        "members": [
            "ibuprofen", "naproxen", "diclofenac", "indomethacin", "celecoxib",
            "meloxicam", "ketorolac", "etoricoxib", "piroxicam", "aspirin high-dose"
        ],
        "hazards": ["HAZARD:ACUTE_RENAL_FAILURE", "HAZARD:GI_BLEED", "HAZARD:HEPATORENAL_SYNDROME", "HAZARD:HF_DECOMPENSATION"]
    },
    "CLASS:BETA_BLOCKER_NONSELECTIVE": {
        "name": "Non-Selective Beta-Adrenergic Antagonists",
        "atc": "C07AA",
        "synonyms": ["non-selective beta-blocker", "nonselective beta blocker", "non-selective beta blocker", "non-cardioselective beta-blocker"],
        "members": [
            "propranolol", "carvedilol", "labetalol", "timolol", "nadolol", "sotalol"
        ],
        "hazards": ["HAZARD:FATAL_BRONCHOSPASM"]
    },
    "CLASS:BETA_BLOCKER_CARDIOSELECTIVE": {
        "name": "Cardioselective (Beta-1 Selective) Antagonists",
        "atc": "C07AB",
        "synonyms": ["cardioselective beta-blocker", "beta-1 selective blocker", "cardioselective beta-1 blocker"],
        "members": [
            "bisoprolol", "metoprolol", "atenolol", "nebivolol"
        ],
        "hazards": ["HAZARD:BRONCHOSPASM_MILD"]
    },
    "CLASS:D2_ANTAGONIST_ANTIPSYCHOTIC": {
        "name": "Dopamine D2 Receptor Antagonists (First & Second Gen Antipsychotics)",
        "atc": "N05A",
        "synonyms": ["typical antipsychotic", "first-generation antipsychotic", "d2-blocking antipsychotic", "d2-blocking atypical"],
        "members": [
            "haloperidol", "chlorpromazine", "fluphenazine", "risperidone", "olanzapine", "aripiprazole"
        ],
        "hazards": ["HAZARD:CATATONIC_PARKINSONISM_FREEZE", "HAZARD:NEUROLEPTIC_MALIGNANT"]
    },
    "CLASS:D2_ANTAGONIST_ANTIEMETIC": {
        "name": "Dopamine D2 Receptor Antagonist Antiemetics",
        "atc": "A03FA",
        "synonyms": ["d2-blocking antiemetic", "d2-blocking antiemetics", "d2 antagonist antiemetic"],
        "members": [
            "metoclopramide", "prochlorperazine", "promethazine", "droperidol"
        ],
        "hazards": ["HAZARD:CATATONIC_PARKINSONISM_FREEZE", "HAZARD:ACUTE_DYSTONIA"]
    },
    "CLASS:TCA_ANTICHOLINERGIC": {
        "name": "Tricyclic Antidepressants with Potent Anticholinergic Properties",
        "atc": "N06AA",
        "synonyms": ["tricyclic antidepressant", "tricyclic antidepressants", "tca", "tcas", "potent anticholinergic"],
        "members": [
            "amitriptyline", "nortriptyline", "imipramine", "clomipramine", "doxepin"
        ],
        "hazards": ["HAZARD:ACUTE_URINARY_RETENTION", "HAZARD:SEVERE_DELIRIUM"]
    },
    "CLASS:SULFONYLUREA_LONGACTING": {
        "name": "Long-Acting Sulfonylureas",
        "atc": "A10BB",
        "synonyms": ["long-acting sulfonylurea", "long acting sulfonylureas", "sulfonylurea", "sulfonylureas"],
        "members": [
            "glibenclamide", "glyburide", "glimepiride"
        ],
        "hazards": ["HAZARD:FATAL_HYPOGLYCEMIC_COMA"]
    },
    "CLASS:OPIOID_SEDATING": {
        "name": "Systemic Sedating Opioids",
        "atc": "N02A",
        "synonyms": ["sedating opioid", "sedating opioids", "full opioid agonist", "systemic opioid"],
        "members": [
            "morphine", "oxycodone", "codeine", "fentanyl", "hydromorphone", "methadone", "tramadol"
        ],
        "hazards": ["HAZARD:HEPATIC_ENCEPHALOPATHY", "HAZARD:RESPIRATORY_DEPRESSION"]
    },
    "CLASS:TRIPTAN": {
        "name": "5-HT1B/1D Receptor Agonists (Triptans)",
        "atc": "N02CC",
        "synonyms": ["triptan", "triptans", "5-ht1b/1d agonist", "5-ht1 receptor agonist"],
        "members": [
            "sumatriptan", "zolmitriptan", "rizatriptan", "eletriptan", "naratriptan", "almotriptan", "frovatriptan"
        ],
        "hazards": ["HAZARD:CORONARY_VASOSPASM_STROKE"]
    },
    "CLASS:ACEI_ARB": {
        "name": "Renin-Angiotensin System Inhibitors (ACEi / ARB)",
        "atc": "C09",
        "synonyms": ["raas inhibitor", "raas inhibitors", "ace inhibitor", "arb", "acei", "arbs"],
        "members": [
            "enalapril", "lisinopril", "ramipril", "captopril", "perindopril",
            "losartan", "valsartan", "candesartan", "telmisartan", "irbesartan"
        ],
        "hazards": ["HAZARD:HYPERKALEMIA_SEVERE", "HAZARD:ACUTE_RENAL_FAILURE"]
    },
    "CLASS:MRA": {
        "name": "Mineralocorticoid Receptor Antagonists",
        "atc": "C03DA",
        "synonyms": ["mra", "aldosterone antagonist", "potassium-sparing diuretic"],
        "members": [
            "spironolactone", "eplerenone"
        ],
        "hazards": ["HAZARD:HYPERKALEMIA_FATAL"]
    },
    "CLASS:BIGUANIDE_FULLDOSE": {
        "name": "Full-Dose Biguanides (Metformin >1000mg/day)",
        "atc": "A10BA02",
        "synonyms": ["full-dose metformin", "metformin >1000mg", "metformin 2000mg", "metformin 1000mg bd"],
        "members": [
            "metformin >1000mg/day", "full-dose metformin", "metformin 1000mg bd", "metformin 2000mg"
        ],
        "hazards": ["HAZARD:FATAL_LACTIC_ACIDOSIS"]
    },
    "CLASS:GABAPENTINOID_FULLDOSE": {
        "name": "Unadjusted Full-Dose Gabapentinoids in Renal Impairment",
        "atc": "N02BF",
        "synonyms": ["unadjusted pregabalin", "unadjusted gabapentin", "full-dose pregabalin", "full-dose gabapentin"],
        "members": [
            "pregabalin full-dose", "gabapentin full-dose", "pregabalin 300mg", "pregabalin 600mg"
        ],
        "hazards": ["HAZARD:MYOCLONUS_NEUROTOXICITY"]
    },
    "CLASS:PARACETAMOL": {
        "name": "Paracetamol / Acetaminophen (Analgesic)",
        "atc": "N02BE01",
        "synonyms": ["paracetamol", "acetaminophen", "apap"],
        "members": [
            "paracetamol", "acetaminophen"
        ],
        "hazards": ["HAZARD:HEPATOTOXICITY_OVERDOSE"]
    },
    "CLASS:SGLT2I": {
        "name": "Sodium-Glucose Cotransporter 2 Inhibitors (SGLT2i)",
        "atc": "A10BK",
        "synonyms": ["sglt2i", "sglt2 inhibitor", "gliflozin"],
        "members": [
            "dapagliflozin", "empagliflozin", "canagliflozin"
        ],
        "hazards": []
    },
    "CLASS:CCB_DIHYDROPYRIDINE": {
        "name": "Dihydropyridine Calcium Channel Blockers",
        "atc": "C08CA",
        "synonyms": ["ccb", "dihydropyridine ccb", "calcium channel blocker"],
        "members": [
            "amlodipine", "nifedipine", "felodipine"
        ],
        "hazards": []
    },
    "CLASS:PPI": {
        "name": "Proton Pump Inhibitors",
        "atc": "A02BC",
        "synonyms": ["ppi", "proton pump inhibitor"],
        "members": [
            "pantoprazole", "omeprazole", "esomeprazole", "lansoprazole", "rabeprazole"
        ],
        "hazards": []
    },
    "CLASS:ORAL_CORTICOSTEROID": {
        "name": "Oral Corticosteroids",
        "atc": "H02AB",
        "synonyms": ["oral corticosteroid", "oral steroid", "systemic steroid"],
        "members": [
            "prednisolone", "prednisone", "dexamethasone"
        ],
        "hazards": ["HAZARD:HYPERGLYCEMIA"]
    }
}

# Standardized Clinical Hazards & Priority Tiers
CLINICAL_HAZARDS: Dict[str, HazardConcept] = {
    "HAZARD:FATAL_BRONCHOSPASM": HazardConcept(
        id="HAZARD:FATAL_BRONCHOSPASM",
        name="Fatal Status Asthmaticus / Life-Threatening Bronchospasm",
        severity_tier=1,
        target_organ="Respiratory",
        description="Non-selective beta-2 blockade triggers immediate catastrophic airway constriction in reactive airway disease."
    ),
    "HAZARD:HEPATORENAL_SYNDROME": HazardConcept(
        id="HAZARD:HEPATORENAL_SYNDROME",
        name="Hepatorenal Syndrome / Refractory Renal Shutdown",
        severity_tier=1,
        target_organ="Hepato-Renal",
        description="NSAID inhibition of renal prostaglandins halts glomerular perfusion in cirrhotic portal hypertension."
    ),
    "HAZARD:FATAL_LACTIC_ACIDOSIS": HazardConcept(
        id="HAZARD:FATAL_LACTIC_ACIDOSIS",
        name="Metformin-Associated Lactic Acidosis (MALA)",
        severity_tier=1,
        target_organ="Metabolic / Systemic",
        description="Accumulation of metformin during reduced renal clearance leads to severe, frequently lethal metabolic acidosis."
    ),
    "HAZARD:CORONARY_VASOSPASM_STROKE": HazardConcept(
        id="HAZARD:CORONARY_VASOSPASM_STROKE",
        name="Ischemic Stroke or Myocardial Infarction from Vasospasm",
        severity_tier=1,
        target_organ="Cardiovascular / Neurovascular",
        description="5-HT1B/1D vasoconstriction in uncontrolled severe hypertension triggers acute coronary occlusive spasm or ischemic stroke."
    ),
    "HAZARD:CATATONIC_PARKINSONISM_FREEZE": HazardConcept(
        id="HAZARD:CATATONIC_PARKINSONISM_FREEZE",
        name="Irreversible Parkinsonian Motor Collapse / Bulbar Freezing",
        severity_tier=1,
        target_organ="Central Nervous System",
        description="Dopamine D2 antagonism produces acute, potentially fatal motor freezing, rigidity, and loss of airway protection."
    ),
    "HAZARD:FATAL_HYPOGLYCEMIC_COMA": HazardConcept(
        id="HAZARD:FATAL_HYPOGLYCEMIC_COMA",
        name="Catastrophic Hypoglycemic Coma and Fall-Induced Trauma",
        severity_tier=1,
        target_organ="Endocrine / Neurological",
        description="Delayed renal clearance of long-acting sulfonylureas in frail elderly causes refractory, prolonged hypoglycemia and fatal falls."
    ),
    "HAZARD:GI_BLEED": HazardConcept(
        id="HAZARD:GI_BLEED",
        name="Catastrophic Upper Gastrointestinal Hemorrhage",
        severity_tier=1,
        target_organ="Gastrointestinal",
        description="Synergistic gastric mucosal ulceration from concurrent NSAID + oral anticoagulation increases fatal rebleeding by 4-6x."
    ),
    "HAZARD:ACUTE_URINARY_RETENTION": HazardConcept(
        id="HAZARD:ACUTE_URINARY_RETENTION",
        name="Acute Complete Urinary Retention & Obstructive Uropathy",
        severity_tier=2,
        target_organ="Renal / Urological",
        description="Potent anticholinergic blockade inhibits detrusor muscle contraction in severe bladder outlet obstruction, requiring emergency catheterization."
    ),
    "HAZARD:ACUTE_RENAL_FAILURE": HazardConcept(
        id="HAZARD:ACUTE_RENAL_FAILURE",
        name="Acute-on-Chronic Kidney Injury",
        severity_tier=2,
        target_organ="Renal",
        description="Hemodynamic reduction in glomerular filtration rate precipitated by nephrotoxic insults."
    ),
    "HAZARD:HYPERKALEMIA_SEVERE": HazardConcept(
        id="HAZARD:HYPERKALEMIA_SEVERE",
        name="Severe Hyperkalemia (K+ >= 5.5 mmol/L)",
        severity_tier=2,
        target_organ="Cardiovascular / Electrophysiology",
        description="Impaired potassium excretion from concurrent RAAS blockade and renal insufficiency carrying risk of cardiac arrest."
    ),
    "HAZARD:HYPERKALEMIA_FATAL": HazardConcept(
        id="HAZARD:HYPERKALEMIA_FATAL",
        name="Fatal Hyperkalemic Arrhythmia / Asystole",
        severity_tier=1,
        target_organ="Cardiovascular / Electrophysiology",
        description="Precipitous rise in serum potassium causing peaked T waves, conduction block, and fatal asystole."
    ),
    "HAZARD:HEPATIC_ENCEPHALOPATHY": HazardConcept(
        id="HAZARD:HEPATIC_ENCEPHALOPATHY",
        name="Acute Precipitated Hepatic Encephalopathy",
        severity_tier=2,
        target_organ="Central Nervous System / Hepatic",
        description="Sedative opioids depress respiration, prolong intestinal transit, and cross the blood-brain barrier in portal shunting."
    ),
    "HAZARD:MYOCLONUS_NEUROTOXICITY": HazardConcept(
        id="HAZARD:MYOCLONUS_NEUROTOXICITY",
        name="Severe Gabapentinoid Myoclonus, Ataxia & Sedation",
        severity_tier=2,
        target_organ="Central Nervous System",
        description="Drug accumulation from 100% renal elimination in reduced GFR causes severe myoclonus, encephalopathy, and falls."
    ),
    "HAZARD:HEPATOTOXICITY_OVERDOSE": HazardConcept(
        id="HAZARD:HEPATOTOXICITY_OVERDOSE",
        name="Dose-Dependent Hepatotoxicity in Impaired Glutathione Reserves",
        severity_tier=2,
        target_organ="Hepatic",
        description="Doses exceeding 2.0g/24h in cirrhosis exhaust depleted hepatic glutathione, producing toxic NAPQI accumulation."
    ),
    "HAZARD:HF_DECOMPENSATION": HazardConcept(
        id="HAZARD:HF_DECOMPENSATION",
        name="Acute Heart Failure Decompensation & Sodium Retention",
        severity_tier=2,
        target_organ="Cardiovascular",
        description="NSAID-induced prostaglandin inhibition promotes renal sodium and water retention, blunting loop diuretic efficacy."
    )
}

# Standardized Clinical Conditions & Organ Vulnerabilities
CLINICAL_CONDITIONS: Dict[str, Dict] = {
    "COND:ASTHMA": {
        "name": "Bronchial Asthma / Reactive Airway Disease",
        "synonyms": ["asthma", "reactive airway", "bronchospasm", "wheezing"],
        "organ": "Respiratory",
        "fatal_sensitivities": ["CLASS:BETA_BLOCKER_NONSELECTIVE"]
    },
    "COND:CKD_ADVANCED": {
        "name": "Chronic Kidney Disease Stage 3b-5 (eGFR < 45 mL/min)",
        "synonyms": ["ckd", "stage 3b", "stage 4", "egfr <45", "egfr <30", "renal impairment", "renal insufficiency"],
        "organ": "Renal",
        "fatal_sensitivities": ["CLASS:NSAID", "CLASS:BIGUANIDE_FULLDOSE"]
    },
    "COND:CIRRHOSIS": {
        "name": "Liver Cirrhosis / End-Stage Liver Disease",
        "synonyms": ["cirrhosis", "liver cirrhosis", "chronic liver disease", "portal hypertension", "variceal"],
        "organ": "Hepatic",
        "fatal_sensitivities": ["CLASS:NSAID", "CLASS:OPIOID_SEDATING"]
    },
    "COND:HFREF": {
        "name": "Heart Failure with Reduced Ejection Fraction (HFrEF)",
        "synonyms": ["heart failure", "hfref", "ef <40", "reduced ejection fraction", "nyha class"],
        "organ": "Cardiovascular",
        "fatal_sensitivities": ["CLASS:NSAID"]
    },
    "COND:PUD_BLEED": {
        "name": "Peptic Ulcer Disease with Recent Hemorrhage on Anticoagulation",
        "synonyms": ["peptic ulcer", "gi bleed", "upper gi bleed", "ulcer hemorrhage", "anticoagulation"],
        "organ": "Gastrointestinal",
        "fatal_sensitivities": ["CLASS:NSAID"]
    },
    "COND:PARKINSONS": {
        "name": "Parkinson's Disease / Lewy Body Dementia",
        "synonyms": ["parkinson", "parkinson's", "parkinsonism", "tremor dominant"],
        "organ": "Neurological",
        "fatal_sensitivities": ["CLASS:D2_ANTAGONIST_ANTIPSYCHOTIC", "CLASS:D2_ANTAGONIST_ANTIEMETIC"]
    },
    "COND:SEVERE_HTN": {
        "name": "Severe / Stage 2 Uncontrolled Hypertension (BP >= 160/100)",
        "synonyms": ["stage 2 hypertension", "uncontrolled hypertension", "bp >=160", "bp >= 160/100"],
        "organ": "Cardiovascular",
        "fatal_sensitivities": ["CLASS:TRIPTAN"]
    },
    "COND:FRAIL_ELDERLY": {
        "name": "Frail Older Adult with High Fall / Dementia Risk",
        "synonyms": ["frail", "frailty", "fall risk", "older adult", "elderly >= 75"],
        "organ": "Geriatric",
        "fatal_sensitivities": ["CLASS:SULFONYLUREA_LONGACTING"]
    },
    "COND:SEVERE_BPH": {
        "name": "Severe Benign Prostatic Hyperplasia (IPSS > 19) / LUTS",
        "synonyms": ["bph", "benign prostatic hyperplasia", "luts", "lower urinary tract symptoms"],
        "organ": "Urological",
        "fatal_sensitivities": ["CLASS:TCA_ANTICHOLINERGIC"]
    },
    "COND:GOUT_ACUTE": {
        "name": "Acute Gouty Arthritis Flare",
        "synonyms": ["acute gout", "gout flare", "gouty arthritis"],
        "organ": "Musculoskeletal",
        "fatal_sensitivities": []
    },
    "COND:OSTEOARTHRITIS": {
        "name": "Degenerative Osteoarthritis",
        "synonyms": ["osteoarthritis", "knee osteoarthritis", "oa"],
        "organ": "Musculoskeletal",
        "fatal_sensitivities": []
    },
    "COND:MIGRAINE": {
        "name": "Acute Migraine Headache",
        "synonyms": ["migraine", "migraine headache", "acute migraine"],
        "organ": "Neurological",
        "fatal_sensitivities": []
    }
}


def normalize_token(text: str) -> str:
    """Standardizes string for regex matching."""
    return re.sub(r'[^\w\s-]', ' ', text.lower()).strip()


def extract_drug_classes_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Scans text for mentions of drug classes or individual member drugs.
    Returns list of (concept_id, matched_term).
    """
    found = []
    norm = normalize_token(text)

    for class_id, cdata in DRUG_CLASSES.items():
        # Check class synonyms
        for syn in cdata["synonyms"]:
            if re.search(r'\b' + re.escape(syn) + r'\b', norm):
                found.append((class_id, syn))
                break

        # Check member drugs
        for mem in cdata["members"]:
            if re.search(r'\b' + re.escape(mem) + r'\b', norm):
                found.append((class_id, mem))

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for cid, term in found:
        key = (cid, term)
        if key not in seen:
            seen.add(key)
            dedup.append((cid, term))
    return dedup


def extract_conditions_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Identifies clinical conditions and triggers mentioned in entry criteria or topics.
    Returns list of (cond_id, matched_term).
    """
    found = []
    norm = normalize_token(text)

    for cond_id, cdata in CLINICAL_CONDITIONS.items():
        for syn in cdata["synonyms"]:
            if re.search(r'\b' + re.escape(syn) + r'\b', norm):
                found.append((cond_id, syn))
                break

    seen = set()
    dedup = []
    for cid, term in found:
        if cid not in seen:
            seen.add(cid)
            dedup.append((cid, term))
    return dedup


def get_paracetamol_dose_ceiling(text: str) -> Optional[float]:
    """
    Detects daily paracetamol dose mentions in grams (e.g., 2g, 2.0g, 3g, 4g).
    """
    norm = text.lower()
    m = re.search(r'paracetamol[^\n.]*?(\d+(?:\.\d+)?)\s*g(?:/day|\s*daily|\s*per 24 hours)?', norm)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Secondary check for 2000mg / 3000mg
    m_mg = re.search(r'paracetamol[^\n.]*?(\d{4})\s*mg', norm)
    if m_mg:
        try:
            return float(m_mg.group(1)) / 1000.0
        except ValueError:
            pass
    return None
