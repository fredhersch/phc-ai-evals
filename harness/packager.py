"""
packager.py - Compiles certified a-CDUs into offline Edge artifacts
Generates SQLite databases (with FTS5 search) for Android/eCHIS and single-file bundle JSONs.
"""

import json
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from .models import ClinicalDecisionUnit


def export_to_sqlite(cdus: List[ClinicalDecisionUnit], db_path: str) -> str:
    """
    Compiles a library of a-CDUs into a lightweight, indexed SQLite database
    optimized for offline frontline deployment (e.g., Kenya eCHIS on Android).
    """
    out_file = Path(db_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    conn = sqlite3.connect(str(out_file))
    cur = conn.cursor()

    # Main relational table
    cur.execute("""
        CREATE TABLE cdus (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            source_guideline TEXT NOT NULL,
            entry_criteria TEXT NOT NULL,
            mandated_actions_json TEXT NOT NULL,
            negative_boundaries_json TEXT NOT NULL,
            priority_tier INTEGER NOT NULL,
            prompt_chunk TEXT NOT NULL,
            estimated_tokens INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            version TEXT NOT NULL,
            certified INTEGER NOT NULL
        )
    """)

    # Fast indexes
    cur.execute("CREATE INDEX idx_cdu_priority ON cdus(priority_tier)")

    # Full-Text Search 5 (FTS5) table for instant offline micro-retrieval
    cur.execute("""
        CREATE VIRTUAL TABLE cdu_fts USING fts5(
            id,
            topic,
            entry_criteria,
            negative_boundaries
        )
    """)

    for c in cdus:
        actions_json = json.dumps(c.mandated_actions)
        neg_json = json.dumps(c.negative_boundaries)
        prompt_chunk = c.to_prompt_chunk()
        tokens = c.estimate_tokens()
        chash = c.compute_hash()

        cur.execute("""
            INSERT INTO cdus VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c.id, c.topic, c.source_guideline, c.entry_criteria,
            actions_json, neg_json, c.priority_tier, prompt_chunk,
            tokens, chash, c.version, 1 if c.certified else 0
        ))

        cur.execute("""
            INSERT INTO cdu_fts (id, topic, entry_criteria, negative_boundaries)
            VALUES (?, ?, ?, ?)
        """, (
            c.id, c.topic, c.entry_criteria, " ".join(c.negative_boundaries)
        ))

    conn.commit()
    conn.close()
    return str(out_file)


def export_to_bundle_json(cdus: List[ClinicalDecisionUnit], output_path: str) -> str:
    """
    Packages CDUs into a cryptographically verified JSON registry bundle.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    items = [c.to_dict() for c in cdus]
    canonical_items = json.dumps(items, sort_keys=True)
    master_checksum = hashlib.sha256(canonical_items.encode("utf-8")).hexdigest()

    bundle = {
        "registry_name": "PHC-aCDU-Certified-Registry",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_units": len(cdus),
        "master_sha256": master_checksum,
        "cdus": items
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    return str(out_file)
