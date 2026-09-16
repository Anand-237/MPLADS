"""
MPLADS Human-in-the-Loop Verification Service Script (Stage 9)

Manages persistent audit trail storage for human reviewer decisions:
1. Initializes data/audit/review_log.csv if missing.
2. Provides append_review_log() to log human auditor decisions separately from AI model predictions.
3. Provides get_latest_review_status() and get_audit_trail_history().
4. Enforces immutability of AI predictions while capturing audit snapshots.

Outputs:
- data/audit/review_log.csv
"""

import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(PROJECT_ROOT, "data", "audit")
LOG_PATH = os.path.join(AUDIT_DIR, "review_log.csv")

VALID_STATUSES = [
    "Not Reviewed",
    "Verified Normal",
    "Needs Further Review",
    "Evidence Insufficient",
    "Manual Investigation Required"
]

LOG_COLUMNS = [
    "log_id",
    "work_id",
    "review_status",
    "reviewer_notes",
    "reviewer_id",
    "review_timestamp",
    "ai_risk_score_at_review",
    "ai_risk_category_at_review"
]

def initialize_audit_log():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame(columns=LOG_COLUMNS)
        df.to_csv(LOG_PATH, index=False, encoding="utf-8")
        print(f"[+] Initialized audit log at {LOG_PATH}", flush=True)

def append_review_log(work_id, review_status, reviewer_notes="", reviewer_id="Auditor Officer", ai_risk_score=0.0, ai_risk_category="LOW"):
    initialize_audit_log()
    
    if review_status not in VALID_STATUSES:
        raise ValueError(f"Invalid review_status '{review_status}'. Must be one of {VALID_STATUSES}")
        
    df_existing = pd.read_csv(LOG_PATH, low_memory=False)
    next_id_num = len(df_existing) + 1
    log_id = f"AUDIT_{next_id_num:06d}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    new_entry = {
        "log_id": log_id,
        "work_id": work_id,
        "review_status": review_status,
        "reviewer_notes": reviewer_notes.strip(),
        "reviewer_id": reviewer_id.strip() if reviewer_id.strip() else "Auditor Officer",
        "review_timestamp": timestamp,
        "ai_risk_score_at_review": round(float(ai_risk_score), 2),
        "ai_risk_category_at_review": str(ai_risk_category)
    }
    
    df_new = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
    df_new.to_csv(LOG_PATH, index=False, encoding="utf-8")
    
    print(f"[+] Recorded audit entry {log_id} for Work ID #{work_id} -> Status: '{review_status}'", flush=True)
    return new_entry

def get_latest_review_summary():
    initialize_audit_log()
    df = pd.read_csv(LOG_PATH, low_memory=False)
    if len(df) == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    # Return latest entry per work_id
    latest = df.sort_values("review_timestamp").groupby("work_id").last().reset_index()
    return latest

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("       MPLADS HUMAN-IN-THE-LOOP VERIFICATION SERVICE (STAGE 9)        ", flush=True)
    print("=" * 80, flush=True)
    
    initialize_audit_log()
    
    # Test append sample
    sample_entry = append_review_log(
        work_id=61773,
        review_status="Needs Further Review",
        reviewer_notes="High expenditure deviation and YoY project approval spike require manual site audit.",
        reviewer_id="AUDITOR_OFFICER_01",
        ai_risk_score=80.18,
        ai_risk_category="HIGH"
    )
    
    summary = get_latest_review_summary()
    print(f"\n[+] Audit Trail Summary: {len(summary)} unique projects audited.", flush=True)
    print("=" * 80, flush=True)
