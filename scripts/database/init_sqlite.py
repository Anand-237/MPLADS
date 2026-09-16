"""
MPLADS Sentinel SQLite Database Migration & Ingestion Script

This script populates SQLite database `data/mplads_database.db` from processed CSV datasets
and sets up optimized B-Tree indexes for fast queries in Streamlit.
"""

import os
import sys
import time
import sqlite3
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
AUDIT_DIR = os.path.join(DATA_DIR, "audit")
DB_PATH = os.path.join(DATA_DIR, "mplads_database.db")

def init_sqlite_db():
    print("=" * 80)
    print("           MPLADS SENTINEL SQLITE DATABASE MIGRATION & INGESTION          ")
    print("=" * 80)
    
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"\n[1] CONNECTING TO SQLITE DATABASE: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Ingest Master Project Anomaly Results Dataset
    master_csv = os.path.join(PROCESSED_DIR, "mplads_anomaly_results.csv")
    if os.path.exists(master_csv):
        print(f"\n[2] INGESTING MASTER PROJECTS DATASET ({master_csv})...")
        df_master = pd.read_csv(master_csv, low_memory=False)
        df_master.to_sql("mplads_anomaly_results", conn, if_exists="replace", index=False)
        print(f"  - Ingested {len(df_master):,} records into table 'mplads_anomaly_results'.")
    else:
        print(f"  - [WARNING] Dataset missing: {master_csv}")
        
    # 2. Ingest MP Summary Dataset
    mp_csv = os.path.join(PROCESSED_DIR, "mp_summary_clean.csv")
    if os.path.exists(mp_csv):
        print(f"\n[3] INGESTING MP SUMMARY DATASET ({mp_csv})...")
        df_mp = pd.read_csv(mp_csv, low_memory=False)
        df_mp.to_sql("mp_summary_clean", conn, if_exists="replace", index=False)
        print(f"  - Ingested {len(df_mp):,} records into table 'mp_summary_clean'.")
        
    # 3. Ingest Potentially Similar Projects Dataset
    sim_csv = os.path.join(PROCESSED_DIR, "potentially_similar_projects.csv")
    if os.path.exists(sim_csv):
        print(f"\n[4] INGESTING SIMILAR PROJECTS CANDIDATES ({sim_csv})...")
        df_sim = pd.read_csv(sim_csv, low_memory=False)
        df_sim.to_sql("potentially_similar_projects", conn, if_exists="replace", index=False)
        print(f"  - Ingested {len(df_sim):,} records into table 'potentially_similar_projects'.")
        
    # 4. Ingest / Initialize Review Log Table
    audit_csv = os.path.join(AUDIT_DIR, "review_log.csv")
    if os.path.exists(audit_csv):
        print(f"\n[5] INGESTING HUMAN AUDIT REVIEW LOG ({audit_csv})...")
        df_audit = pd.read_csv(audit_csv, low_memory=False)
        df_audit.to_sql("review_log", conn, if_exists="replace", index=False)
        print(f"  - Ingested {len(df_audit):,} review records into table 'review_log'.")
    else:
        print(f"\n[5] INITIALIZING EMPTY AUDIT REVIEW LOG TABLE 'review_log'...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            log_id TEXT PRIMARY KEY,
            work_id TEXT,
            review_status TEXT,
            reviewer_notes TEXT,
            reviewer_id TEXT,
            review_timestamp TEXT,
            ai_risk_score_at_review REAL,
            ai_risk_category_at_review TEXT
        )
        """)
        conn.commit()
        
    # 5. Create High-Performance B-Tree Indexes
    print("\n[6] CREATING DATABASE INDEXES FOR FAST QUERYING...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_proj_work_id ON mplads_anomaly_results(work_id)",
        "CREATE INDEX IF NOT EXISTS idx_proj_state ON mplads_anomaly_results(state)",
        "CREATE INDEX IF NOT EXISTS idx_proj_constituency ON mplads_anomaly_results(constituency)",
        "CREATE INDEX IF NOT EXISTS idx_proj_risk ON mplads_anomaly_results(risk_category)",
        "CREATE INDEX IF NOT EXISTS idx_audit_work_id ON review_log(work_id)"
    ]
    for idx_sql in indexes:
        cursor.execute(idx_sql)
    conn.commit()
    conn.close()
    
    t1 = time.time()
    db_size = os.path.getsize(DB_PATH)
    
    print(f"\n" + "=" * 80)
    print(f"  SQLITE MIGRATION COMPLETED SUCCESSFULLY ({t1 - t0:.2f} seconds)")
    print(f"  Database File: {DB_PATH} ({db_size / (1024*1024):,.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    init_sqlite_db()
