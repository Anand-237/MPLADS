"""
MPLADS Master Anomaly Results Assembler Script (Stage 8 Pre-requisite)

This script compiles the master integrated anomaly results dataset:
- Combines data/processed/final_risk_scores.csv, statistical_anomaly_results.csv,
  isolation_forest_results.csv, temporal_features.csv, vendor_features.csv,
  nlp_features.csv, projects_clean.csv, and mp_summary_clean.csv.
- Generates data/processed/mplads_anomaly_results.csv.

Outputs:
- data/processed/mplads_anomaly_results.csv
"""

import os
import sys
import time
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

def build_master_anomaly_results():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("      BUILDING MASTER INTEGRATED ANOMALY RESULTS DATASET (STAGE 8)      ", flush=True)
    print("=" * 80, flush=True)
    
    # Input File Paths
    proj_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
    mp_path = os.path.join(PROCESSED_DIR, "mp_summary_clean.csv")
    risk_path = os.path.join(PROCESSED_DIR, "final_risk_scores.csv")
    stat_path = os.path.join(PROCESSED_DIR, "statistical_anomaly_results.csv")
    iso_path = os.path.join(PROCESSED_DIR, "isolation_forest_results.csv")
    temp_path = os.path.join(FEATURES_DIR, "temporal_features.csv")
    v_path = os.path.join(FEATURES_DIR, "isolation_forest_features.csv")
    nlp_path = os.path.join(FEATURES_DIR, "nlp_features.csv")
    fin_path = os.path.join(FEATURES_DIR, "financial_features.csv")
    
    print("[1] READING COMPONENT DATASETS...", flush=True)
    proj_df = pd.read_csv(proj_path, low_memory=False)
    mp_df = pd.read_csv(mp_path, low_memory=False)
    risk_df = pd.read_csv(risk_path, low_memory=False)
    stat_df = pd.read_csv(stat_path, low_memory=False)
    iso_df = pd.read_csv(iso_path, low_memory=False)
    temp_df = pd.read_csv(temp_path, low_memory=False)
    v_df = pd.read_csv(v_path, low_memory=False)
    nlp_df = pd.read_csv(nlp_path, low_memory=False)
    fin_df = pd.read_csv(fin_path, low_memory=False)
    
    total_projects = len(proj_df)
    print(f"  - Total Projects: {total_projects:,}", flush=True)
    
    # Merge MP summary for MP-level allocated amount, total expenditure, and works statistics
    merged_mp = pd.merge(
        proj_df[["work_id", "mp_name"]],
        mp_df[["mp_name", "allocated_amount", "total_expenditure", "utilization_pct", "completed_works_count", "recommended_works_count", "completion_rate_pct"]],
        on="mp_name",
        how="left"
    )
    
    # Assemble Integrated Master DataFrame
    master_df = pd.DataFrame({
        "work_id": proj_df["work_id"],
        "mp_name": proj_df["mp_name"].fillna("Unspecified MP"),
        "project_status": proj_df["project_status"],
        "state": proj_df["state"],
        "constituency": proj_df["constituency"],
        "category": proj_df["category"].fillna("Normal/Others"),
        "work_description": proj_df["work_description"].fillna("No description provided"),
        "recommended_amount": fin_df["recommended_amount"].fillna(0.0),
        "sanctioned_amount": fin_df["final_amount"].fillna(0.0),
        "effective_amount": stat_df["effective_amount"].fillna(0.0),
        "mp_allocated_amount": merged_mp["allocated_amount"].fillna(0.0),
        "mp_total_expenditure": merged_mp["total_expenditure"].fillna(0.0),
        "mp_utilization_pct": merged_mp["utilization_pct"].fillna(0.0),
        "mp_completed_works": merged_mp["completed_works_count"].fillna(0).astype(int),
        "mp_recommended_works": merged_mp["recommended_works_count"].fillna(0).astype(int),
        "mp_completion_rate_pct": merged_mp["completion_rate_pct"].fillna(0.0),
        "recommendation_date": proj_df["recommendation_date"].fillna("N/A"),
        "sanction_date": "N/A",
        "completion_date": proj_df["completed_date"].fillna("N/A"),
        "duration_days": temp_df["project_duration_days"].fillna(0),
        "recommendation_year": temp_df["recommendation_year"].fillna(2022).astype(int),
        "is_impossible_date_sequence": temp_df["is_impossible_date_sequence"].fillna(0).astype(int),
        "constituency_max_vendor_share": v_df["constituency_max_vendor_share"].fillna(0.0),
        "constituency_pending_payment_ratio": v_df["constituency_pending_payment_ratio"].fillna(0.0),
        "isolation_forest_score": iso_df["normalized_anomaly_score"].round(4),
        "baseline_group": stat_df["baseline_group"],
        "group_mean": stat_df["group_mean"].round(2),
        "group_median": stat_df["group_median"].round(2),
        "group_p99": stat_df["group_p99"].round(2),
        "cost_z_score": stat_df["z_score"].round(2),
        "percentile": stat_df["percentile"].round(1),
        "statistical_flags": stat_df["statistical_flag"].astype(int),
        "nearest_similarity_score": nlp_df["nearest_similarity_score"].round(4),
        "similar_project_count": nlp_df["similar_project_count"].astype(int),
        "financial_score": risk_df["financial_score"],
        "temporal_score": risk_df["temporal_score"],
        "vendor_score": risk_df["vendor_score"],
        "final_risk_score": risk_df["final_risk_score"],
        "risk_category": risk_df["risk_category"],
        "explanation": risk_df["explanation"],
        "verification_status": "Not Reviewed"
    })
    
    out_path = os.path.join(PROCESSED_DIR, "mplads_anomaly_results.csv")
    master_df.to_csv(out_path, index=False, encoding="utf-8")
    
    t1 = time.time()
    print(f"\n[2] MASTER ANOMALY DATASET EXPORTED SUCCESSFULLY ({t1 - t0:.2f} seconds):", flush=True)
    print(f"  - Target Path: {out_path}", flush=True)
    print(f"  - Dimensions:  {len(master_df):,} rows x {len(master_df.columns)} columns", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    build_master_anomaly_results()
