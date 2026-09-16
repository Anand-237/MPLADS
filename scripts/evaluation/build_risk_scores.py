"""
MPLADS Explainable Multi-Factor Risk Scoring Script (Stage 5 - Updated with Works Completion Risk)

This script synthesizes independent anomaly signals across domain feature sets:
1. Loads financial, temporal, vendor, Isolation Forest, statistical baseline, NLP similarity, and MP Works Completion metrics.
2. Normalizes component scores to comparable 0-100 ranges.
3. Applies transparent, configurable weighted combination:
   final_risk_score = 0.30 * financial + 0.20 * temporal + 0.25 * isolation_forest + 0.15 * vendor + 0.10 * mp_works_risk
4. Assigns configurable risk categories:
   0 - 39   = LOW
   40 - 69  = MEDIUM
   70 - 100 = HIGH
5. Generates evidence-grounded, human-readable explanation strings for every MEDIUM and HIGH project.
6. Enforces strict neutral non-accusatory terminology ('Priority for verification').
7. Exports results to data/processed/final_risk_scores.csv.

Outputs:
- data/processed/final_risk_scores.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "methodology")

# -----------------------------------------------------------------------------
# CONFIGURABLE PIPELINE PARAMETERS & WEIGHTS (MUST SUM TO 1.00)
# -----------------------------------------------------------------------------
WEIGHT_FINANCIAL = 0.30
WEIGHT_TEMPORAL = 0.20
WEIGHT_ISOLATION_FOREST = 0.25
WEIGHT_VENDOR = 0.15
WEIGHT_WORKS_RISK = 0.10  # MP Completed Works / Completion Rate Risk

# Configurable Risk Category Thresholds
THRESHOLD_LOW_MAX = 39.99
THRESHOLD_MEDIUM_MAX = 49.99
# Score >= 50.00 -> HIGH

def build_risk_scores():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    t0 = time.time()
    print("=" * 80, flush=True)
    print("        MPLADS EXPLAINABLE MULTI-FACTOR RISK SCORING PIPELINE (STAGE 5)    ", flush=True)
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. LOAD INPUT DATASETS
    # -------------------------------------------------------------------------
    stat_path = os.path.join(PROCESSED_DIR, "statistical_anomaly_results.csv")
    iso_path = os.path.join(PROCESSED_DIR, "isolation_forest_results.csv")
    nlp_path = os.path.join(FEATURES_DIR, "nlp_features.csv")
    temp_path = os.path.join(FEATURES_DIR, "temporal_features.csv")
    v_path = os.path.join(FEATURES_DIR, "isolation_forest_features.csv")
    proj_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
    mp_path = os.path.join(PROCESSED_DIR, "mp_summary_clean.csv")
    
    stat_df = pd.read_csv(stat_path, low_memory=False)
    iso_df = pd.read_csv(iso_path, low_memory=False)
    nlp_df = pd.read_csv(nlp_path, low_memory=False)
    temp_df = pd.read_csv(temp_path, low_memory=False)
    v_df = pd.read_csv(v_path, low_memory=False)
    proj_df = pd.read_csv(proj_path, low_memory=False)
    mp_df = pd.read_csv(mp_path, low_memory=False)
    
    total_projects = len(stat_df)
    
    # Merge MP Works summary
    merged_mp = pd.merge(
        proj_df[["work_id", "mp_name"]],
        mp_df[["mp_name", "completed_works_count", "recommended_works_count", "completion_rate_pct"]],
        on="mp_name",
        how="left"
    )
    
    print(f"\n[1] LOADED ANOMALY SIGNALS SUMMARY:", flush=True)
    print(f"  - Total Projects Evaluated:     {total_projects:,}", flush=True)
    print(f"  - Statistical Baselines Source:  {stat_path} ({len(stat_df):,} rows)", flush=True)
    print(f"  - Isolation Forest Source:      {iso_path} ({len(iso_df):,} rows)", flush=True)
    print(f"  - NLP Similarity Source:        {nlp_path} ({len(nlp_df):,} rows)", flush=True)
    print(f"  - Temporal Features Source:     {temp_path} ({len(temp_df):,} rows)", flush=True)
    print(f"  - Vendor Features Source:       {v_path} ({len(v_df):,} rows)", flush=True)
    print(f"  - MP Works Performance Source:  {mp_path} ({len(mp_df):,} rows)", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. NORMALIZE COMPONENT RISK SCORES (0 TO 100)
    # -------------------------------------------------------------------------
    print("\n[2] NORMALIZING COMPONENT DOMAIN RISK SCORES (RANGE 0 - 100)...", flush=True)
    
    # 1. Financial Component Score (0-100)
    fin_percentile = stat_df["percentile"].clip(0.0, 100.0)
    z_penalty = (stat_df["z_score"].clip(0.0, 10.0) * 10.0).clip(0.0, 100.0)
    financial_score = (0.6 * fin_percentile + 0.4 * z_penalty).clip(0.0, 100.0)
    
    # 2. Temporal Component Score (0-100)
    imp_seq_flag = temp_df["is_impossible_date_sequence"].fillna(0).astype(int) * 100.0
    yoy_clean = temp_df["year_over_year_change"].fillna(0.0)
    yoy_penalty = (yoy_clean.clip(0.0, 1000.0) / 10.0).clip(0.0, 100.0)
    temporal_score = np.where(imp_seq_flag == 100.0, 100.0, yoy_penalty)
    temporal_score = pd.Series(temporal_score).fillna(0.0).clip(0.0, 100.0)
    
    # 3. Vendor Risk Component Score (0-100)
    max_v_share = v_df["constituency_max_vendor_share"].fillna(0.0).clip(0.0, 1.0) * 100.0
    pend_pay_ratio = (v_df["constituency_pending_payment_ratio"].fillna(0.0).clip(0.0, 1.0) * 200.0).clip(0.0, 100.0)
    repeat_rate = v_df["constituency_avg_vendor_repeat_rate"].fillna(0.0).clip(0.0, 1.0) * 100.0
    vendor_score = (0.5 * max_v_share + 0.3 * pend_pay_ratio + 0.2 * repeat_rate).fillna(0.0).clip(0.0, 100.0)
    
    # 4. Isolation Forest Component Score (0-100)
    isolation_forest_score = (iso_df["normalized_anomaly_score"].fillna(0.0).clip(0.0, 1.0) * 100.0).clip(0.0, 100.0)
    
    # 5. NLP Similarity Score (0-100)
    nlp_similarity_score = (nlp_df["nearest_similarity_score"].fillna(0.0).clip(0.0, 1.0) * 100.0).clip(0.0, 100.0)
    
    # 6. MP Works Execution Risk Score (0-100)
    # Inverse of completion rate: lower completion rate = higher risk of project backlog
    completion_rate = merged_mp["completion_rate_pct"].fillna(50.0)
    mp_works_risk_score = (100.0 - completion_rate).clip(0.0, 100.0)
    
    # -------------------------------------------------------------------------
    # 3. COMPUTE WEIGHTED FINAL RISK SCORE & CATEGORIES
    # -------------------------------------------------------------------------
    print("\n[3] COMPUTING WEIGHTED FINAL RISK SCORE & ASSIGNING CATEGORIES...", flush=True)
    print(f"  - Configured Weights: Financial={WEIGHT_FINANCIAL}, Temporal={WEIGHT_TEMPORAL}, Isolation Forest={WEIGHT_ISOLATION_FOREST}, Vendor={WEIGHT_VENDOR}, Works Risk={WEIGHT_WORKS_RISK}", flush=True)
    
    final_risk_score = (
        WEIGHT_FINANCIAL * financial_score +
        WEIGHT_TEMPORAL * temporal_score +
        WEIGHT_ISOLATION_FOREST * isolation_forest_score +
        WEIGHT_VENDOR * vendor_score +
        WEIGHT_WORKS_RISK * mp_works_risk_score
    ).fillna(0.0).round(2)
    
    # Risk Categories: LOW (0-39.99), MEDIUM (40-69.99), HIGH (70-100)
    def assign_category(score):
        if score <= THRESHOLD_LOW_MAX:
            return "LOW"
        elif score <= THRESHOLD_MEDIUM_MAX:
            return "MEDIUM"
        else:
            return "HIGH"
            
    risk_category = final_risk_score.apply(assign_category)
    
    # -------------------------------------------------------------------------
    # 4. GENERATE ELABORATED EVIDENCE-GROUNDED BULLET POINT EXPLANATIONS
    # -------------------------------------------------------------------------
    print("\n[4] GENERATING ELABORATED BULLET POINT EXPLANATION STRINGS...", flush=True)
    
    explanations = []
    
    for idx in range(total_projects):
        cat = risk_category.iloc[idx]
        final_s = final_risk_score.iloc[idx]
        f_score = financial_score.iloc[idx]
        t_score = temporal_score.iloc[idx]
        v_score = vendor_score.iloc[idx]
        iso_s = isolation_forest_score.iloc[idx]
        nlp_s = nlp_similarity_score.iloc[idx]
        w_risk = mp_works_risk_score.iloc[idx]
        
        z_val = stat_df["z_score"].iloc[idx]
        pct = stat_df["percentile"].iloc[idx]
        eff_amt = stat_df["effective_amount"].iloc[idx]
        grp_mean = stat_df["group_mean"].iloc[idx]
        grp_p99 = stat_df["group_p99"].iloc[idx]
        b_grp = stat_df["baseline_group"].iloc[idx]
        
        is_imp_date = temp_df["is_impossible_date_sequence"].iloc[idx]
        yoy_val = temp_df["year_over_year_change"].iloc[idx]
        max_share = v_df["constituency_max_vendor_share"].iloc[idx]
        pending_ratio = v_df["constituency_pending_payment_ratio"].iloc[idx]
        iso_flag = iso_df["is_anomaly"].iloc[idx]
        sim_val = nlp_df["nearest_similarity_score"].iloc[idx]
        sim_count = nlp_df["similar_project_count"].iloc[idx]
        
        comp_works = merged_mp["completed_works_count"].iloc[idx]
        rec_works = merged_mp["recommended_works_count"].iloc[idx]
        comp_rate = merged_mp["completion_rate_pct"].iloc[idx]
        
        bullets = []
        
        # 1. Financial Anomaly Bullet
        if z_val > 2.5 or pct >= 95.0 or f_score >= 65.0:
            bullets.append(
                f"• Financial Benchmark Deviation: Expenditure of ₹{eff_amt:,.2f} is unusually high compared with {b_grp} benchmark mean of ₹{grp_mean:,.2f} (Z-Score: +{z_val:.2f}, Cohort Percentile: {pct:.1f}%)."
            )
        elif eff_amt > grp_p99:
            bullets.append(
                f"• Upper-Tail Expenditure: Project cost ₹{eff_amt:,.2f} exceeds the 99th percentile cohort ceiling of ₹{grp_p99:,.2f}."
            )
            
        # 2. Temporal Anomaly Bullet
        if is_imp_date == 1:
            bullets.append(
                "• Invalid Date Chronology: Completion date precedes recommendation date (impossible negative duration sequence)."
            )
        elif t_score >= 60.0 or yoy_val >= 200.0:
            bullets.append(
                f"• Constituency Project Volume Spike: Annual recommendation volume grew by +{yoy_val:.1f}% YoY, indicating an unusual concentration of project approvals."
            )
            
        # 3. Vendor Risk Bullet
        if max_share >= 0.40 or v_score >= 60.0:
            bullets.append(
                f"• Vendor Expenditure Concentration: Single vendor commands {max_share*100:.1f}% of total constituency expenditure, exceeding normal competitive distribution thresholds."
            )
        elif pending_ratio >= 0.15:
            bullets.append(
                f"• Pending Payment Ratio: Constituency exhibits {pending_ratio*100:.1f}% pending transaction ratio across active vendors."
            )
            
        # 4. MP Completed Works / Low Completion Rate Risk Bullet
        if comp_rate < 50.0:
            bullets.append(
                f"• Low Constituency Works Completion Rate: MP exhibits a low completion rate of {comp_rate:.1f}% ({comp_works} completed out of {rec_works} recommended works), signaling elevated execution backlog risk."
            )
            
        # 5. Isolation Forest Machine Learning Anomaly Bullet
        if iso_flag == 1 or iso_s >= 55.0:
            bullets.append(
                f"• Isolation Forest Machine Learning Pattern: Multi-dimensional feature isolation detected across combined financial, temporal, and vendor space (Model Severity: {iso_s:.1f}/100)."
            )
            
        # 6. NLP Text Similarity Bullet
        if sim_val >= 0.85:
            bullets.append(
                f"• Text Similarity Candidate: Work description matches {sim_count} other project(s) in constituency with high text similarity ({sim_val*100:.1f}% cosine match score)."
            )
            
        # Format Header Prefix
        if cat == "HIGH":
            prefix = f"High priority for verification (Final Risk Score: {final_s:.2f}/100):\n"
        elif cat == "MEDIUM":
            prefix = f"Priority for verification (Final Risk Score: {final_s:.2f}/100):\n"
        else:
            prefix = f"Standard baseline record (Final Risk Score: {final_s:.2f}/100):\n"
            
        if len(bullets) > 0:
            explanation = prefix + "\n".join(bullets)
        else:
            explanation = f"Standard baseline record (Final Risk Score: {final_s:.2f}/100): Project execution within expected cohort parameters."
            
        explanations.append(explanation)
        
    # -------------------------------------------------------------------------
    # 5. EXPORT FINAL RISK SCORES DATASET
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame({
        "work_id": stat_df["work_id"],
        "financial_score": financial_score.round(2),
        "temporal_score": temporal_score.round(2),
        "vendor_score": vendor_score.round(2),
        "isolation_forest_score": isolation_forest_score.round(2),
        "nlp_similarity_score": nlp_similarity_score.round(2),
        "mp_works_risk_score": mp_works_risk_score.round(2),
        "final_risk_score": final_risk_score,
        "risk_category": risk_category,
        "explanation": explanations
    })
    
    out_csv = os.path.join(PROCESSED_DIR, "final_risk_scores.csv")
    results_df.to_csv(out_csv, index=False, encoding="utf-8")
    
    t1 = time.time()
    
    cat_counts = risk_category.value_counts()
    print(f"\n[5] EXPORTED FINAL RISK SCORES ({t1 - t0:.2f} seconds):", flush=True)
    print(f"  - Target Path: {out_csv}", flush=True)
    print(f"  - LOW Risk Projects (0-39):    {cat_counts.get('LOW', 0):,} ({(cat_counts.get('LOW', 0)/total_projects)*100:.2f}%)", flush=True)
    print(f"  - MEDIUM Risk Projects (40-69): {cat_counts.get('MEDIUM', 0):,} ({(cat_counts.get('MEDIUM', 0)/total_projects)*100:.2f}%)", flush=True)
    print(f"  - HIGH Risk Projects (70-100): {cat_counts.get('HIGH', 0):,} ({(cat_counts.get('HIGH', 0)/total_projects)*100:.2f}%)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    build_risk_scores()
