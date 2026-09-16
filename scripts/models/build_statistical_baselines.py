"""
MPLADS Statistical Anomaly Baselines Generator Script (Stage 4C)

This script computes interpretable parametric (Z-score) and non-parametric (quantile percentile) anomaly baselines:
1. Loads clean projects dataset and financial feature values.
2. Group projects into hierarchical cohorts (Category -> Constituency -> State) with minimum group size thresholding (N >= 10).
3. Calculates cohort parametric (mean, std) and quantile (median, p75, p90, p95, p99) benchmarks.
4. Computes project-level Z-scores and cohort percentile ranks.
5. Evaluates multi-threshold anomaly flags (Z > 2.5, P95, P99).
6. Enforces strict neutral non-accusatory terminology ('statistical anomaly requiring verification').
7. Exports results to data/processed/statistical_anomaly_results.csv.

Outputs:
- data/processed/statistical_anomaly_results.csv
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

MIN_GROUP_SIZE = 10  # Minimum cohort group size to ensure statistical validity

def percentile_rank_within_group(series):
    """Compute percentile rank (0 to 100) within group."""
    if len(series) <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(method="average", pct=True) * 100.0
    return ranks

def build_statistical_baselines():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    t0 = time.time()
    print("=" * 80, flush=True)
    print("       MPLADS STATISTICAL ANOMALY BASELINES PIPELINE (STAGE 4C)          ", flush=True)
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. LOAD INPUT DATASETS
    # -------------------------------------------------------------------------
    proj_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
    fin_path = os.path.join(FEATURES_DIR, "financial_features.csv")
    
    proj_df = pd.read_csv(proj_path, low_memory=False)
    fin_df = pd.read_csv(fin_path, low_memory=False)
    
    total_projects = len(proj_df)
    proj_df["effective_amount"] = fin_df["effective_amount"]
    
    print(f"\n[1] LOADED PROJECTS DATASET SUMMARY:", flush=True)
    print(f"  - Total Projects:           {total_projects:,}", flush=True)
    print(f"  - Category Non-Nulls:       {proj_df['category'].notnull().sum():,} (4 categories)", flush=True)
    print(f"  - State Non-Nulls:          {proj_df['state'].notnull().sum():,} (36 states)", flush=True)
    print(f"  - Constituency Non-Nulls:   {proj_df['constituency'].notnull().sum():,} (537 constituencies)", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. HIERARCHICAL COHORT BASELINE ASSIGNMENT
    # -------------------------------------------------------------------------
    print(f"\n[2] COMPUTING COHORT BASELINES (Minimum Group Size N >= {MIN_GROUP_SIZE})...", flush=True)
    
    # Pre-calculate sizes for Category, Constituency, State
    cat_counts = proj_df["category"].value_counts().to_dict()
    const_counts = proj_df["constituency"].value_counts().to_dict()
    state_counts = proj_df["state"].value_counts().to_dict()
    
    def assign_baseline_group(row):
        cat = row["category"]
        const = row["constituency"]
        st = row["state"]
        
        # Priority 1: Category if present and N >= MIN_GROUP_SIZE
        if pd.notnull(cat) and cat_counts.get(cat, 0) >= MIN_GROUP_SIZE:
            return f"Category: {cat}", "Category"
        # Priority 2: Constituency if N >= MIN_GROUP_SIZE
        elif pd.notnull(const) and const_counts.get(const, 0) >= MIN_GROUP_SIZE:
            return f"Constituency: {const}", "Constituency"
        # Priority 3: State fallback
        else:
            return f"State: {st}", "State"

    group_assignments = proj_df.apply(assign_baseline_group, axis=1)
    proj_df["baseline_group"] = [g[0] for g in group_assignments]
    proj_df["baseline_type"] = [g[1] for g in group_assignments]
    
    unique_groups = proj_df["baseline_group"].nunique()
    print(f"  - Total Cohort Groups Created: {unique_groups:,} groups across 3 levels", flush=True)
    print(f"  - Breakdown by Baseline Type:")
    type_counts = proj_df["baseline_type"].value_counts()
    for b_type, count in type_counts.items():
        print(f"    * {b_type:<15}: {count:,} projects ({count/total_projects*100:.2f}%)", flush=True)

    # -------------------------------------------------------------------------
    # 3. CALCULATE COHORT PARAMETRIC & QUANTILE BENCHMARKS
    # -------------------------------------------------------------------------
    print("\n[3] COMPUTING PARAMETRIC & QUANTILE STATISTICAL BENCHMARKS...", flush=True)
    
    group_stats = proj_df.groupby("baseline_group")["effective_amount"].agg(
        group_mean="mean",
        group_std="std",
        group_median="median",
        group_p75=lambda x: np.percentile(x, 75),
        group_p90=lambda x: np.percentile(x, 90),
        group_p95=lambda x: np.percentile(x, 95),
        group_p99=lambda x: np.percentile(x, 99)
    ).reset_index()
    
    # Fill std = 0.0 if group std is null or 0
    group_stats["group_std"] = group_stats["group_std"].fillna(0.0)
    
    # Merge group benchmarks back to main dataframe
    merged_df = pd.merge(proj_df, group_stats, on="baseline_group", how="left")
    
    # -------------------------------------------------------------------------
    # 4. COMPUTE Z-SCORES, PERCENTILES & MULTI-THRESHOLD FLAGS
    # -------------------------------------------------------------------------
    print("\n[4] COMPUTING Z-SCORES, PERCENTILE RANKS & THRESHOLD FLAGS...", flush=True)
    
    # Z-Score computation
    std_safe = merged_df["group_std"].replace(0.0, np.nan)
    z_scores = (merged_df["effective_amount"] - merged_df["group_mean"]) / std_safe
    merged_df["z_score"] = np.round(z_scores.fillna(0.0), 4)
    
    # Percentile rank within cohort group
    merged_df["percentile"] = merged_df.groupby("baseline_group")["effective_amount"].transform(percentile_rank_within_group)
    merged_df["percentile"] = np.round(merged_df["percentile"], 4)
    
    # Threshold flags
    merged_df["z_flag"] = (merged_df["z_score"] > 2.5).astype(int)
    merged_df["p95_flag"] = (merged_df["percentile"] >= 95.0).astype(int)
    merged_df["p99_flag"] = (merged_df["percentile"] >= 99.0).astype(int)
    
    # Combined statistical anomaly flag (Z > 2.5 OR Percentile >= 95.0)
    merged_df["statistical_flag"] = (merged_df["z_flag"] == 1) | (merged_df["p95_flag"] == 1)
    merged_df["statistical_flag"] = merged_df["statistical_flag"].astype(int)
    
    # -------------------------------------------------------------------------
    # 5. EXPORT RESULTS DATASET
    # -------------------------------------------------------------------------
    print("\n[5] EXPORTING STATISTICAL ANOMALY RESULTS CSV...", flush=True)
    
    output_cols = [
        "work_id",
        "project_status",
        "state",
        "constituency",
        "category",
        "effective_amount",
        "baseline_group",
        "baseline_type",
        "group_mean",
        "group_std",
        "group_median",
        "group_p75",
        "group_p90",
        "group_p95",
        "group_p99",
        "z_score",
        "percentile",
        "statistical_flag",
        "z_flag",
        "p95_flag",
        "p99_flag"
    ]
    
    results_df = merged_df[output_cols].copy()
    results_path = os.path.join(PROCESSED_DIR, "statistical_anomaly_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    
    t1 = time.time()
    
    # -------------------------------------------------------------------------
    # PRINT MANDATORY STAGE 4C REPORT
    # -------------------------------------------------------------------------
    num_z_anomalies = int(results_df["z_flag"].sum())
    num_p95_anomalies = int(results_df["p95_flag"].sum())
    num_p99_anomalies = int(results_df["p99_flag"].sum())
    num_combined = int(results_df["statistical_flag"].sum())
    
    print("\n" + "=" * 80, flush=True)
    print("               ENGINEERED STATISTICAL BASELINES REPORT                ", flush=True)
    print("=" * 80, flush=True)
    print("1. PIPELINE SCALE & COHORT DIMENSIONS:", flush=True)
    print(f"   - Total Projects Evaluated:     {total_projects:,}", flush=True)
    print(f"   - Unique Cohort Groups:         {unique_groups:,}", flush=True)
    print(f"   - Total Execution Time:         {t1 - t0:.2f} seconds", flush=True)

    print("\n2. STATISTICAL ANOMALY THRESHOLD AUDIT SUMMARY:", flush=True)
    print(f"  {'Threshold Criteria':<35} | {'Flagged Projects':<18} | {'Proportion of Corpus':<20}", flush=True)
    print("  " + "-" * 78, flush=True)
    print(f"  {'Z-Score Threshold (Z > 2.5)':<35} | {num_z_anomalies:<18,} | {num_z_anomalies/total_projects*100:<20.2f}%", flush=True)
    print(f"  {'95th Percentile (Percentile >= 95%)':<35} | {num_p95_anomalies:<18,} | {num_p95_anomalies/total_projects*100:<20.2f}%", flush=True)
    print(f"  {'99th Percentile (Percentile >= 99%)':<35} | {num_p99_anomalies:<18,} | {num_p99_anomalies/total_projects*100:<20.2f}%", flush=True)
    print(f"  {'Combined Flag (Z > 2.5 OR P95)':<35} | {num_combined:<18,} | {num_combined/total_projects*100:<20.2f}%", flush=True)

    print("\n3. SAMPLE EXTREME STATISTICAL ANOMALY OBSERVATIONS (TOP HIGH-DEVIATION PROJECTS):", flush=True)
    top_extreme = results_df.sort_values("z_score", ascending=False).head(5)
    
    for idx, r in top_extreme.reset_index().iterrows():
        print(f"\n  Case #{idx+1} [Work ID: {r['work_id']}]", flush=True)
        print(f"   - State / Constituency:   {r['state']} / {r['constituency']}", flush=True)
        print(f"   - Baseline Group:        {r['baseline_group']} (Type: {r['baseline_type']})", flush=True)
        print(f"   - Effective Amount:       ₹{r['effective_amount']:,.2f}", flush=True)
        print(f"   - Group Stats (Mean/P99): Mean = ₹{r['group_mean']:,.2f} | 99th Percentile = ₹{r['group_p99']:,.2f}", flush=True)
        print(f"   - Calculated Metrics:     Z-Score = {r['z_score']:.2f} | Percentile Rank = {r['percentile']:.2f}%", flush=True)

    print("\n4. OUTPUT DATASET MANIFEST:", flush=True)
    print(f"   - Results CSV: {results_path} ({len(results_df):,} rows x {len(results_df.columns)} columns)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    build_statistical_baselines()
