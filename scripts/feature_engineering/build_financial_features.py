"""
MPLADS Financial Feature Engineering Script (Stage 3A)

This script engineers numerical financial features for anomaly detection:
1. expenditure_ratio = final_amount / recommended_amount
2. release_ratio = total_expenditure / allocated_amount (MP summary level)
3. recommendation_sanction_ratio = recommended_amount / state_mean_recommended
4. unspent_balance = recommended_amount - final_amount (projects) and unspent_amount (MP summary)
5. expenditure_deviation = Z-score of final_amount within category
6. cost_deviation = Z-score of effective_amount within state
7. cost_percentile = Percentile rank of effective_amount within Category (N >= 10)
8. state_cost_percentile = Percentile rank of effective_amount within State (N >= 10)
9. constituency_cost_percentile = Percentile rank of effective_amount within Constituency (N >= 10)

Enforces MIN_GROUP_SIZE = 10 thresholding to avoid misleading small-group statistics.

Outputs:
- data/features/financial_features.csv
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

MIN_GROUP_SIZE = 10  # Minimum observations required per group to compute valid group-relative statistics

def compute_group_percentile(df, cost_col, group_col, min_size=MIN_GROUP_SIZE):
    """Compute percentile rank within group, setting result to NaN if group size < min_size."""
    result = pd.Series(index=df.index, dtype="float64")
    valid_mask = df[cost_col].notnull() & df[group_col].notnull()
    
    group_sizes = df[valid_mask].groupby(group_col)[cost_col].transform("count")
    eligible_mask = valid_mask & (group_sizes >= min_size)
    
    pcts = df[eligible_mask].groupby(group_col)[cost_col].rank(pct=True) * 100.0
    result.loc[eligible_mask] = pcts
    return result

def compute_group_zscore(df, cost_col, group_col, min_size=MIN_GROUP_SIZE):
    """Compute Z-score within group, setting result to NaN if group size < min_size or std == 0."""
    result = pd.Series(index=df.index, dtype="float64")
    valid_mask = df[cost_col].notnull() & df[group_col].notnull()
    
    group_sizes = df[valid_mask].groupby(group_col)[cost_col].transform("count")
    eligible_mask = valid_mask & (group_sizes >= min_size)
    
    means = df[eligible_mask].groupby(group_col)[cost_col].transform("mean")
    stds = df[eligible_mask].groupby(group_col)[cost_col].transform("std")
    
    non_zero_std = eligible_mask & (stds > 1e-6)
    z_scores = (df.loc[non_zero_std, cost_col] - means.loc[non_zero_std]) / stds.loc[non_zero_std]
    result.loc[non_zero_std] = z_scores
    return result

def build_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    print("=" * 80)
    print("           MPLADS FINANCIAL FEATURE ENGINEERING PIPELINE (STAGE 3A)        ")
    print("=" * 80)
    
    projects_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
    mp_path = os.path.join(PROCESSED_DIR, "mp_summary_clean.csv")
    
    proj = pd.read_csv(projects_path, low_memory=False)
    mp = pd.read_csv(mp_path, low_memory=False)
    
    print(f"\n[1] LOADED INPUT DATASETS:")
    print(f"  - projects_clean.csv: {len(proj):,} records")
    print(f"  - mp_summary_clean.csv: {len(mp):,} records")
    
    # -------------------------------------------------------------------------
    # DYNAMIC COLUMN DETECTION
    # -------------------------------------------------------------------------
    avail_cols = proj.columns.tolist()
    print("\n[2] DYNAMIC FINANCIAL COLUMN DETECTION:")
    print(f"  - Detected project attributes: {avail_cols}")
    
    has_rec_amt = "recommended_amount" in avail_cols
    has_fin_amt = "final_amount" in avail_cols
    
    print(f"  - recommended_amount available: {has_rec_amt}")
    print(f"  - final_amount available: {has_fin_amt}")
    
    # Create unified financial scale metric
    proj["effective_amount"] = proj["final_amount"].fillna(proj["recommended_amount"])
    
    features_df = pd.DataFrame()
    features_df["work_id"] = proj["work_id"]
    features_df["project_status"] = proj["project_status"]
    features_df["state"] = proj["state"]
    features_df["constituency"] = proj["constituency"]
    features_df["category"] = proj["category"]
    features_df["effective_amount"] = proj["effective_amount"]
    features_df["recommended_amount"] = proj["recommended_amount"]
    features_df["final_amount"] = proj["final_amount"]

    # 1. expenditure_ratio = final_amount / recommended_amount (where both exist)
    features_df["expenditure_ratio"] = proj["final_amount"] / proj["recommended_amount"]
    
    # 2. unspent_balance = recommended_amount - final_amount (for linked completed projects)
    features_df["unspent_balance"] = proj["recommended_amount"] - proj["final_amount"]
    
    # 3. recommendation_sanction_ratio = recommended_amount / state_mean_recommended
    state_mean_rec = proj.groupby("state")["recommended_amount"].transform("mean")
    features_df["recommendation_sanction_ratio"] = proj["recommended_amount"] / state_mean_rec
    
    # 4. expenditure_deviation = Z-score of final_amount within category
    features_df["expenditure_deviation"] = compute_group_zscore(proj, "final_amount", "category", min_size=MIN_GROUP_SIZE)
    
    # 5. cost_deviation = Z-score of effective_amount within state
    features_df["cost_deviation"] = compute_group_zscore(proj, "effective_amount", "state", min_size=MIN_GROUP_SIZE)
    
    # 6. cost_percentile = Percentile rank of effective_amount within Category (N >= 10)
    features_df["cost_percentile"] = compute_group_percentile(proj, "effective_amount", "category", min_size=MIN_GROUP_SIZE)
    
    # 7. state_cost_percentile = Percentile rank of effective_amount within State (N >= 10)
    features_df["state_cost_percentile"] = compute_group_percentile(proj, "effective_amount", "state", min_size=MIN_GROUP_SIZE)
    
    # 8. constituency_cost_percentile = Percentile rank of effective_amount within Constituency (N >= 10)
    features_df["constituency_cost_percentile"] = compute_group_percentile(proj, "effective_amount", "constituency", min_size=MIN_GROUP_SIZE)
    
    # 9. MP Summary Level Features (release_ratio & mp_unspent_balance)
    mp_release_ratio = mp["total_expenditure"] / mp["allocated_amount"]
    mp_unspent_balance = mp["unspent_amount"]
    
    # Merge MP-level release_ratio back onto projects feature matrix for project-level modeling
    mp_lookup = mp[["mp_name", "constituency", "total_expenditure", "allocated_amount"]].copy()
    mp_lookup["mp_release_ratio"] = mp_lookup["total_expenditure"] / mp_lookup["allocated_amount"]
    
    proj_mp_merged = pd.merge(
        proj[["work_id", "mp_name", "constituency"]],
        mp_lookup[["mp_name", "constituency", "mp_release_ratio"]],
        on=["mp_name", "constituency"],
        how="left"
    )
    features_df["release_ratio"] = proj_mp_merged["mp_release_ratio"]

    # Save to CSV
    output_path = os.path.join(FEATURES_DIR, "financial_features.csv")
    features_df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n[3] FINANCIAL FEATURES GENERATED & SAVED:")
    print(f"  - Output file: {output_path}")
    print(f"  - Total records: {len(features_df):,} rows")
    print(f"  - Total columns: {len(features_df.columns)} columns")
    
    # -------------------------------------------------------------------------
    # PRINT FORMULAS, MISSING VALUES, AND DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------
    feature_meta = [
        {
            "name": "expenditure_ratio",
            "formula": "final_amount / recommended_amount",
            "description": "Ratio of completed cost to original recommended cost for linked projects"
        },
        {
            "name": "release_ratio",
            "formula": "total_expenditure / allocated_amount (MP Summary)",
            "description": "Fund release and utilization efficiency ratio at MP constituency level"
        },
        {
            "name": "recommendation_sanction_ratio",
            "formula": "recommended_amount / mean(recommended_amount, State)",
            "description": "Project recommended amount normalized by State average recommendation"
        },
        {
            "name": "unspent_balance",
            "formula": "recommended_amount - final_amount",
            "description": "Net difference between recommended allocation and actual final expenditure"
        },
        {
            "name": "expenditure_deviation",
            "formula": "(final_amount - mean(Category)) / std(Category) [N >= 10]",
            "description": "Z-score deviation of final expenditure within project category group"
        },
        {
            "name": "cost_deviation",
            "formula": "(effective_amount - mean(State)) / std(State) [N >= 10]",
            "description": "Z-score deviation of project financial cost within State group"
        },
        {
            "name": "cost_percentile",
            "formula": "Percentile Rank of effective_amount within Category [N >= 10]",
            "description": "Percentile score of project cost relative to category cohort"
        },
        {
            "name": "state_cost_percentile",
            "formula": "Percentile Rank of effective_amount within State [N >= 10]",
            "description": "Percentile score of project cost relative to State cohort"
        },
        {
            "name": "constituency_cost_percentile",
            "formula": "Percentile Rank of effective_amount within Constituency [N >= 10]",
            "description": "Percentile score of project cost relative to Constituency cohort"
        }
    ]

    print("\n" + "=" * 80)
    print("                 ENGINEERED FINANCIAL FEATURES REPORT                  ")
    print("=" * 80)
    print(f"Total Usable Records in Feature Matrix: {len(features_df):,}\n")
    
    for meta in feature_meta:
        col = meta["name"]
        s = features_df[col]
        null_cnt = int(s.isnull().sum())
        non_null_cnt = int(s.notnull().sum())
        null_pct = (null_cnt / len(features_df)) * 100.0
        
        print(f"FEATURE NAME: {col}")
        print(f"  Formula:     {meta['formula']}")
        print(f"  Description: {meta['description']}")
        print(f"  Usable / Non-Null Records: {non_null_cnt:,} ({100.0 - null_pct:.2f}%)")
        print(f"  Missing / Null Records:    {null_cnt:,} ({null_pct:.2f}%)")
        
        if non_null_cnt > 0:
            print("  Descriptive Statistics:")
            print(f"    - Mean:   {s.mean():.4f}")
            print(f"    - Std:    {s.std():.4f}")
            print(f"    - Min:    {s.min():.4f}")
            print(f"    - 25%:    {s.quantile(0.25):.4f}")
            print(f"    - 50%:    {s.quantile(0.50):.4f}")
            print(f"    - 75%:    {s.quantile(0.75):.4f}")
            print(f"    - Max:    {s.max():.4f}")
        print("-" * 80)

if __name__ == "__main__":
    build_features()
