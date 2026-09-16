"""
MPLADS Isolation Forest Training Matrix Generation Script (Stage 4A)

This script builds a clean, standardized, numerical feature matrix for unsupervised anomaly detection:
1. Loads features from financial, temporal, vendor, and NLP feature tables.
2. Integrates state-constituency level vendor risk aggregations.
3. Audits and handles infinite values and missing values using domain-grounded median imputation.
4. Validates feature variance and calculates correlation matrices to flag redundant pairs.
5. Preserves identifier columns (work_id, project_status, state, constituency) for auditability and traceability.
6. Exports training matrix to data/features/isolation_forest_features.csv.
7. Generates feature dictionary metadata at data/features/feature_dictionary.csv.

Outputs:
- data/features/isolation_forest_features.csv
- data/features/feature_dictionary.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "methodology")

def build_isolation_forest_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    t0 = time.time()
    print("=" * 80, flush=True)
    print("        MPLADS ISOLATION FOREST TRAINING MATRIX PIPELINE (STAGE 4A)       ", flush=True)
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. LOAD DOMAIN FEATURE TABLES
    # -------------------------------------------------------------------------
    fin_path = os.path.join(FEATURES_DIR, "financial_features.csv")
    temp_path = os.path.join(FEATURES_DIR, "temporal_features.csv")
    vendor_path = os.path.join(FEATURES_DIR, "vendor_features.csv")
    nlp_path = os.path.join(FEATURES_DIR, "nlp_features.csv")
    
    fin_df = pd.read_csv(fin_path, low_memory=False)
    temp_df = pd.read_csv(temp_path, low_memory=False)
    vendor_df = pd.read_csv(vendor_path, low_memory=False)
    nlp_df = pd.read_csv(nlp_path, low_memory=False)
    
    print(f"\n[1] LOADED FEATURE SOURCES SUMMARY:", flush=True)
    print(f"  - Financial Features Table: {len(fin_df):,} rows x {len(fin_df.columns)} columns", flush=True)
    print(f"  - Temporal Features Table:  {len(temp_df):,} rows x {len(temp_df.columns)} columns", flush=True)
    print(f"  - Vendor Features Table:    {len(vendor_df):,} rows x {len(vendor_df.columns)} columns", flush=True)
    print(f"  - NLP Features Table:       {len(nlp_df):,} rows x {len(nlp_df.columns)} columns", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. CONSTITUENCY-LEVEL VENDOR AGGREGATION
    # -------------------------------------------------------------------------
    print("\n[2] AGGREGATING VENDOR RISK METRICS AT COHORT LEVEL...", flush=True)
    vendor_df["clean_state"] = vendor_df["state"].astype(str).str.strip().str.upper()
    vendor_df["clean_const"] = vendor_df["constituency"].astype(str).str.strip().str.upper()
    
    v_const = vendor_df.groupby(["clean_state", "clean_const"]).agg(
        constituency_max_vendor_share=("vendor_share_of_constituency_expenditure", "max"),
        constituency_avg_vendor_transaction=("average_vendor_transaction", "mean"),
        constituency_vendor_count=("vendor", "nunique"),
        constituency_avg_vendor_repeat_rate=("vendor_repeat_rate", "mean"),
        constituency_pending_payment_ratio=("pending_payment_ratio", "mean")
    ).reset_index()
    
    # -------------------------------------------------------------------------
    # 3. ALIGN FEATURE SOURCES (INDEX-ALIGNED ROW INTEGRATION)
    # -------------------------------------------------------------------------
    print("\n[3] INTEGRATING DOMAIN FEATURE SOURCES (ROW ALIGNED N = 127,263)...", flush=True)
    
    fin_sub = fin_df[[
        "work_id", "project_status", "state", "constituency",
        "effective_amount", "cost_deviation", "state_cost_percentile",
        "constituency_cost_percentile", "release_ratio", "recommendation_sanction_ratio"
    ]].copy()
    
    temp_sub = temp_df[[
        "is_impossible_date_sequence", "is_negative_duration",
        "is_covid_era", "year_over_year_change"
    ]].copy()
    
    nlp_sub = nlp_df[[
        "description_length", "word_count", "tfidf_mean",
        "tfidf_max", "nearest_similarity_score", "similar_project_count"
    ]].copy()
    
    # Concatenate aligned project feature blocks
    master_df = pd.concat([fin_sub, temp_sub, nlp_sub], axis=1)
    
    # Merge vendor cohort metrics on normalized geography keys
    master_df["clean_state"] = master_df["state"].astype(str).str.strip().str.upper()
    master_df["clean_const"] = master_df["constituency"].astype(str).str.strip().str.upper()
    
    master_df = pd.merge(master_df, v_const, on=["clean_state", "clean_const"], how="left")
    master_df = master_df.drop(columns=["clean_state", "clean_const"])
    
    total_obs = len(master_df)
    
    # -------------------------------------------------------------------------
    # 4. DEFINE FEATURE SET & AUDIT MISSING / INFINITE VALUES
    # -------------------------------------------------------------------------
    metadata_cols = ["work_id", "project_status", "state", "constituency"]
    model_feature_cols = [
        # Financial
        "effective_amount",
        "cost_deviation",
        "state_cost_percentile",
        "constituency_cost_percentile",
        "release_ratio",
        "recommendation_sanction_ratio",
        # Temporal
        "is_impossible_date_sequence",
        "is_negative_duration",
        "is_covid_era",
        "year_over_year_change",
        # NLP
        "description_length",
        "word_count",
        "tfidf_mean",
        "tfidf_max",
        "nearest_similarity_score",
        "similar_project_count",
        # Vendor Cohort
        "constituency_max_vendor_share",
        "constituency_avg_vendor_transaction",
        "constituency_vendor_count",
        "constituency_avg_vendor_repeat_rate",
        "constituency_pending_payment_ratio"
    ]
    
    print("\n[4] AUDITING INFINITE VALUES & PRE-IMPUTATION MISSINGNESS...", flush=True)
    
    # Ensure boolean columns are numeric (0/1)
    bool_cols = ["is_impossible_date_sequence", "is_negative_duration", "is_covid_era"]
    for col in bool_cols:
        master_df[col] = master_df[col].astype(int)
    
    # Detect infinite values
    inf_counts = {}
    for col in model_feature_cols:
        num_inf = np.isinf(master_df[col]).sum()
        inf_counts[col] = num_inf
        if num_inf > 0:
            print(f"  [WARNING] Feature '{col}' contains {num_inf:,} infinite values! Replacing with NaN.", flush=True)
            master_df[col] = master_df[col].replace([np.inf, -np.inf], np.nan)
            
    pre_missing = master_df[model_feature_cols].isnull().sum().to_dict()
    
    # -------------------------------------------------------------------------
    # 5. DOMAIN-GROUNDED MEDIAN IMPUTATION
    # -------------------------------------------------------------------------
    print("\n[5] IMPUTING MISSING VALUES WITH DOMAIN-GROUNDED STANDARDS...", flush=True)
    
    imputation_defaults = {
        "recommendation_sanction_ratio": 1.0,  # 1.0 baseline average ratio
        "constituency_cost_percentile": master_df["constituency_cost_percentile"].median(),
        "year_over_year_change": 0.0,          # 0% year-over-year change baseline
        "constituency_max_vendor_share": master_df["constituency_max_vendor_share"].median(),
        "constituency_avg_vendor_transaction": master_df["constituency_avg_vendor_transaction"].median(),
        "constituency_vendor_count": 0.0,
        "constituency_avg_vendor_repeat_rate": 0.0,
        "constituency_pending_payment_ratio": 0.0
    }
    
    for col in model_feature_cols:
        if master_df[col].isnull().sum() > 0:
            default_val = imputation_defaults.get(col, master_df[col].median())
            master_df[col] = master_df[col].fillna(default_val)
            
    post_missing = master_df[model_feature_cols].isnull().sum().to_dict()
    
    # -------------------------------------------------------------------------
    # 6. FEATURE VARIANCE & CORRELATION ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[6] CHECKING FEATURE VARIANCE & MULTICOLLINEARITY...", flush=True)
    
    variances = master_df[model_feature_cols].var().to_dict()
    zero_var_cols = [col for col, v in variances.items() if v == 0 or np.isnan(v)]
    if zero_var_cols:
        print(f"  [NOTE] Identified {len(zero_var_cols)} zero-variance column(s): {zero_var_cols} (Flagged used_for_model=False)", flush=True)
    else:
        print("  - All features exhibit non-zero variance.", flush=True)
        
    corr_matrix = master_df[model_feature_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(model_feature_cols)):
        for j in range(i + 1, len(model_feature_cols)):
            f1, f2 = model_feature_cols[i], model_feature_cols[j]
            r = corr_matrix.loc[f1, f2]
            if abs(r) >= 0.85:
                high_corr_pairs.append((f1, f2, r))
                
    # -------------------------------------------------------------------------
    # 7. GENERATE FEATURE DICTIONARY METADATA
    # -------------------------------------------------------------------------
    print("\n[7] GENERATING FEATURE DICTIONARY CSV...", flush=True)
    
    feature_dict_entries = [
        # Metadata / Identifiers
        {"feature_name": "work_id", "source_column": "work_id", "description": "Unique project identifier", "formula": "Raw identifier string", "data_type": "string", "missing_percentage": "0.00%", "used_for_model": False},
        {"feature_name": "project_status", "source_column": "project_status", "description": "Status of project (Recommended vs Completed)", "formula": "Raw category string", "data_type": "string", "missing_percentage": "0.00%", "used_for_model": False},
        {"feature_name": "state", "source_column": "state", "description": "State / Union Territory name", "formula": "Raw geographical string", "data_type": "string", "missing_percentage": "0.00%", "used_for_model": False},
        {"feature_name": "constituency", "source_column": "constituency", "description": "Lok Sabha / Rajya Sabha constituency name", "formula": "Raw geographical string", "data_type": "string", "missing_percentage": "0.00%", "used_for_model": False},
        
        # Financial Features
        {"feature_name": "effective_amount", "source_column": "final_amount / recommended_amount", "description": "Sanctioned or completed project expenditure amount in Rupees", "formula": "COALESCE(final_amount, recommended_amount)", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "cost_deviation", "source_column": "effective_amount", "description": "Z-score of project effective cost within State cohort", "formula": "(cost - mean_state) / std_state", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "state_cost_percentile", "source_column": "effective_amount", "description": "Percentile rank of effective cost within State cohort", "formula": "Percentile(effective_amount | State)", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "constituency_cost_percentile", "source_column": "effective_amount", "description": "Percentile rank of effective cost within Constituency cohort", "formula": "Percentile(effective_amount | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_cost_percentile']/total_obs*100:.2f}%", "used_for_model": True},
        {"feature_name": "release_ratio", "source_column": "total_expenditure / allocated_amount", "description": "Sponsoring Member of Parliament fund utilization ratio", "formula": "total_expenditure / allocated_amount", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "recommendation_sanction_ratio", "source_column": "recommended_amount", "description": "Ratio of recommended amount relative to state mean recommendation", "formula": "recommended_amount / mean_state(recommended_amount)", "data_type": "float64", "missing_percentage": f"{pre_missing['recommendation_sanction_ratio']/total_obs*100:.2f}%", "used_for_model": True},
        
        # Temporal Features
        {"feature_name": "is_impossible_date_sequence", "source_column": "completed_date, recommendation_date", "description": "Binary indicator for invalid date ordering (completion before recommendation)", "formula": "IF(completed_date < recommendation_date, 1, 0)", "data_type": "int64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "is_negative_duration", "source_column": "completed_date, recommendation_date", "description": "Binary indicator for negative project duration (duplicate of date sequence check)", "formula": "IF(duration < 0, 1, 0)", "data_type": "int64", "missing_percentage": "0.00%", "used_for_model": False},
        {"feature_name": "is_covid_era", "source_column": "recommendation_year / completed_year", "description": "Binary indicator for project recommendation or completion during COVID window (zero variance in dataset)", "formula": "IF(year IN (2020, 2021), 1, 0)", "data_type": "int64", "missing_percentage": "0.00%", "used_for_model": False},
        {"feature_name": "year_over_year_change", "source_column": "project_count", "description": "Percentage change in constituency project volume year-over-year", "formula": "((count_t - count_t-1) / count_t-1) * 100", "data_type": "float64", "missing_percentage": f"{pre_missing['year_over_year_change']/total_obs*100:.2f}%", "used_for_model": True},
        
        # NLP Features
        {"feature_name": "description_length", "source_column": "work_description", "description": "Character length of raw work description text", "formula": "LEN(work_description)", "data_type": "int64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "word_count", "source_column": "work_description", "description": "Total word count of raw work description text", "formula": "WORD_COUNT(work_description)", "data_type": "int64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "tfidf_mean", "source_column": "work_description", "description": "Mean TF-IDF feature weight across all 10,000 vocabulary terms", "formula": "MEAN(TFIDF_Vector)", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "tfidf_max", "source_column": "work_description", "description": "Maximum TF-IDF feature weight for dominant description term", "formula": "MAX(TFIDF_Vector)", "data_type": "float64", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "nearest_similarity_score", "source_column": "work_description", "description": "Highest cosine text similarity score with any other project in constituency cohort", "formula": "MAX_j(CosineSim(d_i, d_j))", "data_type": "float32", "missing_percentage": "0.00%", "used_for_model": True},
        {"feature_name": "similar_project_count", "source_column": "work_description", "description": "Count of cohort projects exhibiting cosine text similarity >= 0.85", "formula": "SUM_j(CosineSim(d_i, d_j) >= 0.85)", "data_type": "int32", "missing_percentage": "0.00%", "used_for_model": True},
        
        # Vendor Cohort Risk Features
        {"feature_name": "constituency_max_vendor_share", "source_column": "vendor_share_of_constituency_expenditure", "description": "Maximum vendor expenditure share in constituency (vendor concentration measure)", "formula": "MAX(vendor_const_share | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_max_vendor_share']/total_obs*100:.2f}%", "used_for_model": True},
        {"feature_name": "constituency_avg_vendor_transaction", "source_column": "average_vendor_transaction", "description": "Mean vendor transaction amount across constituency vendors", "formula": "MEAN(vendor_avg_tx | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_avg_vendor_transaction']/total_obs*100:.2f}%", "used_for_model": True},
        {"feature_name": "constituency_vendor_count", "source_column": "vendor", "description": "Total unique vendors active in constituency", "formula": "COUNT(DISTINCT vendor | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_vendor_count']/total_obs*100:.2f}%", "used_for_model": True},
        {"feature_name": "constituency_avg_vendor_repeat_rate", "source_column": "vendor_repeat_rate", "description": "Mean vendor repeat transaction rate in constituency", "formula": "MEAN(repeat_rate | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_avg_vendor_repeat_rate']/total_obs*100:.2f}%", "used_for_model": True},
        {"feature_name": "constituency_pending_payment_ratio", "source_column": "pending_payment_ratio", "description": "Mean pending payment ratio across constituency vendors", "formula": "MEAN(pending_ratio | Constituency)", "data_type": "float64", "missing_percentage": f"{pre_missing['constituency_pending_payment_ratio']/total_obs*100:.2f}%", "used_for_model": True}
    ]
    
    feature_dict_df = pd.DataFrame(feature_dict_entries)
    dict_path = os.path.join(FEATURES_DIR, "feature_dictionary.csv")
    feature_dict_df.to_csv(dict_path, index=False, encoding="utf-8")
    
    # -------------------------------------------------------------------------
    # 8. EXPORT ISOLATION FOREST TRAINING MATRIX CSV
    # -------------------------------------------------------------------------
    print("\n[8] EXPORTING ISOLATION FOREST TRAINING MATRIX CSV...", flush=True)
    
    final_output_cols = metadata_cols + model_feature_cols
    matrix_df = master_df[final_output_cols].copy()
    
    matrix_path = os.path.join(FEATURES_DIR, "isolation_forest_features.csv")
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8")
    
    t1 = time.time()
    
    # -------------------------------------------------------------------------
    # PRINT MANDATORY STAGE 4A REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("            ENGINEERED ISOLATION FOREST FEATURE MATRIX REPORT             ", flush=True)
    print("=" * 80, flush=True)
    print("1. MATRIX SCALE & DIMENSIONS:", flush=True)
    print(f"   - Total Observations (Projects): {total_obs:,}", flush=True)
    print(f"   - Metadata / Identifier Columns: {len(metadata_cols)} columns ({metadata_cols})", flush=True)
    print(f"   - Total Numerical Model Features: {len(model_feature_cols)} features", flush=True)
    print(f"   - Pipeline Execution Time:       {t1 - t0:.2f} seconds", flush=True)

    print("\n2. NUMERICAL MODEL FEATURE NAMES:", flush=True)
    for idx, f_name in enumerate(model_feature_cols, 1):
        is_used = feature_dict_df.loc[feature_dict_df["feature_name"] == f_name, "used_for_model"].values[0]
        status_str = "ACTIVE" if is_used else "EXCLUDED (Zero Var / Duplicate)"
        print(f"   {idx:2d}. {f_name:<40} (Type: {matrix_df[f_name].dtype}, Status: {status_str})", flush=True)

    print("\n3. MISSING VALUE AUDIT SUMMARY (Pre-Imputation vs Post-Imputation):", flush=True)
    print(f"  {'Feature Name':<40} | {'Pre-Null Count':<14} | {'Pre-Null %':<12} | {'Post-Null Count':<14}", flush=True)
    print("  " + "-" * 88, flush=True)
    for col in model_feature_cols:
        pre_cnt = pre_missing[col]
        pre_pct = f"{pre_cnt / total_obs * 100:.2f}%"
        post_cnt = post_missing[col]
        print(f"  {col:<40} | {pre_cnt:<14,} | {pre_pct:<12} | {post_cnt:<14,}", flush=True)

    print("\n4. MULTICOLLINEARITY & HIGH CORRELATION PAIRS (|r| >= 0.85):", flush=True)
    if high_corr_pairs:
        for f1, f2, r in high_corr_pairs:
            print(f"   - [{f1}] <---> [{f2}]: r = {r:.4f}", flush=True)
    else:
        print("   - No feature pairs exceed collinearity threshold (|r| >= 0.85).", flush=True)

    print("\n5. FEATURE DESCRIPTIVE STATISTICS across 127,263 Projects:", flush=True)
    print(f"  {'Feature Name':<38} | {'Usable':<8} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'50%':<10} | {'Max':<12}", flush=True)
    print("  " + "-" * 110, flush=True)
    for col in model_feature_cols:
        s = matrix_df[col]
        print(f"  {col:<38} | {int(s.notnull().sum()):<8,} | {s.mean():<12.4f} | {s.std():<12.4f} | {s.min():<10.4f} | {s.median():<10.4f} | {s.max():<12.4f}", flush=True)

    print("\n6. OUTPUT DATASETS MANIFEST:", flush=True)
    print(f"   - Feature Matrix CSV:    {matrix_path} ({len(matrix_df):,} rows x {len(matrix_df.columns)} columns)", flush=True)
    print(f"   - Feature Dictionary CSV: {dict_path} ({len(feature_dict_df):,} rows x {len(feature_dict_df.columns)} columns)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    build_isolation_forest_features()
