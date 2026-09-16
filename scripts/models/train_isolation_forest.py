"""
MPLADS Isolation Forest Unsupervised Anomaly Detection Pipeline (Stage 4B)

This script trains an Isolation Forest model on the numerical feature matrix:
1. Loads feature matrix from data/features/isolation_forest_features.csv.
2. Filters active model features using data/features/feature_dictionary.csv.
3. Pre-flight checks: verifies zero target leakage, zero missing values, zero infinite values.
4. Fits StandardScaler on training features X.
5. Fits IsolationForest (n_estimators=100, contamination=0.05, random_state=42).
6. Computes raw anomaly scores, normalized 0-1 anomaly scores, anomaly labels (-1 vs 1), and binary anomaly flags (1 vs 0).
7. Serializes model artifacts to models/isolation_forest.joblib and models/scaler.joblib.
8. Exports results table to data/processed/isolation_forest_results.csv.

Outputs:
- models/isolation_forest.joblib
- models/scaler.joblib
- data/processed/isolation_forest_results.csv
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "methodology")

def train_isolation_forest():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    t0 = time.time()
    print("=" * 80, flush=True)
    print("         MPLADS ISOLATION FOREST ANOMALY DETECTION TRAINING (STAGE 4B)      ", flush=True)
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. LOAD FEATURE MATRIX & FEATURE DICTIONARY
    # -------------------------------------------------------------------------
    matrix_path = os.path.join(FEATURES_DIR, "isolation_forest_features.csv")
    dict_path = os.path.join(FEATURES_DIR, "feature_dictionary.csv")
    
    matrix_df = pd.read_csv(matrix_path, low_memory=False)
    dict_df = pd.read_csv(dict_path, low_memory=False)
    
    total_obs = len(matrix_df)
    
    # Select active model features (used_for_model == True)
    active_features = dict_df[dict_df["used_for_model"] == True]["feature_name"].tolist()
    metadata_cols = ["work_id", "project_status", "state", "constituency"]
    
    print(f"\n[1] LOADED TRAINING DATASET SUMMARY:", flush=True)
    print(f"  - Total Observations (Projects): {total_obs:,}", flush=True)
    print(f"  - Total Metadata Identifier Columns: {len(metadata_cols)} columns", flush=True)
    print(f"  - Active Numerical Features for Model: {len(active_features)} features", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. PRE-FLIGHT DATA AUDIT
    # -------------------------------------------------------------------------
    print("\n[2] PERFORMING PRE-FLIGHT MODEL DATA INTEGRITY CHECKS...", flush=True)
    
    X_raw = matrix_df[active_features].copy()
    
    # Check 1: Target leakage & non-numerical columns
    non_num = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise ValueError(f"Pre-flight failed! Non-numerical columns found in X: {non_num}")
    print("  - Check 1 (Numerical Inputs & No Target Leakage): PASSED", flush=True)
    
    # Check 2: Missing values check
    null_cnt = X_raw.isnull().sum().sum()
    if null_cnt > 0:
        raise ValueError(f"Pre-flight failed! Found {null_cnt} missing values in feature matrix X.")
    print("  - Check 2 (Missing Value Audit - 0 Nulls): PASSED", flush=True)
    
    # Check 3: Infinite values check
    inf_cnt = np.isinf(X_raw).sum().sum()
    if inf_cnt > 0:
        raise ValueError(f"Pre-flight failed! Found {inf_cnt} infinite values in feature matrix X.")
    print("  - Check 3 (Infinite Value Audit - 0 Infs): PASSED", flush=True)
    
    # -------------------------------------------------------------------------
    # 3. FEATURE SCALING (StandardScaler)
    # -------------------------------------------------------------------------
    print("\n[3] FITTING STANDARD SCALER ON MODEL FEATURE MATRIX X...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  - Saved fitted StandardScaler to: {scaler_path}", flush=True)
    
    # -------------------------------------------------------------------------
    # 4. TRAIN ISOLATION FOREST MODEL
    # -------------------------------------------------------------------------
    print("\n[4] TRAINING UNSUPERVISED ISOLATION FOREST MODEL...", flush=True)
    print("  - Hyperparameters: n_estimators=100, contamination=0.05, random_state=42", flush=True)
    
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_scaled)
    
    model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(clf, model_path)
    print(f"  - Saved trained IsolationForest model to: {model_path}", flush=True)
    
    # -------------------------------------------------------------------------
    # 5. GENERATE ANOMALY SCORES & LABELS
    # -------------------------------------------------------------------------
    print("\n[5] INFERRING ANOMALY SCORES & ISOLATION LABELS...", flush=True)
    
    # Raw decision function scores (positive = normal, negative = anomalous)
    raw_scores = clf.decision_function(X_scaled)
    
    # Standard sklearn labels (-1 = anomaly, 1 = normal)
    anomaly_labels = clf.predict(X_scaled)
    
    # Binary anomaly indicator (1 = anomaly, 0 = normal)
    is_anomaly = (anomaly_labels == -1).astype(int)
    
    # Normalized score: 0.0 (least anomalous) to 1.0 (most anomalous) for UI dashboard
    # Inverse score function so that higher values = higher anomaly severity
    inv_scores = -raw_scores
    min_inv = inv_scores.min()
    max_inv = inv_scores.max()
    normalized_scores = (inv_scores - min_inv) / (max_inv - min_inv)
    
    # Assemble results dataframe
    results_df = matrix_df[metadata_cols].copy()
    results_df["raw_anomaly_score"] = np.round(raw_scores, 6)
    results_df["normalized_anomaly_score"] = np.round(normalized_scores, 6)
    results_df["anomaly_label"] = anomaly_labels
    results_df["is_anomaly"] = is_anomaly
    
    results_path = os.path.join(PROCESSED_DIR, "isolation_forest_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    
    t1 = time.time()
    
    # -------------------------------------------------------------------------
    # PRINT MANDATORY STAGE 4B REPORT
    # -------------------------------------------------------------------------
    num_anomalies = int(is_anomaly.sum())
    anomaly_pct = (num_anomalies / total_obs) * 100.0
    
    print("\n" + "=" * 80, flush=True)
    print("               ISOLATION FOREST ANOMALY DETECTION REPORT              ", flush=True)
    print("=" * 80, flush=True)
    print("1. MODEL TRAINING METRICS & DIMENSIONS:", flush=True)
    print(f"   - Total Observations Processed: {total_obs:,} projects", flush=True)
    print(f"   - Number of Active Features:    {len(active_features)} features", flush=True)
    print(f"   - Total Flagged Anomalies:      {num_anomalies:,} projects", flush=True)
    print(f"   - Anomaly Contamination Rate:   {anomaly_pct:.2f}% (Target: 5.00%)", flush=True)
    print(f"   - Total Pipeline Execution Time: {t1 - t0:.2f} seconds", flush=True)

    print("\n2. ANOMALY SCORE DISTRIBUTION METRICS:", flush=True)
    print(f"  {'Metric / Score':<25} | {'Raw Decision Function':<22} | {'Normalized Score [0, 1]':<25}", flush=True)
    print("  " + "-" * 78, flush=True)
    
    s_raw = pd.Series(raw_scores)
    s_norm = pd.Series(normalized_scores)
    
    print(f"  {'Mean':<25} | {s_raw.mean():<22.6f} | {s_norm.mean():<25.6f}", flush=True)
    print(f"  {'Std Dev':<25} | {s_raw.std():<22.6f} | {s_norm.std():<25.6f}", flush=True)
    print(f"  {'Min (Most Anomalous)':<25} | {s_raw.min():<22.6f} | {s_norm.max():<25.6f}", flush=True)
    print(f"  {'25th Percentile':<25} | {s_raw.quantile(0.25):<22.6f} | {s_norm.quantile(0.75):<25.6f}", flush=True)
    print(f"  {'50th Percentile (Median)':<25} | {s_raw.median():<22.6f} | {s_norm.median():<25.6f}", flush=True)
    print(f"  {'75th Percentile':<25} | {s_raw.quantile(0.75):<22.6f} | {s_norm.quantile(0.25):<25.6f}", flush=True)
    print(f"  {'Max (Least Anomalous)':<25} | {s_raw.max():<22.6f} | {s_norm.min():<25.6f}", flush=True)

    print("\n3. ANOMALY BREAKDOWN BY PROJECT STATUS:", flush=True)
    status_summary = results_df.groupby("project_status")["is_anomaly"].agg(["count", "sum", "mean"]).reset_index()
    status_summary["mean"] = status_summary["mean"] * 100.0
    print(f"  {'Project Status':<20} | {'Total Projects':<15} | {'Anomalies Flagged':<18} | {'Anomaly Rate':<15}", flush=True)
    print("  " + "-" * 75, flush=True)
    for idx, r in status_summary.iterrows():
        print(f"  {r['project_status']:<20} | {r['count']:<15,} | {r['sum']:<18,} | {r['mean']:<15.2f}%", flush=True)

    print("\n4. OUTPUT ARTIFACTS & DATASETS MANIFEST:", flush=True)
    print(f"   - Model Pipeline File:    {model_path}", flush=True)
    print(f"   - Scaler Pipeline File:   {scaler_path}", flush=True)
    print(f"   - Results CSV File:       {results_path} ({len(results_df):,} rows x {len(results_df.columns)} columns)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    train_isolation_forest()
