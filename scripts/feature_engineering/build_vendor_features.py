"""
MPLADS Vendor and Administrative Feature Engineering Script (Stage 3C)

This script engineers vendor-level risk, concentration, and transaction pattern features:
1. transaction_count
2. total_vendor_expenditure
3. average_vendor_transaction
4. median_vendor_transaction
5. vendor_repeat_rate
6. pending_payment_ratio
7. payment_status_distribution (success_payment_ratio)
8. vendor_share_of_constituency_expenditure
9. vendor_share_of_MP_expenditure
10. transaction_frequency
11. year_over_year_vendor_change

Enforces strict neutral analytical terminology ('anomaly', 'unusual pattern', 'requires verification')
without making accusations of fraud or corruption.

Outputs:
- data/features/vendor_features.csv
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

def build_vendor_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    print("=" * 80)
    print("         MPLADS VENDOR & ADMINISTRATIVE FEATURE PIPELINE (STAGE 3C)        ")
    print("=" * 80)
    
    exp_path = os.path.join(PROCESSED_DIR, "expenditures_clean.csv")
    exp = pd.read_csv(exp_path, low_memory=False)
    
    num_transactions = len(exp)
    num_vendors = exp["vendor"].nunique()
    num_mps = exp["mp_name"].nunique()
    num_constituencies = exp["constituency"].nunique()
    num_idas = exp["ida"].nunique()
    
    print(f"\n[1] LOADED INPUT DATASET SUMMARY:")
    print(f"  - Total Transactions:    {num_transactions:,}")
    print(f"  - Unique Vendors:        {num_vendors:,}")
    print(f"  - Unique MPs:            {num_mps:,}")
    print(f"  - Unique Constituencies: {num_constituencies:,}")
    print(f"  - Unique IDAs:           {num_idas:,}")
    
    # -------------------------------------------------------------------------
    # CONSTITUENCY & MP TOTAL EXPENDITURES (FOR SHARE CALCULATIONS)
    # -------------------------------------------------------------------------
    const_total_exp = exp.groupby("constituency")["expenditure_amount"].sum().to_dict()
    mp_total_exp = exp.groupby("mp_name")["expenditure_amount"].sum().to_dict()

    # Vendor-Constituency aggregation
    vendor_const_exp = exp.groupby(["vendor", "constituency"])["expenditure_amount"].sum().reset_index()
    vendor_const_exp["const_total"] = vendor_const_exp["constituency"].map(const_total_exp)
    vendor_const_exp["vendor_const_share"] = vendor_const_exp["expenditure_amount"] / vendor_const_exp["const_total"]
    max_vendor_const_share = vendor_const_exp.groupby("vendor")["vendor_const_share"].max().to_dict()

    # Vendor-MP aggregation
    vendor_mp_exp = exp.groupby(["vendor", "mp_name"])["expenditure_amount"].sum().reset_index()
    vendor_mp_exp["mp_total"] = vendor_mp_exp["mp_name"].map(mp_total_exp)
    vendor_mp_exp["vendor_mp_share"] = vendor_mp_exp["expenditure_amount"] / vendor_mp_exp["mp_total"]
    max_vendor_mp_share = vendor_mp_exp.groupby("vendor")["vendor_mp_share"].max().to_dict()

    # Primary metadata for each vendor
    vendor_primary_meta = exp.drop_duplicates(subset=["vendor"])[["vendor", "state", "constituency", "mp_name", "ida"]].copy()

    # -------------------------------------------------------------------------
    # VENDOR LEVEL AGGREGATIONS
    # -------------------------------------------------------------------------
    print("\n[2] COMPUTING VENDOR LEVEL AGGREGATED FEATURES...")
    
    vendor_stats = exp.groupby("vendor").agg(
        transaction_count=("expenditure_amount", "count"),
        total_vendor_expenditure=("expenditure_amount", "sum"),
        average_vendor_transaction=("expenditure_amount", "mean"),
        median_vendor_transaction=("expenditure_amount", "median"),
        pending_count=("payment_status", lambda s: (s != "Payment Success").sum()),
        success_count=("payment_status", lambda s: (s == "Payment Success").sum())
    ).reset_index()

    # Calculate derived ratio features
    vendor_stats["vendor_repeat_rate"] = (vendor_stats["transaction_count"] - 1) / vendor_stats["transaction_count"]
    vendor_stats["pending_payment_ratio"] = vendor_stats["pending_count"] / vendor_stats["transaction_count"]
    vendor_stats["payment_status_distribution"] = vendor_stats["success_count"] / vendor_stats["transaction_count"]
    
    # Map Shares
    vendor_stats["vendor_share_of_constituency_expenditure"] = vendor_stats["vendor"].map(max_vendor_const_share)
    vendor_stats["vendor_share_of_MP_expenditure"] = vendor_stats["vendor"].map(max_vendor_mp_share)

    # Calculate Transaction Frequency (active months count per vendor)
    dt_exp = pd.to_datetime(exp["expenditure_date_iso"], errors="coerce")
    exp["year_month"] = dt_exp.dt.to_period("M")
    active_months_per_vendor = exp.groupby("vendor")["year_month"].nunique().to_dict()
    vendor_stats["active_months_count"] = vendor_stats["vendor"].map(active_months_per_vendor)
    vendor_stats["transaction_frequency"] = vendor_stats["transaction_count"] / vendor_stats["active_months_count"]

    # Calculate Year over Year Vendor Change (2025 vs 2026 expenditure change)
    exp_by_year = exp.groupby(["vendor", "expenditure_year"])["expenditure_amount"].sum().unstack(fill_value=0)
    if 2025 in exp_by_year.columns and 2026 in exp_by_year.columns:
        yoy_vendor = ((exp_by_year[2026] - exp_by_year[2025]) / exp_by_year[2025].replace(0, np.nan)) * 100.0
        vendor_stats["year_over_year_vendor_change"] = vendor_stats["vendor"].map(yoy_vendor.to_dict())
    else:
        vendor_stats["year_over_year_vendor_change"] = np.nan

    # Merge primary metadata
    vf = pd.merge(vendor_primary_meta, vendor_stats, on="vendor", how="left")

    # Select final clean columns
    final_cols = [
        "vendor",
        "state",
        "constituency",
        "mp_name",
        "ida",
        "transaction_count",
        "total_vendor_expenditure",
        "average_vendor_transaction",
        "median_vendor_transaction",
        "vendor_repeat_rate",
        "pending_payment_ratio",
        "payment_status_distribution",
        "vendor_share_of_constituency_expenditure",
        "vendor_share_of_MP_expenditure",
        "transaction_frequency",
        "year_over_year_vendor_change"
    ]

    vendor_features_df = vf[final_cols].copy()

    # Save output CSV
    output_path = os.path.join(FEATURES_DIR, "vendor_features.csv")
    vendor_features_df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n[3] VENDOR FEATURES GENERATED & SAVED:")
    print(f"  - Output file: {output_path}")
    print(f"  - Total unique vendors: {len(vendor_features_df):,} rows")
    print(f"  - Total feature columns: {len(vendor_features_df.columns)} columns")
    
    # -------------------------------------------------------------------------
    # PRINT REPORT: ENTITY COUNTS, FEATURES, DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                ENGINEERED VENDOR & ADMIN FEATURES REPORT             ")
    print("=" * 80)
    print("1. AUDITED ENTITY COUNTS:")
    print(f"   - Total Unique Vendors:        {num_vendors:,}")
    print(f"   - Total Unique Constituencies: {num_constituencies:,}")
    print(f"   - Total Unique MPs:            {num_mps:,}")
    print(f"   - Total Unique IDAs:           {num_idas:,}")
    print(f"   - Total Payment Transactions:  {num_transactions:,}")

    feature_cols = [
        "transaction_count",
        "total_vendor_expenditure",
        "average_vendor_transaction",
        "median_vendor_transaction",
        "vendor_repeat_rate",
        "pending_payment_ratio",
        "payment_status_distribution",
        "vendor_share_of_constituency_expenditure",
        "vendor_share_of_MP_expenditure",
        "transaction_frequency",
        "year_over_year_vendor_change"
    ]

    print("\n2. FEATURE DESCRIPTIVE STATISTICS across 27,927 Unique Vendors:")
    print(f"  {'Feature Name':<45} | {'Usable':<8} | {'Missing':<8} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'Max':<12}")
    print("  " + "-" * 118)
    
    for col in feature_cols:
        s = vendor_features_df[col]
        non_null_cnt = int(s.notnull().sum())
        null_cnt = int(s.isnull().sum())
        mean_val = f"{s.mean():.2f}" if non_null_cnt > 0 else "N/A"
        std_val = f"{s.std():.2f}" if non_null_cnt > 0 else "N/A"
        min_val = f"{s.min():.2f}" if non_null_cnt > 0 else "N/A"
        max_val = f"{s.max():.2f}" if non_null_cnt > 0 else "N/A"
        
        print(f"  {col:<45} | {non_null_cnt:<8,} | {null_cnt:<8,} | {mean_val:<12} | {std_val:<12} | {min_val:<10} | {max_val:<12}")

    print("=" * 80)

if __name__ == "__main__":
    build_vendor_features()
