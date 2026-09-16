"""
MPLADS Data Ingestion and Inventory Script (Stage 1)

This script performs reproducible ingestion of the four official MPLADS datasets:
1. recommended_works.csv
2. completed_works.csv
3. expenditures.csv
4. mp_summary.csv

Features:
- Preserves raw files exactly into data/raw/ without modification.
- Generates SHA256 checksums and records file size, row count, column count, schema, missing values.
- Logs every ingestion action to data/raw/ingestion.log and data/raw/ingestion_manifest.json.
- Prints exact inventory summary to stdout.
"""

import os
import shutil
import hashlib
import json
import datetime
import sys
import pandas as pd

# Reconfigure stdout to utf-8 to prevent console encoding issues with special characters (e.g. Rupee symbol ₹)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_FILES = [
    "recommended_works.csv",
    "completed_works.csv",
    "expenditures.csv",
    "mp_summary.csv"
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LOG_FILE_PATH = os.path.join(RAW_DATA_DIR, "ingestion.log")
MANIFEST_FILE_PATH = os.path.join(RAW_DATA_DIR, "ingestion_manifest.json")

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def log_message(msg):
    """Write log entry with timestamp to stdout and log file."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_entry = f"[{timestamp}] {msg}"
    print(log_entry)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def ingest_and_inventory():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    manifest = {
        "ingestion_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasets": {}
    }

    log_message("Starting MPLADS Data Ingestion & Inventory Operation")

    for filename in DATA_FILES:
        source_path = os.path.join(PROJECT_ROOT, filename)
        target_path = os.path.join(RAW_DATA_DIR, filename)

        if not os.path.exists(source_path):
            log_message(f"ERROR: Source file {filename} not found in project root {PROJECT_ROOT}")
            continue

        # Check if already present in target directory
        if os.path.exists(target_path):
            source_sha = compute_sha256(source_path)
            target_sha = compute_sha256(target_path)
            if source_sha == target_sha:
                log_message(f"Preserving existing raw file: {filename} in data/raw/ (Checksum match: {source_sha[:10]}...)")
            else:
                log_message(f"WARNING: File {filename} exists in data/raw/ but SHA256 differs! Preserving existing file per raw data immutability rules.")
        else:
            log_message(f"Copying {filename} to data/raw/...")
            shutil.copy2(source_path, target_path)
            log_message(f"Successfully copied {filename} to data/raw/")

        # Compute metadata from raw target file
        file_sha256 = compute_sha256(target_path)
        file_size_bytes = os.path.getsize(target_path)

        # Inspect dataframe
        df = pd.read_csv(target_path, low_memory=False)
        num_rows, num_cols = df.shape

        col_metadata = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = float((null_count / num_rows) * 100) if num_rows > 0 else 0.0
            col_metadata.append({
                "column_name": col,
                "data_type": str(df[col].dtype),
                "null_count": null_count,
                "null_percentage": round(null_pct, 4)
            })

        ds_info = {
            "filename": filename,
            "source_location": source_path,
            "target_location": target_path,
            "acquisition_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sha256_checksum": file_sha256,
            "size_bytes": file_size_bytes,
            "row_count": num_rows,
            "column_count": num_cols,
            "columns": col_metadata
        }

        manifest["datasets"][filename] = ds_info

        # Log overview
        log_message(f"Processed Dataset: {filename} | Rows: {num_rows} | Cols: {num_cols} | Size: {file_size_bytes/1024/1024:.2f} MB | SHA256: {file_sha256[:16]}...")

    # Write manifest JSON
    with open(MANIFEST_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_message(f"Manifest written to {MANIFEST_FILE_PATH}")

    # Print Data Inventory Report to stdout
    print("\n" + "="*80)
    print("                    MPLADS DATASET INVENTORY REPORT                    ")
    print("="*80)
    for fname, info in manifest["datasets"].items():
        print(f"\nFILENAME: {info['filename']}")
        print(f"Path: {info['target_location']}")
        print(f"SHA-256 Checksum: {info['sha256_checksum']}")
        print(f"File Size: {info['size_bytes']:,} bytes ({info['size_bytes']/1024/1024:.2f} MB)")
        print(f"Number of Rows: {info['row_count']:,}")
        print(f"Number of Columns: {info['column_count']}")
        print("\nCOLUMN DETAILS:")
        print(f"  {'Column Name':<35} | {'Data Type':<10} | {'Missing Count':<13} | {'Missing %':<10}")
        print("  " + "-"*75)
        for col in info["columns"]:
            print(f"  {col['column_name']:<35} | {col['data_type']:<10} | {col['null_count']:<13,} | {col['null_percentage']}%")
        print("-" * 80)

if __name__ == "__main__":
    ingest_and_inventory()
