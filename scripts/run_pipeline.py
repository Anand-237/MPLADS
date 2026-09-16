"""
MPLADS AI Monitoring System - Master End-to-End Pipeline Orchestrator (Stage 10)

Executes all 10 pipeline stages sequentially from raw input data:
- Stage 1: Data Ingestion & Schema Standardization
- Stage 2: Data Cleaning & Integration
- Stage 3A: Financial Feature Engineering
- Stage 3B: Temporal Feature Engineering
- Stage 3C: Vendor Feature Engineering
- Stage 3D: NLP Text Similarity Analysis
- Stage 4A: Isolation Forest Training Matrix Construction
- Stage 4B: Isolation Forest Anomaly Model Training
- Stage 4C: Statistical Anomaly Baselines & Cohort Profiling
- Stage 5: Explainable Multi-Factor Risk Scoring
- Stage 6: Anomaly Model Evaluation & Validation
- Stage 8: Master Anomaly Dataset Compilation
- Stage 9: Human Verification Audit Service Initialization

Usage: python scripts/run_pipeline.py
"""

import os
import sys
import time
import logging
import subprocess
import yaml
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "pipeline.log")
    
    logger = logging.getLogger("MPLADS_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File Handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    file_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)
    
    return logger

def run_pipeline():
    logger = setup_logging()
    t_start = time.time()
    
    logger.info("=" * 80)
    logger.info("      MPLADS AI ANOMALY MONITORING SYSTEM — MASTER PIPELINE (STAGE 10)     ")
    logger.info("=" * 80)
    
    # Verify Config
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Configuration file missing at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded Configuration file: {CONFIG_PATH}")
    
    # Pipeline Execution List
    stages = [
        ("Stage 1: Ingest Data", os.path.join(PROJECT_ROOT, "scripts", "data_ingestion", "ingest_data.py")),
        ("Stage 2: Clean Data", os.path.join(PROJECT_ROOT, "scripts", "preprocessing", "clean_mplads_data.py")),
        ("Stage 3A: Financial Features", os.path.join(PROJECT_ROOT, "scripts", "feature_engineering", "build_financial_features.py")),
        ("Stage 3B: Temporal Features", os.path.join(PROJECT_ROOT, "scripts", "feature_engineering", "build_temporal_features.py")),
        ("Stage 3C: Vendor Features", os.path.join(PROJECT_ROOT, "scripts", "feature_engineering", "build_vendor_features.py")),
        ("Stage 3D: NLP Similarity", os.path.join(PROJECT_ROOT, "scripts", "feature_engineering", "build_nlp_features.py")),
        ("Stage 4A: Isolation Forest Matrix", os.path.join(PROJECT_ROOT, "scripts", "feature_engineering", "build_isolation_forest_features.py")),
        ("Stage 4B: Train Isolation Forest", os.path.join(PROJECT_ROOT, "scripts", "models", "train_isolation_forest.py")),
        ("Stage 4C: Statistical Baselines", os.path.join(PROJECT_ROOT, "scripts", "models", "build_statistical_baselines.py")),
        ("Stage 5: Multi-Factor Risk Scoring", os.path.join(PROJECT_ROOT, "scripts", "evaluation", "build_risk_scores.py")),
        ("Stage 6: Model Evaluation", os.path.join(PROJECT_ROOT, "scripts", "evaluation", "evaluate_anomaly_model.py")),
        ("Stage 8: Master Anomaly Dataset", os.path.join(PROJECT_ROOT, "scripts", "data_processing", "build_master_anomaly_results.py")),
        ("Stage 9: Human Verification Service", os.path.join(PROJECT_ROOT, "scripts", "audit", "human_verification_service.py")),
    ]
    
    for idx, (stage_name, script_path) in enumerate(stages, 1):
        s_t0 = time.time()
        logger.info(f"\n---> [{idx}/{len(stages)}] EXECUTING {stage_name.upper()}...")
        logger.info(f"     Script: {script_path}")
        
        if not os.path.exists(script_path):
            logger.error(f"Script missing: {script_path}")
            sys.exit(1)
            
        try:
            res = subprocess.run([sys.executable, script_path], check=True)
            s_t1 = time.time()
            logger.info(f"✅ {stage_name} COMPLETED SUCCESSFULLY in {s_t1 - s_t0:.2f}s.")
        except subprocess.CalledProcessError as e:
            s_t1 = time.time()
            logger.error(f"❌ {stage_name} FAILED with exit code {e.returncode} after {s_t1 - s_t0:.2f}s!")
            sys.exit(1)
            
    t_end = time.time()
    logger.info("\n" + "=" * 80)
    logger.info(f"🎉 MASTER PIPELINE EXECUTION FINISHED SUCCESSFULLY IN {t_end - t_start:.2f} SECONDS!")
    logger.info("=" * 80)
    logger.info("Operational Policy Reminder: AI model predictions prioritize projects for human verification.")
    logger.info("The system strictly DOES NOT automatically claim fraud, corruption, or wrongdoing.")
    logger.info("Launch dashboard with: python -m streamlit run app.py")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_pipeline()
