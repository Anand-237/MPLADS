"""
MPLADS Sentinel - FastAPI Backend Server
Serves real-time REST API endpoints for the React + Vite Frontend Web Application.
"""

import os
import sys
import json
import sqlite3
import joblib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "mplads_database.db")
GEOJSON_PATH = os.path.join(PROJECT_ROOT, "data", "india_states.geojson")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
MASTER_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "mplads_anomaly_results.csv")

app = FastAPI(
    title="MPLADS Sentinel API",
    description="REST API for MPLADS AI Anomaly Oversight & Risk Intelligence",
    version="2.0.0"
)

# Enable CORS for React Frontend (default Vite port 5173 or any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded data and ML models
_db_cache = {}

def get_db_connection():
    if not os.path.exists(DB_PATH):
        gz_db = DB_PATH + ".gz"
        if os.path.exists(gz_db):
            try:
                import gzip, shutil
                with gzip.open(gz_db, 'rb') as f_in:
                    with open(DB_PATH, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception:
                pass
    if os.path.exists(DB_PATH):
        return sqlite3.connect(DB_PATH)
    return None

def clean_text_simple(text):
    if not text or pd.isnull(text):
        return ""
    text_str = str(text).lower()
    clean_chars = [c if c.isalnum() or c.isspace() else " " for c in text_str]
    return " ".join("".join(clean_chars).split())

def load_master_df():
    if "master_df" in _db_cache:
        return _db_cache["master_df"]
    conn = get_db_connection()
    if conn is not None:
        try:
            df = pd.read_sql_query("SELECT * FROM mplads_anomaly_results", conn)
            conn.close()
            _db_cache["master_df"] = df
            return df
        except Exception:
            pass
    target_master = MASTER_CSV if os.path.exists(MASTER_CSV) else (MASTER_CSV + ".gz")
    if os.path.exists(target_master):
        df = pd.read_csv(target_master, low_memory=False)
        _db_cache["master_df"] = df
        return df
    raise HTTPException(status_code=500, detail="Master dataset not found.")

def load_ml_models():
    if "clf" in _db_cache and "scaler" in _db_cache:
        return _db_cache["clf"], _db_cache["scaler"]
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        _db_cache["clf"] = clf
        _db_cache["scaler"] = scaler
        return clf, scaler
    return None, None

def build_tfidf():
    if "vec" in _db_cache and "X_tfidf" in _db_cache:
        return _db_cache["vec"], _db_cache["X_tfidf"]
    df = load_master_df()
    descs = [clean_text_simple(d) for d in df["work_description"].fillna("").tolist()]
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_tfidf = vec.fit_transform(descs)
    _db_cache["vec"] = vec
    _db_cache["X_tfidf"] = X_tfidf
    return vec, X_tfidf

# Pydantic Schemas
class ProposedWorkInput(BaseModel):
    work_description: str
    category: str
    recommended_amount: float
    sanctioned_amount: Optional[float] = None
    mp_name: Optional[str] = "Hon'ble Member of Parliament"
    constituency: Optional[str] = "General Constituency"
    state: Optional[str] = "General State"
    house: Optional[str] = "Lok Sabha"
    implementing_agency: Optional[str] = "District Authority"
    recommendation_date: Optional[str] = "2026-09-01"
    sanction_date: Optional[str] = None
    completion_date: Optional[str] = None

class AuditLogInput(BaseModel):
    work_id: str
    review_status: str
    reviewer_notes: str
    reviewer_id: str
    ai_risk_score_at_review: float
    ai_risk_category_at_review: str

# API Routes
@app.get("/")
def read_root():
    return {
        "system": "MPLADS Sentinel API",
        "status": "ONLINE",
        "database": "data/mplads_database.db",
        "version": "2.0.0"
    }

@app.get("/api/overview")
def get_overview():
    df = load_master_df()
    total_proj = len(df)
    high_cnt = int((df["risk_category"] == "HIGH").sum())
    med_cnt = int((df["risk_category"] == "MEDIUM").sum())
    low_cnt = int((df["risk_category"] == "LOW").sum())
    avg_score = float(df["final_risk_score"].mean())
    
    total_effective = float(df["effective_amount"].sum())
    
    # State risk ranking
    st_rank = df.groupby("state")["final_risk_score"].median().round(1).reset_index()
    st_rank.columns = ["state", "median_risk_score"]
    st_rank = st_rank.sort_values("median_risk_score", ascending=False).to_dict(orient="records")

    return {
        "total_projects": total_proj,
        "high_risk_count": high_cnt,
        "high_risk_pct": round((high_cnt / total_proj) * 100.0, 2),
        "medium_risk_count": med_cnt,
        "medium_risk_pct": round((med_cnt / total_proj) * 100.0, 2),
        "low_risk_count": low_cnt,
        "low_risk_pct": round((low_cnt / total_proj) * 100.0, 2),
        "average_risk_score": round(avg_score, 2),
        "total_effective_amount": total_effective,
        "top_state_risks": st_rank[:10]
    }

@app.get("/api/risk-map/states")
def get_state_risk_map():
    df = load_master_df()
    
    state_agg = df.groupby("state").agg(
        total_works=("work_id", "count"),
        high_risk_works=("risk_category", lambda x: (x == "HIGH").sum()),
        med_risk_works=("risk_category", lambda x: (x == "MEDIUM").sum()),
        low_risk_works=("risk_category", lambda x: (x == "LOW").sum()),
        median_risk_score=("final_risk_score", "median"),
        avg_risk_score=("final_risk_score", "mean"),
        total_amount=("effective_amount", "sum")
    ).reset_index()

    state_agg["high_risk_pct"] = np.round((state_agg["high_risk_works"] / state_agg["total_works"]) * 100.0, 2)
    state_agg["area_risk_score"] = state_agg["median_risk_score"].round(1)

    def assign_status(row):
        if row["total_works"] < 5:
            return "INSUFFICIENT DATA"
        elif row["area_risk_score"] >= 50.0 or row["high_risk_pct"] >= 10.0 or row["high_risk_works"] >= 10:
            return "HIGH"
        elif row["area_risk_score"] >= 40.0 or row["high_risk_pct"] >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"

    state_agg["area_status"] = state_agg.apply(assign_status, axis=1)
    return state_agg.sort_values("area_risk_score", ascending=False).to_dict(orient="records")

@app.get("/api/risk-map/constituencies")
def get_constituency_risk_map(state: Optional[str] = None):
    df = load_master_df()
    if state:
        df = df[df["state"] == state]
        
    const_agg = df.groupby(["state", "constituency"]).agg(
        total_works=("work_id", "count"),
        high_risk_works=("risk_category", lambda x: (x == "HIGH").sum()),
        medium_risk_works=("risk_category", lambda x: (x == "MEDIUM").sum()),
        low_risk_works=("risk_category", lambda x: (x == "LOW").sum()),
        median_risk_score=("final_risk_score", "median"),
        high_z_count=("cost_z_score", lambda x: (x > 2.5).sum()),
        high_sim_count=("nearest_similarity_score", lambda x: (x >= 0.85).sum()),
        high_iso_count=("isolation_forest_score", lambda x: (x >= 55.0).sum()),
        imp_date_count=("is_impossible_date_sequence", lambda x: (x == 1).sum())
    ).reset_index()

    const_agg["high_risk_pct"] = np.round((const_agg["high_risk_works"] / const_agg["total_works"]) * 100.0, 2)
    const_agg["area_risk_score"] = const_agg["median_risk_score"].round(1)

    def assign_status(row):
        if row["total_works"] < 5:
            return "INSUFFICIENT DATA"
        elif row["area_risk_score"] >= 50.0 or row["high_risk_pct"] >= 10.0 or row["high_risk_works"] >= 5:
            return "HIGH"
        elif row["area_risk_score"] >= 40.0 or row["high_risk_pct"] >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"

    const_agg["area_status"] = const_agg.apply(assign_status, axis=1)
    return const_agg.sort_values("area_risk_score", ascending=False).to_dict(orient="records")

@app.get("/api/risk-map/geojson")
def get_geojson():
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="GeoJSON file not found.")

@app.get("/api/projects")
def get_projects(
    state: Optional[str] = None,
    constituency: Optional[str] = None,
    category: Optional[str] = None,
    risk_category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    df = load_master_df()
    if state:
        df = df[df["state"] == state]
    if constituency:
        df = df[df["constituency"] == constituency]
    if category:
        df = df[df["category"] == category]
    if risk_category:
        df = df[df["risk_category"] == risk_category]
    if search:
        s_lower = search.lower()
        df = df[df["work_description"].fillna("").str.lower().str.contains(s_lower) | df["work_id"].fillna("").astype(str).str.contains(s_lower)]

    total_records = len(df)
    paged_df = df.sort_values("final_risk_score", ascending=False).iloc[offset : offset + limit]
    
    return {
        "total_records": total_records,
        "limit": limit,
        "offset": offset,
        "projects": paged_df.to_dict(orient="records")
    }

@app.post("/api/screen-work")
def screen_new_work(work: ProposedWorkInput):
    df = load_master_df()
    clf, scaler = load_ml_models()
    vec, X_tfidf = build_tfidf()
    
    prop_amt = float(work.recommended_amount)
    sanc_amt = float(work.sanctioned_amount) if work.sanctioned_amount is not None else prop_amt
    prop_cat = work.category
    rec_sanc_ratio = (sanc_amt / prop_amt) if prop_amt > 0 else 1.0
    
    # 1. Cost benchmark
    cat_df = df[df["category"] == prop_cat] if prop_cat in df["category"].values else df
    hist_median = float(cat_df["effective_amount"].median()) if len(cat_df) > 0 else prop_amt
    comp_count = len(cat_df)
    diff_pct = float(((prop_amt - hist_median) / hist_median) * 100.0) if hist_median > 0 else 0.0
    
    # Z-score
    cat_std = float(cat_df["effective_amount"].std()) if len(cat_df) > 1 else 1.0
    z_score = float((prop_amt - hist_median) / cat_std) if cat_std > 0 else 0.0
    
    # Date sequence checks
    is_impossible_date = 0
    is_negative_dur = 0
    is_covid = 0
    try:
        r_dt = pd.to_datetime(work.recommendation_date) if work.recommendation_date else pd.to_datetime("2026-09-01")
        s_dt = pd.to_datetime(work.sanction_date) if work.sanction_date else r_dt
        c_dt = pd.to_datetime(work.completion_date) if work.completion_date else (s_dt + pd.Timedelta(days=180))
        
        if s_dt < r_dt or c_dt < s_dt:
            is_impossible_date = 1
        if c_dt < s_dt:
            is_negative_dur = 1
        if pd.to_datetime("2020-03-01") <= r_dt <= pd.to_datetime("2021-12-31"):
            is_covid = 1
    except Exception:
        pass

    # 2. NLP Cosine Similarity
    clean_prop_desc = clean_text_simple(work.work_description)
    prop_vec = vec.transform([clean_prop_desc])
    sim_scores = cosine_similarity(prop_vec, X_tfidf).flatten()
    top_indices = np.argsort(sim_scores)[::-1][:5]
    
    similar_matches = []
    for idx in top_indices:
        match_row = df.iloc[idx]
        similar_matches.append({
            "work_id": str(match_row["work_id"]),
            "work_description": str(match_row["work_description"]),
            "category": str(match_row["category"]),
            "effective_amount": float(match_row["effective_amount"]),
            "state": str(match_row["state"]),
            "similarity_pct": float(round(sim_scores[idx] * 100.0, 1))
        })
    max_sim = float(round(sim_scores[top_indices[0]] * 100.0, 1)) if len(top_indices) > 0 else 0.0
    
    # 3. Isolation Forest Anomaly Score
    iso_score = 30.0
    iso_prediction_label = "NORMAL (1)"
    if clf is not None and scaler is not None and hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
        feat_dict = {
            "effective_amount": prop_amt,
            "cost_deviation": prop_amt - hist_median,
            "state_cost_percentile": 50.0,
            "constituency_cost_percentile": 50.0,
            "release_ratio": 1.0,
            "recommendation_sanction_ratio": float(rec_sanc_ratio),
            "is_impossible_date_sequence": int(is_impossible_date),
            "is_negative_duration": int(is_negative_dur),
            "is_covid_era": int(is_covid),
            "year_over_year_change": 0.0,
            "description_length": float(len(work.work_description)),
            "word_count": float(len(work.work_description.split())),
            "tfidf_mean": float(prop_vec.mean()),
            "tfidf_max": float(prop_vec.max()),
            "nearest_similarity_score": float(max_sim / 100.0),
            "similar_project_count": float((sim_scores >= 0.85).sum()),
            "constituency_max_vendor_share": 0.2,
            "constituency_avg_vendor_transaction": prop_amt,
            "constituency_vendor_count": 10.0,
            "constituency_avg_vendor_repeat_rate": 0.1,
            "constituency_pending_payment_ratio": 0.0
        }
        feat_row = [feat_dict.get(fname, 0.0) for fname in expected_features]
        X_feat_df = pd.DataFrame([feat_row], columns=expected_features)
        X_scaled = scaler.transform(X_feat_df)
        raw_dec = clf.decision_function(X_scaled)[0]
        inv_dec = -raw_dec
        iso_score = float(np.clip((inv_dec - (-0.20)) / (0.30 - (-0.20)) * 100.0, 0.0, 100.0))
        
        raw_pred = clf.predict(X_scaled)[0]
        if raw_pred == -1 or iso_score >= 55.0:
            iso_prediction_label = "ANOMALOUS (-1)"

    # Composite Risk Score
    comp_score = min(100.0, max(0.0, (0.45 * (min(z_score, 4.0)/4.0 * 100.0)) + (0.30 * max_sim) + (0.25 * iso_score)))
    risk_level = "HIGH" if comp_score >= 70.0 else ("MEDIUM" if comp_score >= 40.0 else "LOW")
    
    return {
        "proposed_amount": prop_amt,
        "sanctioned_amount": sanc_amt,
        "recommendation_sanction_ratio": round(rec_sanc_ratio, 2),
        "historical_median": round(hist_median, 2),
        "cost_difference_pct": round(diff_pct, 1),
        "comparable_projects_count": comp_count,
        "cost_z_score": round(z_score, 2),
        "nearest_similarity_pct": max_sim,
        "similar_matches": similar_matches,
        "isolation_forest_score": round(iso_score, 1),
        "isolation_forest_label": iso_prediction_label,
        "is_impossible_date_sequence": is_impossible_date,
        "final_risk_score": round(comp_score, 1),
        "risk_level": risk_level
    }

@app.get("/api/audit-logs")
def get_audit_logs():
    conn = get_db_connection()
    if conn is not None:
        try:
            df_log = pd.read_sql_query("SELECT * FROM review_log ORDER BY review_timestamp DESC", conn)
            conn.close()
            return df_log.to_dict(orient="records")
        except Exception:
            pass
    audit_csv = os.path.join(PROJECT_ROOT, "data", "audit", "review_log.csv")
    if os.path.exists(audit_csv):
        df_log = pd.read_csv(audit_csv, low_memory=False)
        return df_log.to_dict(orient="records")
    return []

@app.post("/api/audit-logs")
def add_audit_log(entry: AuditLogInput):
    conn = get_db_connection()
    timestamp = datetime.now(timezone.utc).isoformat()
    log_id = f"AUDIT_{int(datetime.now().timestamp()):06d}"
    
    new_data = {
        "log_id": log_id,
        "work_id": entry.work_id,
        "review_status": entry.review_status,
        "reviewer_notes": entry.reviewer_notes.strip(),
        "reviewer_id": entry.reviewer_id.strip() if entry.reviewer_id.strip() else "Auditor Officer",
        "review_timestamp": timestamp,
        "ai_risk_score_at_review": round(float(entry.ai_risk_score_at_review), 2),
        "ai_risk_category_at_review": entry.ai_risk_category_at_review
    }
    
    if conn is not None:
        try:
            pd.DataFrame([new_data]).to_sql("review_log", conn, if_exists="append", index=False)
            conn.close()
        except Exception:
            pass
            
    audit_csv = os.path.join(PROJECT_ROOT, "data", "audit", "review_log.csv")
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)
    if os.path.exists(audit_csv):
        df_ex = pd.read_csv(audit_csv)
        df_new = pd.concat([df_ex, pd.DataFrame([new_data])], ignore_index=True)
        df_new.to_csv(audit_csv, index=False)
    else:
        pd.DataFrame([new_data]).to_csv(audit_csv, index=False)
        
    return new_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
