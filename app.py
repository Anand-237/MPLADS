"""
MPLADS Audit Dashboard - Streamlit Application (Stage 8 & Stage 9)

Human-in-the-Loop Interactive Prototype for Anomaly & Risk Score Review.
Features 6 key views:
1. Overview Metrics & Macro Analytics
2. Project Explorer & Multi-Criteria Filtering
3. Project Detail Deep-Dive
4. Historical Benchmark Comparison
5. Similar Projects (NLP Work Description Match)
6. Human Verification Audit & Audit Trail Storage (Stage 9)

Run with: python -m streamlit run app.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_isolation_model():
    model_path = os.path.join("models", "isolation_forest.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return clf, scaler
    return None, None

def clean_text_simple(text):
    if pd.isnull(text):
        return ""
    text_str = str(text).lower()
    clean_chars = [c if c.isalnum() or c.isspace() else " " for c in text_str]
    return " ".join("".join(clean_chars).split())

@st.cache_resource
def build_tfidf_index_cached(descriptions_tuple):
    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )
    clean_descs = [clean_text_simple(d) for d in descriptions_tuple]
    X_tfidf = vec.fit_transform(clean_descs)
    return vec, X_tfidf

# Page Configuration
st.set_page_config(
    page_title="MPLADS Sentinel - AI Oversight Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page Configuration
st.set_page_config(
    page_title="MPLADS Sentinel - AI Oversight Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling (Ultra-Modern UI & Enlarged Text Size)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
    
    /* Global Viewport & Light Container Background */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F7F9FB !important;
        color: #0F172A !important;
        font-size: 1.15rem !important;
        line-height: 1.6;
    }

    /* Remove Black Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: #F7F9FB !important;
        background: transparent !important;
    }

    /* Ensure main view container is light background */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #F7F9FB !important;
    }
    
    [data-testid="stAppViewContainer"] {
        animation: fadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Institutional Dark Sidebar Styling (#0F172A) */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
        padding-top: 1.0rem;
    }
    
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] caption {
        color: #F8FAFC !important;
    }

    /* Ultra-Modern Sidebar Radio Navigation Buttons */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        gap: 12px;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        margin-bottom: 6px !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background-color: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
        transform: translateX(4px);
    }

    /* Typography Font Sizing & High Contrast Labels */
    .main-title {
        font-size: 2.75rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        color: #0F172A !important;
        margin-bottom: 0.3rem;
    }
    
    .sub-title {
        font-size: 1.3rem !important;
        color: #475569 !important;
        margin-bottom: 1.8rem;
    }

    h1, h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #0F172A !important;
    }

    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 1.95rem !important; }
    h3 { font-size: 1.6rem !important; }
    h4 { font-size: 1.35rem !important; }

    /* Main view labels and text */
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main span,
    [data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] p {
        color: #0F172A !important;
    }

    /* Expanders with Clean White/Light Background & High Contrast Labels */
    div[data-testid="stExpander"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        border-radius: 12px !important;
        color: #334155 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stExpander"] summary * {
        color: #334155 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Metrics & Card Containers - Compact & Proportionate Sizing */
    [data-testid="stMetric"], .metric-card {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 0.85rem 1.0rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        overflow: hidden !important;
    }
    
    [data-testid="stMetric"]:hover, .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08) !important;
        border-color: #94A3B8 !important;
    }
    
    /* Responsive & Fully Visible Metric Values Without Truncation Points */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', -apple-system, monospace !important;
        font-size: clamp(1.0rem, 1.2vw, 1.45rem) !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
    }

    [data-testid="stMetricValue"] > div {
        font-size: clamp(1.0rem, 1.2vw, 1.45rem) !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 0.2rem !important;
        white-space: nowrap !important;
    }

    /* Form Field Labels */
    [data-testid="stAppViewContainer"] .main label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* Multiselect / Selectbox White Dropdown Control ("Choose options") */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        color: #0F172A !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #CBD5E1 !important;
        background-color: #FFFFFF !important;
    }

    /* Select Placeholder Text */
    div[data-baseweb="select"] [data-aria-hidden="true"],
    div[data-baseweb="select"] span {
        color: #64748B !important;
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
    }

    div[data-baseweb="option"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    div[data-baseweb="option"]:hover {
        background-color: #F1F5F9 !important;
    }

    /* Input Numbers and Buttons */
    div[data-baseweb="input"] > div {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }

    div[data-baseweb="input"] input {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Secondary Buttons (Reset Filters) */
    button[kind="secondary"],
    button[type="secondary"],
    div[data-testid="stButton"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #2563EB !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover,
    button[type="secondary"]:hover,
    div[data-testid="stButton"] button:hover {
        background-color: #F8FAFC !important;
        border-color: #2563EB !important;
        color: #1D4ED8 !important;
    }
    
    /* Table Typography */
    .stDataFrame, table {
        font-size: 1.05rem !important;
    }
    
    table th {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.0rem !important;
        background-color: #F1F5F9 !important;
        color: #334155 !important;
    }

    /* Tabs Styling */
    [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
    }
    
    [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        color: #475569 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    
    /* Status Badges */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1.5px solid #FCA5A5;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.0rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1.5px solid #FCD34D;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.0rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1.5px solid #6EE7B7;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.0rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Data Paths & Database Path
DATA_PATH = os.path.join("data", "processed", "mplads_anomaly_results.csv")
SIMILAR_PATH = os.path.join("data", "processed", "potentially_similar_projects.csv")
AUDIT_LOG_PATH = os.path.join("data", "audit", "review_log.csv")
DB_PATH = os.path.join("data", "mplads_database.db")

def get_sqlite_connection():
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
        try:
            conn = sqlite3.connect(DB_PATH)
            return conn
        except Exception:
            return None
    return None

@st.cache_data
def load_master_data():
    conn = get_sqlite_connection()
    if conn is not None:
        try:
            df = pd.read_sql_query("SELECT * FROM mplads_anomaly_results", conn)
            conn.close()
            
            # Load audit status from SQLite review_log
            audit_df = load_audit_trail_log()
            if len(audit_df) > 0 and "work_id" in audit_df.columns:
                latest_audit = audit_df.sort_values("review_timestamp").groupby("work_id").last().reset_index()
                df = df.drop(columns=["verification_status"], errors="ignore")
                df = pd.merge(df, latest_audit[["work_id", "review_status"]], on="work_id", how="left")
                df["verification_status"] = df["review_status"].fillna("Not Reviewed")
                df = df.drop(columns=["review_status"], errors="ignore")
            else:
                if "verification_status" not in df.columns:
                    df["verification_status"] = "Not Reviewed"
            return df
        except Exception:
            pass

    target_data_path = DATA_PATH if os.path.exists(DATA_PATH) else (DATA_PATH + ".gz")
    if not os.path.exists(target_data_path):
        st.error(f"Dataset missing at {DATA_PATH}. Please run data pipelines first.")
        st.stop()
    df = pd.read_csv(target_data_path, low_memory=False)
    
    # Ensure MP summary columns exist even if old cached version is present
    if "mp_allocated_amount" not in df.columns or "mp_completed_works" not in df.columns:
        mp_summary_path = os.path.join("data", "processed", "mp_summary_clean.csv")
        target_mp_path = mp_summary_path if os.path.exists(mp_summary_path) else (mp_summary_path + ".gz")
        if os.path.exists(target_mp_path):
            mp_df = pd.read_csv(target_mp_path, low_memory=False)
            df = pd.merge(df, mp_df[["mp_name", "allocated_amount", "total_expenditure", "utilization_pct", "completed_works_count", "recommended_works_count", "completion_rate_pct"]], on="mp_name", how="left")
            df["mp_allocated_amount"] = df["allocated_amount"].fillna(0.0)
            df["mp_total_expenditure"] = df["total_expenditure"].fillna(0.0)
            df["mp_utilization_pct"] = df["utilization_pct"].fillna(0.0)
            df["mp_completed_works"] = df["completed_works_count"].fillna(0).astype(int)
            df["mp_recommended_works"] = df["recommended_works_count"].fillna(0).astype(int)
            df["mp_completion_rate_pct"] = df["completion_rate_pct"].fillna(0.0)
        else:
            df["mp_allocated_amount"] = 0.0
            df["mp_total_expenditure"] = 0.0
            df["mp_utilization_pct"] = 0.0
            df["mp_completed_works"] = 0
            df["mp_recommended_works"] = 0
            df["mp_completion_rate_pct"] = 0.0

    # Load latest review status from audit trail log
    target_audit_path = AUDIT_LOG_PATH if os.path.exists(AUDIT_LOG_PATH) else (AUDIT_LOG_PATH + ".gz")
    if os.path.exists(target_audit_path):
        audit_df = pd.read_csv(target_audit_path, low_memory=False)
        if len(audit_df) > 0 and "work_id" in audit_df.columns:
            latest_audit = audit_df.sort_values("review_timestamp").groupby("work_id").last().reset_index()
            df = df.drop(columns=["verification_status"], errors="ignore")
            df = pd.merge(df, latest_audit[["work_id", "review_status"]], on="work_id", how="left")
            df["verification_status"] = df["review_status"].fillna("Not Reviewed")
            df = df.drop(columns=["review_status"], errors="ignore")
        else:
            if "verification_status" not in df.columns:
                df["verification_status"] = "Not Reviewed"
    else:
        if "verification_status" not in df.columns:
            df["verification_status"] = "Not Reviewed"
            
    return df

@st.cache_data
def load_similar_pairs():
    conn = get_sqlite_connection()
    if conn is not None:
        try:
            df_sim = pd.read_sql_query("SELECT * FROM potentially_similar_projects", conn)
            conn.close()
            return df_sim
        except Exception:
            pass
    target_sim_path = SIMILAR_PATH if os.path.exists(SIMILAR_PATH) else (SIMILAR_PATH + ".gz")
    if os.path.exists(target_sim_path):
        return pd.read_csv(target_sim_path, low_memory=False)
    return pd.DataFrame()

def load_audit_trail_log():
    conn = get_sqlite_connection()
    if conn is not None:
        try:
            df_audit = pd.read_sql_query("SELECT * FROM review_log", conn)
            conn.close()
            return df_audit
        except Exception:
            pass
    if os.path.exists(AUDIT_LOG_PATH):
        return pd.read_csv(AUDIT_LOG_PATH, low_memory=False)
    return pd.DataFrame(columns=[
        "log_id", "work_id", "review_status", "reviewer_notes",
        "reviewer_id", "review_timestamp", "ai_risk_score_at_review",
        "ai_risk_category_at_review"
    ])

def append_to_audit_log(work_id, review_status, reviewer_notes, reviewer_id, ai_score, ai_category):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    df_existing = load_audit_trail_log()
    
    next_id_num = len(df_existing) + 1
    log_id = f"AUDIT_{next_id_num:06d}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    new_entry = {
        "log_id": log_id,
        "work_id": work_id,
        "review_status": review_status,
        "reviewer_notes": reviewer_notes.strip(),
        "reviewer_id": reviewer_id.strip() if reviewer_id.strip() else "Auditor Officer",
        "review_timestamp": timestamp,
        "ai_risk_score_at_review": round(float(ai_score), 2),
        "ai_risk_category_at_review": str(ai_category)
    }
    
    conn = get_sqlite_connection()
    if conn is not None:
        try:
            pd.DataFrame([new_entry]).to_sql("review_log", conn, if_exists="append", index=False)
            conn.close()
        except Exception:
            pass
            
    df_new = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
    df_new.to_csv(AUDIT_LOG_PATH, index=False, encoding="utf-8")
    return new_entry

df = load_master_data()
similar_df = load_similar_pairs()

# Fast O(1) Dictionary Lookups for Selectbox Formatting
work_to_risk = dict(zip(df["work_id"], df["final_risk_score"]))
work_to_cat = dict(zip(df["work_id"], df["risk_category"]))
work_to_grp = dict(zip(df["work_id"], df["baseline_group"]))
work_to_status = dict(zip(df["work_id"], df["verification_status"]))

# Header Banner & Neutral Compliance Statement
st.markdown('<div class="main-title">🛡️ MPLADS Human-in-the-Loop Audit Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable Anomaly Detection, Benchmark Deviations & Priority Verification Workflows</div>', unsafe_allow_html=True)

st.info("ℹ️ **Operational Policy**: High and Medium risk designations reflect multi-factor statistical deviations across expenditure, duration, vendor concentration, or text similarity. Automated labeling of fraud or corruption is strictly prohibited. Use scores to prioritize human administrative verification.")

# Sidebar Navigation
st.sidebar.title("🛡️ MPLADS SENTINEL")
st.sidebar.caption("AI Governance & Anomaly Audit System")

nav_selection = st.sidebar.radio(
    "Select System Section:",
    [
        "🗺️ Risk Map",
        "📊 Overview",
        "🆕 New Work Screening",
        "🔍 Project Explorer",
        "📄 Project Detail",
        "📈 Historical Benchmark",
        "🔎 Similar Projects (NLP)",
        "📋 Human Verification Audit"
    ]
)

# -----------------------------------------------------------------------------
# 1. OVERVIEW SECTION
# -----------------------------------------------------------------------------
if nav_selection == "📊 Overview":
    st.header("📊 Executive Anomaly & Risk Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_proj = len(df)
    high_cnt = len(df[df["risk_category"] == "HIGH"])
    med_cnt = len(df[df["risk_category"] == "MEDIUM"])
    low_cnt = len(df[df["risk_category"] == "LOW"])
    avg_score = df["final_risk_score"].mean()
    
    col1.metric("Total Projects", f"{total_proj:,}")
    col2.metric("HIGH Risk Priority", f"{high_cnt:,}", f"{(high_cnt/total_proj)*100:.2f}%", delta_color="inverse")
    col3.metric("MEDIUM Risk Priority", f"{med_cnt:,}", f"{(med_cnt/total_proj)*100:.2f}%", delta_color="off")
    col4.metric("LOW Risk Baseline", f"{low_cnt:,}", f"{(low_cnt/total_proj)*100:.2f}%", delta_color="normal")
    col5.metric("Average Risk Score", f"{avg_score:.2f} / 100")
    
    # Financial & MP Works Summary Metrics
    st.markdown("#### 💰 Corpus Financial & MP Works Summary")
    m1, m2, m3, m4 = st.columns(4)
    total_effective = df["effective_amount"].sum()
    total_allocated = df.groupby("mp_name")["mp_allocated_amount"].first().sum()
    total_mp_exp = df.groupby("mp_name")["mp_total_expenditure"].first().sum()
    total_completed_works = df.groupby("mp_name")["mp_completed_works"].first().sum()
    
    m1.metric("Total Effective Project Value", f"₹{total_effective:,.2f}")
    m2.metric("Total MP Allocated Funds", f"₹{total_allocated:,.2f}")
    m3.metric("Total MP Cumulative Expenditure", f"₹{total_mp_exp:,.2f}")
    m4.metric("Total MP Completed Works", f"{total_completed_works:,}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk Priority Category Breakdown")
        cat_counts = df["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Risk Category", "Count"]
        fig_pie = px.pie(
            cat_counts,
            names="Risk Category",
            values="Count",
            color="Risk Category",
            color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, width="stretch")
        
    with c2:
        st.subheader("Final Risk Score Distribution")
        fig_hist = px.histogram(
            df,
            x="final_risk_score",
            nbins=40,
            color="risk_category",
            color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
            labels={"final_risk_score": "Final Risk Score (0-100)", "count": "Project Count"}
        )
        st.plotly_chart(fig_hist, width="stretch")
        
    st.subheader("Top 10 States by High-Risk Project Count")
    high_state_df = df[df["risk_category"] == "HIGH"]["state"].value_counts().head(10).reset_index()
    high_state_df.columns = ["State", "High Risk Count"]
    fig_state = px.bar(
        high_state_df,
        x="State",
        y="High Risk Count",
        color="High Risk Count",
        color_continuous_scale="Reds",
        labels={"High Risk Count": "High Priority Projects"}
    )
    st.plotly_chart(fig_state, width="stretch")

# -----------------------------------------------------------------------------
# 2. PROJECT EXPLORER SECTION
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Project Explorer":
    st.header("🔍 Interactive Project Explorer & Multi-Criteria Filter")
    
    with st.expander("🔻 Search & Filter Options", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            selected_states = st.multiselect("State:", sorted(df["state"].dropna().unique().tolist()))
            selected_categories = st.multiselect("Project Type / Category:", sorted(df["category"].dropna().unique().tolist()))
            
        with f_col2:
            constituencies = sorted(df["constituency"].dropna().unique().tolist())
            if selected_states:
                constituencies = sorted(df[df["state"].isin(selected_states)]["constituency"].dropna().unique().tolist())
            selected_constituencies = st.multiselect("Constituency:", constituencies)
            selected_risk = st.multiselect("Risk Category:", ["HIGH", "MEDIUM", "LOW"], default=["HIGH"])
            
        with f_col3:
            years = sorted(df["recommendation_year"].dropna().unique().tolist())
            selected_years = st.multiselect("Recommendation Year:", years)
            search_id = st.text_input("Search Work ID / MP Name / Description:")

    # Apply Filters
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
    if selected_constituencies:
        filtered_df = filtered_df[filtered_df["constituency"].isin(selected_constituencies)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_risk:
        filtered_df = filtered_df[filtered_df["risk_category"].isin(selected_risk)]
    if selected_years:
        filtered_df = filtered_df[filtered_df["recommendation_year"].isin(selected_years)]
    if search_id:
        s_term = search_id.strip().lower()
        filtered_df = filtered_df[
            filtered_df["work_id"].astype(str).str.lower().str.contains(s_term) |
            filtered_df["mp_name"].astype(str).str.lower().str.contains(s_term) |
            filtered_df["work_description"].astype(str).str.lower().str.contains(s_term)
        ]
        
    st.write(f"Showing **{len(filtered_df):,}** matching projects out of {len(df):,}")
    
    display_cols = [
        "work_id", "mp_name", "state", "constituency", "category", 
        "effective_amount", "mp_allocated_amount", "mp_total_expenditure",
        "mp_completed_works", "mp_completion_rate_pct",
        "final_risk_score", "risk_category", "isolation_forest_score",
        "cost_z_score", "verification_status"
    ]
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[valid_cols].sort_values("final_risk_score", ascending=False),
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# 3. PROJECT DETAIL SECTION
# -----------------------------------------------------------------------------
elif nav_selection == "📄 Project Detail":
    st.header("📄 Comprehensive Project Audit & Risk Detail")
    
    # Filter dropdown by Risk Category or State for fast browsing
    d_f1, d_f2 = st.columns(2)
    with d_f1:
        sel_risk_filter = st.selectbox("Filter Projects by Risk Category:", ["HIGH", "MEDIUM", "LOW", "All"], index=0)
    with d_f2:
        sel_state_filter = st.selectbox("Filter Projects by State:", ["All"] + sorted(df["state"].dropna().unique().tolist()), index=0)
        
    detail_df = df.copy()
    if sel_risk_filter != "All":
        detail_df = detail_df[detail_df["risk_category"] == sel_risk_filter]
    if sel_state_filter != "All":
        detail_df = detail_df[detail_df["state"] == sel_state_filter]
        
    if len(detail_df) == 0:
        st.warning("No projects match the selected filters.")
        st.stop()
        
    selected_work_id = st.selectbox(
        "Select Work ID for Deep-Dive Review:",
        options=detail_df["work_id"].tolist(),
        format_func=lambda x: f"Work ID #{x} ({work_to_cat.get(x, 'LOW')} Risk - Score: {work_to_risk.get(x, 0.0):.2f})"
    )
    
    p_data = df[df["work_id"] == selected_work_id].iloc[0]
    
    st.subheader(f"Project Work ID: #{p_data['work_id']}")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write(f"**State**: {p_data['state']}")
        st.write(f"**Constituency**: {p_data['constituency']}")
        st.write(f"**MP Name**: {p_data['mp_name']}")
    with col_b:
        st.write(f"**Category**: {p_data['category']}")
        st.write(f"**Status**: {p_data['project_status']}")
        st.write(f"**Recommendation Year**: {p_data['recommendation_year']}")
    with col_c:
        st.metric("Final Risk Score", f"{p_data['final_risk_score']:.2f} / 100", delta=p_data['risk_category'])
        st.write(f"**Verification Status**: `{p_data['verification_status']}`")

    st.divider()
    
    # 4 Financial / Temporal / Vendor / Model Cards
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("**💰 Financial & MP Summary**")
        st.write(f"Effective Amount: **₹{p_data['effective_amount']:,.2f}**")
        st.write(f"Recommended Amount: ₹{p_data['recommended_amount']:,.2f}")
        
        sanc_amt = float(p_data['sanctioned_amount'])
        if sanc_amt > 0:
            st.write(f"Sanctioned Amount: ₹{sanc_amt:,.2f}")
        else:
            st.write("Sanctioned Amount: ₹0.00 *(Pending Sanction)*")
            
        st.write(f"MP Allocated Amount: **₹{p_data['mp_allocated_amount']:,.2f}**")
        st.write(f"MP Total Expenditure: **₹{p_data['mp_total_expenditure']:,.2f}**")
        st.write(f"MP Utilization: **{p_data['mp_utilization_pct']:.2f}%**")
        st.write(f"MP Completed Works: **{p_data['mp_completed_works']}**")
        st.write(f"MP Recommended Works: {p_data['mp_recommended_works']}")
        st.write(f"MP Completion Rate: **{p_data['mp_completion_rate_pct']:.2f}%**")
    with d2:
        st.markdown("**⏱️ Temporal Details**")
        st.write(f"Recommendation: {p_data['recommendation_date']}")
        
        comp_date = str(p_data['completion_date']).strip() if pd.notnull(p_data['completion_date']) else ""
        if comp_date and comp_date.lower() not in ["none", "nan", "nat", ""]:
            st.write(f"Completion: {comp_date}")
        else:
            st.write("Completion: *Pending Completion*")
            
        dur_days = float(p_data['duration_days']) if pd.notnull(p_data['duration_days']) else 0.0
        if comp_date and comp_date.lower() not in ["none", "nan", "nat", ""] and dur_days > 0:
            st.write(f"Duration: {dur_days:.0f} days")
        elif dur_days < 0:
            st.write(f"Duration: **{dur_days:.0f} days** *(Invalid Date Flag)*")
        else:
            st.write("Duration: *Pending Completion (N/A)*")
            
        st.write(f"Impossible Date Flag: {'YES' if p_data['is_impossible_date_sequence']==1 else 'No'}")
    with d3:
        st.markdown("**🏪 Vendor Risk Metrics**")
        st.write(f"Max Vendor Share: **{p_data['constituency_max_vendor_share']*100:.1f}%**")
        
        pend_ratio = float(p_data['constituency_pending_payment_ratio']) * 100.0
        if pend_ratio > 0:
            st.write(f"Pending Payment Ratio: **{pend_ratio:.1f}%**")
        else:
            st.write("Pending Payment Ratio: **0.0%** *(No Pending Payment Backlog)*")
            
        st.write(f"Vendor Score: {p_data['vendor_score']:.1f}/100")
    with d4:
        st.markdown("**🤖 Model Scores**")
        st.write(f"Isolation Forest: **{p_data['isolation_forest_score']:.1f}/100**")
        st.write(f"Cost Z-Score: **+{p_data['cost_z_score']:.2f}**")
        st.write(f"Percentile Rank: **{p_data['percentile']:.1f}%**")
        works_risk = max(0.0, 100.0 - float(p_data['mp_completion_rate_pct']))
        st.write(f"MP Works Risk Score: **{works_risk:.1f}/100**")

    st.subheader("💡 Evidence-Grounded Explanation")
    st.text_area("Elaborated Audit Explanation", value=p_data['explanation'], height=150, disabled=True)

# -----------------------------------------------------------------------------
# 4. HISTORICAL BENCHMARK SECTION
# -----------------------------------------------------------------------------
elif nav_selection == "📈 Historical Benchmark":
    st.header("📈 Historical Benchmark Cohort Comparison")
    
    # Filter Controls for Fast Navigation
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        bm_risk_filter = st.selectbox("Risk Filter:", ["HIGH", "MEDIUM", "LOW", "All"], index=0, key="bm_risk")
    with h_col2:
        all_groups = ["All"] + sorted(df["baseline_group"].dropna().unique().tolist())
        bm_group_filter = st.selectbox("Filter by Baseline Cohort Group:", all_groups, index=0, key="bm_group")
    with h_col3:
        bm_search = st.text_input("Search Work ID / MP Name:", key="bm_search")
        
    bm_df = df.copy()
    if bm_risk_filter != "All":
        bm_df = bm_df[bm_df["risk_category"] == bm_risk_filter]
    if bm_group_filter != "All":
        bm_df = bm_df[bm_df["baseline_group"] == bm_group_filter]
    if bm_search:
        b_term = bm_search.strip().lower()
        bm_df = bm_df[
            bm_df["work_id"].astype(str).str.lower().str.contains(b_term) |
            bm_df["mp_name"].astype(str).str.lower().str.contains(b_term)
        ]
        
    if len(bm_df) == 0:
        st.warning("No projects match the selected benchmark filters.")
        st.stop()
        
    selected_work_id = st.selectbox(
        "Select Project to Compare Against Historical Cohort:",
        options=bm_df["work_id"].tolist(),
        format_func=lambda x: f"Work ID #{x} (Group: {work_to_grp.get(x, 'N/A')} | Risk: {work_to_cat.get(x, 'LOW')})"
    )
    
    p_data = df[df["work_id"] == selected_work_id].iloc[0]
    
    b_grp = p_data["baseline_group"]
    eff_amt = p_data["effective_amount"]
    grp_med = p_data["group_median"]
    grp_mean = p_data["group_mean"]
    grp_p99 = p_data["group_p99"]
    z_val = p_data["cost_z_score"]
    pct = p_data["percentile"]
    
    st.markdown(f"### Baseline Group: `{b_grp}`")
    
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Project Value", f"₹{eff_amt:,.2f}")
    bm2.metric("Cohort Median (P50)", f"₹{grp_med:,.2f}")
    bm3.metric("Cost Deviation (Z-Score)", f"+{z_val:.2f} σ")
    bm4.metric("Cohort Percentile Rank", f"{pct:.1f}%")
    
    st.divider()
    
    # Plotly Bar Comparison Chart
    fig_bm = io.Figure()
    fig_bm.add_trace(io.Bar(name="Selected Project Value", x=["Expenditure Benchmark"], y=[eff_amt], marker_color="#EF4444"))
    fig_bm.add_trace(io.Bar(name="Cohort Median (P50)", x=["Expenditure Benchmark"], y=[grp_med], marker_color="#10B981"))
    fig_bm.add_trace(io.Bar(name="Cohort Mean (μ)", x=["Expenditure Benchmark"], y=[grp_mean], marker_color="#3B82F6"))
    fig_bm.add_trace(io.Bar(name="Cohort 99th Percentile (P99)", x=["Expenditure Benchmark"], y=[grp_p99], marker_color="#F59E0B"))
    
    fig_bm.update_layout(barmode="group", title=f"Expenditure Benchmark Comparison for Work ID #{selected_work_id}", yaxis_title="Amount (₹)")
    st.plotly_chart(fig_bm, width="stretch")
    
    # Cohort Summary Table
    st.subheader("📋 Top Baseline Cohorts Expenditure Statistics Summary")
    cohort_summary = df.groupby("baseline_group").agg(
        project_count=("work_id", "count"),
        median_expenditure=("group_median", "first"),
        mean_expenditure=("group_mean", "first"),
        p99_ceiling=("group_p99", "first")
    ).reset_index().sort_values("project_count", ascending=False).head(15)
    
    cohort_summary.columns = ["Baseline Group Cohort", "Project Count", "Cohort Median (₹)", "Cohort Mean (₹)", "Cohort P99 Ceiling (₹)"]
    st.dataframe(cohort_summary, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# 5. SIMILAR PROJECTS SECTION (NLP)
# -----------------------------------------------------------------------------
elif nav_selection == "🔎 Similar Projects (NLP)":
    st.header("🔎 Potentially Similar Project Work Descriptions (NLP Analysis)")
    st.markdown("Discover projects with high text similarity (>= 85% Cosine Similarity) using TF-IDF N-gram analysis.")
    
    tab_mode1, tab_mode2 = st.tabs([
        "🔍 Existing Project Similarity Lookup",
        "⚡ Custom Text Real-Time Similarity Search"
    ])
    
    with tab_mode1:
        # Filter controls for fast project selection
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            sim_risk_filter = st.selectbox("Filter by Risk Category:", ["HIGH", "MEDIUM", "LOW", "All"], index=0, key="sim_risk")
        with s_c2:
            sim_state_filter = st.selectbox("Filter by State:", ["All"] + sorted(df["state"].dropna().unique().tolist()), index=0, key="sim_state")
        with s_c3:
            sim_search_input = st.text_input("Search Work ID or Description Keyword:", key="sim_search_inp")
            
        sim_df_filtered = df.copy()
        if sim_risk_filter != "All":
            sim_df_filtered = sim_df_filtered[sim_df_filtered["risk_category"] == sim_risk_filter]
        if sim_state_filter != "All":
            sim_df_filtered = sim_df_filtered[sim_df_filtered["state"] == sim_state_filter]
        if sim_search_input:
            sterm = sim_search_input.strip().lower()
            sim_df_filtered = sim_df_filtered[
                sim_df_filtered["work_id"].astype(str).str.lower().str.contains(sterm) |
                sim_df_filtered["work_description"].astype(str).str.lower().str.contains(sterm) |
                sim_df_filtered["mp_name"].astype(str).str.lower().str.contains(sterm)
            ]
            
        if len(sim_df_filtered) == 0:
            st.warning("No projects match the selected filters.")
        else:
            selected_work_id = st.selectbox(
                "Select Target Project Work ID:",
                options=sim_df_filtered["work_id"].tolist(),
                format_func=lambda x: f"Work ID #{x} ({work_to_cat.get(x, 'LOW')} Risk - Max Similarity: {df[df['work_id']==x]['nearest_similarity_score'].values[0]*100:.1f}%)",
                key="sim_work_select"
            )
            
            p_data = df[df["work_id"] == selected_work_id].iloc[0]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Target Work ID", f"#{selected_work_id}")
            m2.metric("State / Constituency", f"{p_data['state']} / {p_data['constituency']}")
            m3.metric("Nearest Cosine Similarity", f"{p_data['nearest_similarity_score']*100:.1f}%")
            m4.metric("Similar Candidate Matches", f"{int(p_data.get('similar_project_count', 0))}")
            
            st.info(f"**Target Work Description**: {p_data['work_description']}")
            st.divider()
            
            # Find matches in precomputed table (supporting both work_id_1/work_id_2 and project_id_1/project_id_2)
            id_col1 = "work_id_1" if "work_id_1" in similar_df.columns else ("project_id_1" if "project_id_1" in similar_df.columns else None)
            id_col2 = "work_id_2" if "work_id_2" in similar_df.columns else ("project_id_2" if "project_id_2" in similar_df.columns else None)
            
            matches = pd.DataFrame()
            if id_col1 and id_col2 and len(similar_df) > 0:
                matches = similar_df[(similar_df[id_col1] == selected_work_id) | (similar_df[id_col2] == selected_work_id)]
                
            if len(matches) > 0:
                st.subheader(f"High Similarity Pairwise Matches Identified ({len(matches)} matches)")
                st.dataframe(matches, width="stretch", hide_index=True)
            else:
                # Real-time fallback search against the corpus via TF-IDF
                st.subheader("🔍 Real-Time TF-IDF Similarity Matches (Top 10)")
                all_descs = df["work_description"].fillna("").tolist()
                vec, X_tfidf = build_tfidf_index_cached(tuple(all_descs))
                
                target_clean = clean_text_simple(p_data['work_description'])
                target_vec = vec.transform([target_clean])
                sim_scores = cosine_similarity(target_vec, X_tfidf).flatten()
                
                top_indices = np.argsort(sim_scores)[::-1]
                top_matches = []
                for idx in top_indices:
                    cand_id = df.iloc[idx]["work_id"]
                    if cand_id != selected_work_id:
                        top_matches.append({
                            "Matched Work ID": cand_id,
                            "Cosine Similarity": f"{sim_scores[idx]*100:.1f}%",
                            "State": df.iloc[idx]["state"],
                            "Constituency": df.iloc[idx]["constituency"],
                            "MP Name": df.iloc[idx]["mp_name"],
                            "Amount (₹)": f"₹{df.iloc[idx]['effective_amount']:,.2f}",
                            "Matched Work Description": df.iloc[idx]["work_description"]
                        })
                    if len(top_matches) >= 10:
                        break
                        
                top_df = pd.DataFrame(top_matches)
                st.dataframe(top_df, width="stretch", hide_index=True)
                
    with tab_mode2:
        st.subheader("⚡ Search Any Custom Work Description Against Corpus")
        user_query = st.text_input("Enter work description to find similar historical projects:", value="Construction of CC Road and Drain", key="custom_nlp_input")
        
        if user_query:
            all_descs = df["work_description"].fillna("").tolist()
            vec, X_tfidf = build_tfidf_index_cached(tuple(all_descs))
            
            clean_q = clean_text_simple(user_query)
            q_vec = vec.transform([clean_q])
            q_scores = cosine_similarity(q_vec, X_tfidf).flatten()
            
            top_q_indices = np.argsort(q_scores)[::-1][:10]
            q_results = []
            for idx in top_q_indices:
                q_results.append({
                    "Work ID": df.iloc[idx]["work_id"],
                    "Cosine Similarity": f"{q_scores[idx]*100:.1f}%",
                    "State": df.iloc[idx]["state"],
                    "Constituency": df.iloc[idx]["constituency"],
                    "Category": df.iloc[idx]["category"],
                    "Amount (₹)": f"₹{df.iloc[idx]['effective_amount']:,.2f}",
                    "Risk Score": f"{df.iloc[idx]['final_risk_score']:.1f}",
                    "Work Description": df.iloc[idx]["work_description"]
                })
            
            q_df = pd.DataFrame(q_results)
            st.dataframe(q_df, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# 6. HUMAN VERIFICATION AUDIT SECTION (STAGE 9 ENHANCED)
# -----------------------------------------------------------------------------
elif nav_selection == "📋 Human Verification Audit":
    st.header("📋 Stage 9: Human-in-the-Loop Verification & Audit Trail Manager")
    st.markdown("Auditors investigate AI-generated anomaly signals, review full evidence vectors, and record official audit decisions.")
    
    v_f1, v_f2, v_f3 = st.columns(3)
    with v_f1:
        v_risk_filter = st.selectbox("Filter Flagged Projects by Risk Category:", ["HIGH", "MEDIUM", "LOW", "All"], index=0, key="v_risk_stg9")
    with v_f2:
        v_status_filter = st.selectbox("Filter by Audit Review Status:", ["All", "Not Reviewed", "Verified Normal", "Needs Further Review", "Evidence Insufficient", "Manual Investigation Required"], index=0, key="v_status_stg9")
    with v_f3:
        v_work_id_search = st.text_input("🔎 Search / Check by Work ID #:", value="", placeholder="Enter Work ID (e.g. 1024)...", key="v_work_id_stg9", help="Directly type or paste a Work ID to inspect")
        
    v_df_select = df.copy()
    if v_work_id_search.strip():
        search_term = v_work_id_search.strip().lower()
        v_df_select = v_df_select[v_df_select["work_id"].astype(str).str.lower().str.contains(search_term)]
    if v_risk_filter != "All":
        v_df_select = v_df_select[v_df_select["risk_category"] == v_risk_filter]
    if v_status_filter != "All":
        v_df_select = v_df_select[v_df_select["verification_status"] == v_status_filter]
        
    if len(v_df_select) == 0:
        st.warning(f"No projects match the selected filters or Work ID search '{v_work_id_search}'. Try adjusting your search query.")
        st.stop()
        
    selected_work_id = st.selectbox(
        "Select Flagged Project for Official Audit Review:",
        options=v_df_select["work_id"].tolist(),
        format_func=lambda x: f"Work ID #{x} ({work_to_cat.get(x, 'LOW')} Risk - Status: {work_to_status.get(x, 'Not Reviewed')})",
        key="verif_select_stg9"
    )
    
    p_data = df[df["work_id"] == selected_work_id].iloc[0]
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # DISPLAY ALL 7 REQUIRED INVESTIGATION PANELS FOR FLAGGED PROJECT
    # -------------------------------------------------------------------------
    st.subheader(f"🔍 Flagged Project Investigation Workspace: Work ID #{selected_work_id}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. 📌 Project Details",
        "2. 💰 Financial Anomaly",
        "3. ⏱️ Temporal Anomaly",
        "4. 🏪 Vendor Anomaly",
        "5. 📄 NLP Similarity",
        "6. 📈 Historical Benchmark",
        "7. 🤖 Isolation Forest Score"
    ])
    
    with tab1:
        st.markdown("### 📌 Project Metadata & Administrative Info")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"### **Work ID**: `#{p_data['work_id']}`")
            st.write(f"**State**: {p_data['state']}")
            st.write(f"**Constituency**: {p_data['constituency']}")
            st.write(f"**MP Name**: {p_data['mp_name']}")
        with c_b:
            st.write(f"**Category**: {p_data['category']}")
            st.write(f"**Project Status**: {p_data['project_status']}")
            st.write(f"**Recommendation Year**: {p_data['recommendation_year']}")
            st.write(f"**AI Risk Priority**: `{p_data['risk_category']}` ({p_data['final_risk_score']:.2f}/100)")
        st.info(f"**Work Description**: {p_data['work_description']}")

    with tab2:
        st.markdown("### 💰 Financial Anomaly Breakdown")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Effective Amount", f"₹{p_data['effective_amount']:,.2f}")
        f2.metric("Cohort Mean (μ)", f"₹{p_data['group_mean']:,.2f}")
        f3.metric("Cohort Median (P50)", f"₹{p_data['group_median']:,.2f}")
        f4.metric("Cost Deviation (Z-Score)", f"+{p_data['cost_z_score']:.2f} σ")
        st.write(f"**Cohort Percentile Rank**: {p_data['percentile']:.1f}% | **Financial Component Score**: {p_data['financial_score']:.1f}/100")

    with tab3:
        st.markdown("### ⏱️ Temporal Anomaly Breakdown")
        t1, t2, t3 = st.columns(3)
        t1.write(f"**Recommendation Date**: {p_data['recommendation_date']}")
        t2.write(f"**Completion Date**: {p_data['completion_date']}")
        t3.write(f"**Duration**: {p_data['duration_days']} days")
        st.write(f"**Impossible Date Sequence Flag**: {'YES (Negative Duration)' if p_data['is_impossible_date_sequence']==1 else 'No'}")
        st.write(f"**Temporal Component Score**: {p_data['temporal_score']:.1f}/100")

    with tab4:
        st.markdown("### 🏪 Vendor Anomaly Breakdown")
        v1, v2, v3 = st.columns(3)
        v1.metric("Constituency Max Vendor Share", f"{p_data['constituency_max_vendor_share']*100:.1f}%")
        v2.metric("Pending Payment Ratio", f"{p_data['constituency_pending_payment_ratio']*100:.1f}%")
        v3.metric("Vendor Component Score", f"{p_data['vendor_score']:.1f}/100")

    with tab5:
        st.markdown("### 📄 NLP Text Similarity Results")
        st.write(f"**Nearest Cosine Similarity Match**: {p_data['nearest_similarity_score']*100:.1f}%")
        st.write(f"**Similar Project Candidates Count**: {p_data['similar_project_count']}")

    with tab6:
        st.markdown("### 📈 Historical Expenditure Benchmark")
        fig_bm_tab = io.Figure()
        fig_bm_tab.add_trace(io.Bar(name="Project Value", x=["Expenditure"], y=[p_data['effective_amount']], marker_color="#EF4444"))
        fig_bm_tab.add_trace(io.Bar(name="Cohort Median (P50)", x=["Expenditure"], y=[p_data['group_median']], marker_color="#10B981"))
        fig_bm_tab.add_trace(io.Bar(name="Cohort Mean (μ)", x=["Expenditure"], y=[p_data['group_mean']], marker_color="#3B82F6"))
        fig_bm_tab.add_trace(io.Bar(name="Cohort 99th Percentile (P99)", x=["Expenditure"], y=[p_data['group_p99']], marker_color="#F59E0B"))
        fig_bm_tab.update_layout(barmode="group", title=f"Expenditure Comparison for Work ID #{selected_work_id}")
        st.plotly_chart(fig_bm_tab, width="stretch")

    with tab7:
        st.markdown("### 🤖 Isolation Forest Machine Learning Score")
        st.metric("Isolation Forest Anomaly Score", f"{p_data['isolation_forest_score']:.1f} / 100")
        st.write(f"**Model Anomaly Flag**: {'ANOMALY (-1)' if p_data['isolation_forest_score'] >= 55.0 else 'NORMAL (1)'}")

    st.divider()
    
    # -------------------------------------------------------------------------
    # AUDITOR VERIFICATION FORM & ACTION SELECTION
    # -------------------------------------------------------------------------
    st.subheader(f"🛡️ Record Official Auditor Verification Decision for Work ID #{selected_work_id}")
    st.info("🔒 **Immutability Guarantee**: AI model predictions are strictly read-only. Your auditor decision will be recorded separately in the audit trail log (`data/audit/review_log.csv`).")
    
    av_col1, av_col2 = st.columns(2)
    
    with av_col1:
        st.markdown(f"**Auditing Work ID**: `#{p_data['work_id']}`")
        st.write(f"**Current Recorded Status**: `{p_data['verification_status']}`")
        
        reviewer_action = st.radio(
            "Select Official Reviewer Action:",
            [
                "Verified Normal",
                "Needs Further Review",
                "Evidence Insufficient",
                "Manual Investigation Required"
            ],
            index=0
        )
        
        reviewer_id = st.text_input("Reviewer Officer Name / Badge ID:", value="AUDITOR_OFFICER_01")
        reviewer_notes = st.text_area("Auditor Investigation Notes & Justification:", placeholder="Enter findings from manual document audit, invoice verification, site inspection...")
        
        if st.button("💾 Submit & Record Official Audit Decision", type="primary"):
            new_log = append_to_audit_log(
                work_id=selected_work_id,
                review_status=reviewer_action,
                reviewer_notes=reviewer_notes,
                reviewer_id=reviewer_id,
                ai_score=p_data["final_risk_score"],
                ai_category=p_data["risk_category"]
            )
            st.success(f"Successfully recorded audit decision '{reviewer_action}' for Work ID #{selected_work_id} under Log ID `{new_log['log_id']}`!")
            st.rerun()

    with av_col2:
        st.markdown("### 📋 Complete Audit Trail History")
        audit_history_df = load_audit_trail_log()
        
        filter_this_work = st.checkbox(f"Filter History for Work ID #{selected_work_id} Only", value=False, key="chk_filter_work_id")
        if filter_this_work and len(audit_history_df) > 0 and "work_id" in audit_history_df.columns:
            audit_history_df = audit_history_df[audit_history_df["work_id"].astype(str) == str(selected_work_id)]
            
        if len(audit_history_df) > 0:
            st.dataframe(
                audit_history_df.sort_values("review_timestamp", ascending=False),
                width="stretch",
                hide_index=True
            )
        else:
            st.write("No audit log entries recorded yet for this view.")

# -----------------------------------------------------------------------------
# 7. NEW WORK SCREENING SECTION
# -----------------------------------------------------------------------------
elif nav_selection == "🆕 New Work Screening":
    st.header("🆕 New Work Screening")
    st.markdown("Screen newly proposed MPLADS works against historical project datasets, TF-IDF text similarity patterns, and the pre-trained Isolation Forest machine learning model.")
    
    st.info("ℹ️ **Notice**: Screening results provide decision support for administrative review. Final approval or rejection rests with authorized officials.")
    
    st.subheader("Input New Work Parameters")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        in_desc = st.text_area(
            "Work Description *",
            value="Construction of Community Hall and Library Building",
            height=100,
            help="Detailed description of proposed MPLADS work"
        )
        
        all_categories = sorted([str(c) for c in df["category"].dropna().unique() if str(c).strip()])
        in_category = st.selectbox("Category *", options=all_categories, index=0)
        
        in_amount = st.number_input(
            "Recommended Amount (₹) *",
            min_value=1000.0,
            step=50000.0,
            value=2500000.0,
            format="%.2f"
        )
        
        in_sanctioned_amount = st.number_input(
            "Sanctioned Amount (₹) * (Required for Isolation Forest Ratio)",
            min_value=0.0,
            step=50000.0,
            value=2500000.0,
            format="%.2f",
            help="Actual sanctioned amount. Used by Isolation Forest to compute sanction-to-recommendation ratio."
        )
        
        in_mp_name = st.text_input("MP Name", value="Shri Anupam Sharma")
        in_constituency = st.text_input("Constituency", value="New Delhi")
        
    with col_in2:
        all_states = sorted([str(s) for s in df["state"].dropna().unique() if str(s).strip()])
        in_state = st.selectbox("State *", options=all_states, index=0)
        in_house = st.selectbox("House", options=["Lok Sabha", "Rajya Sabha"])
        in_ida = st.text_input("Implementing Agency / IDA", value="Public Works Department (PWD)")
        
        today_date = datetime.today().date()
        in_date = st.date_input("Recommendation Date *", value=today_date)
        in_sanction_date = st.date_input("Sanction Date * (Required for Isolation Forest Sequence)", value=today_date)
        in_completion_date = st.date_input("Target / Expected Completion Date *", value=today_date + timedelta(days=180))

    st.divider()
    
    screen_btn = st.button("Screen New Work", type="primary")
    
    if screen_btn:
        st.session_state["screening_data"] = {
            "description": in_desc,
            "category": in_category,
            "amount": in_amount,
            "sanctioned_amount": in_sanctioned_amount,
            "mp_name": in_mp_name,
            "constituency": in_constituency,
            "state": in_state,
            "house": in_house,
            "ida": in_ida,
            "recommendation_date": in_date,
            "sanction_date": in_sanction_date,
            "completion_date": in_completion_date
        }
        
    if "screening_data" in st.session_state:
        scr_data = st.session_state["screening_data"]
        
        work_description = scr_data["description"]
        category = scr_data["category"]
        recommended_amount = float(scr_data["amount"])
        sanctioned_amount = float(scr_data.get("sanctioned_amount", recommended_amount))
        state = scr_data["state"]
        constituency = scr_data["constituency"]
        
        rec_date = scr_data.get("recommendation_date", today_date)
        sanc_date = scr_data.get("sanction_date", today_date)
        comp_date = scr_data.get("completion_date", today_date + timedelta(days=180))
        
        # 1. Cost Comparison & Baseline Aggregation
        cat_df = df[df["category"].astype(str).str.lower() == category.lower()]
        if len(cat_df) == 0:
            cat_df = df
            
        hist_median = float(cat_df["effective_amount"].median()) if len(cat_df) > 0 else 1000000.0
        hist_mean = float(cat_df["effective_amount"].mean()) if len(cat_df) > 0 else 1000000.0
        hist_std = float(cat_df["effective_amount"].std()) if len(cat_df) > 1 and cat_df["effective_amount"].std() > 0 else 1.0
        
        cost_diff_pct = ((recommended_amount - hist_median) / hist_median * 100.0) if hist_median > 0 else 0.0
        num_comparable = len(cat_df)
        
        # 2. NLP Text Similarity
        desc_tuple = tuple(df["work_description"].fillna("").tolist())
        vec, X_tfidf_hist = build_tfidf_index_cached(desc_tuple)
        
        clean_in = clean_text_simple(work_description)
        X_new = vec.transform([clean_in])
        sim_scores = cosine_similarity(X_new, X_tfidf_hist).ravel()
        
        top_5_indices = np.argsort(sim_scores)[::-1][:5]
        top_5_rows = df.iloc[top_5_indices].copy()
        
        top_5_display = pd.DataFrame({
            "Work ID": top_5_rows["work_id"].values,
            "Description": top_5_rows["work_description"].values,
            "Amount (₹)": [f"₹{amt:,.2f}" for amt in top_5_rows["effective_amount"].values],
            "Similarity %": [f"{sim*100.0:.1f}%" for sim in sim_scores[top_5_indices]]
        })
        
        max_sim = float(sim_scores.max()) if len(sim_scores) > 0 else 0.0
        sim_count_85 = int((sim_scores >= 0.85).sum())
        
        # 3. Isolation Forest Feature Extraction & Anomaly Scoring
        clf, scaler = load_isolation_model()
        
        state_amounts = df[df["state"] == state]["effective_amount"]
        state_percentile = float((state_amounts < recommended_amount).mean() * 100.0) if len(state_amounts) > 0 else float((df["effective_amount"] < recommended_amount).mean() * 100.0)
        
        const_amounts = df[df["constituency"] == constituency]["effective_amount"]
        const_percentile = float((const_amounts < recommended_amount).mean() * 100.0) if len(const_amounts) > 0 else state_percentile
        
        cost_dev = (recommended_amount - hist_mean) / hist_std if hist_std > 0 else 0.0
        rec_sanc_ratio = (sanctioned_amount / recommended_amount) if recommended_amount > 0 else 1.0
        
        # Temporal anomaly flags
        is_impossible_date = 1 if (sanc_date < rec_date or comp_date < sanc_date) else 0
        is_negative_dur = 1 if (comp_date < sanc_date) else 0
        covid_start = datetime(2020, 3, 1).date()
        covid_end = datetime(2021, 12, 31).date()
        is_covid = 1 if (covid_start <= rec_date <= covid_end) else 0
        
        # Vendor cohort defaults
        const_df = df[df["constituency"] == constituency]
        v_share = float(const_df["constituency_max_vendor_share"].median()) if "constituency_max_vendor_share" in const_df.columns and len(const_df) > 0 else 0.35
        v_tx = float(const_df["constituency_avg_vendor_transaction"].median()) if "constituency_avg_vendor_transaction" in const_df.columns and len(const_df) > 0 else 500000.0
        v_cnt = 5.0
        v_rep = 0.5
        v_pend = float(const_df["constituency_pending_payment_ratio"].median()) if "constituency_pending_payment_ratio" in const_df.columns and len(const_df) > 0 else 0.1
        
        iso_prediction_label = "NORMAL (1)"
        iso_prediction_code = 1
        
        if clf is not None and scaler is not None:
            feat_val_map = {
                "effective_amount": recommended_amount,
                "cost_deviation": float(cost_dev),
                "state_cost_percentile": float(state_percentile),
                "constituency_cost_percentile": float(const_percentile),
                "release_ratio": 1.0,
                "recommendation_sanction_ratio": float(rec_sanc_ratio),
                "is_impossible_date_sequence": int(is_impossible_date),
                "is_negative_duration": int(is_negative_dur),
                "is_covid_era": int(is_covid),
                "year_over_year_change": 0.0,
                "description_length": len(work_description),
                "word_count": len(work_description.split()),
                "tfidf_mean": float(X_new.mean()),
                "tfidf_max": float(X_new.max()),
                "nearest_similarity_score": float(max_sim),
                "similar_project_count": int(sim_count_85),
                "constituency_max_vendor_share": v_share,
                "constituency_avg_vendor_transaction": v_tx,
                "constituency_vendor_count": v_cnt,
                "constituency_avg_vendor_repeat_rate": v_rep,
                "constituency_pending_payment_ratio": v_pend
            }
            
            if hasattr(scaler, "feature_names_in_"):
                expected_cols = list(scaler.feature_names_in_)
                X_feat_df = pd.DataFrame([{col: feat_val_map.get(col, 0.0) for col in expected_cols}], columns=expected_cols)
            else:
                X_feat_df = pd.DataFrame([feat_val_map])
                
            X_scaled = scaler.transform(X_feat_df)
            raw_dec = float(clf.decision_function(X_scaled)[0])
            inv_dec = -raw_dec
            iso_score_100 = float(np.clip((inv_dec - (-0.20)) / (0.30 - (-0.20)) * 100.0, 0.0, 100.0))
            
            raw_pred = clf.predict(X_scaled)[0]
            if raw_pred == -1 or iso_score_100 >= 55.0:
                iso_prediction_label = "ANOMALOUS (-1)"
                iso_prediction_code = -1
            else:
                iso_prediction_label = "NORMAL (1)"
                iso_prediction_code = 1
        else:
            iso_score_100 = 50.0
            
        # 4. Screening Result Score & Status
        fin_cost_score = np.clip(state_percentile * 0.6 + min(abs(cost_dev) * 20.0, 100.0) * 0.4, 0.0, 100.0)
        nlp_sim_score = max_sim * 100.0
        
        screening_score = round(0.40 * fin_cost_score + 0.35 * iso_score_100 + 0.25 * nlp_sim_score, 1)
        screening_score = float(np.clip(screening_score, 0.0, 100.0))
        
        if screening_score < 40.0:
            status_label = "LOW — Normal pattern"
            badge_color = "#D1FAE5"
            text_color = "#065F46"
            status_level = "LOW"
        elif screening_score <= 69.99:
            status_label = "MEDIUM — Review recommended"
            badge_color = "#FEF3C7"
            text_color = "#92400E"
            status_level = "MEDIUM"
        else:
            status_label = "HIGH — Priority for verification"
            badge_color = "#FEE2E2"
            text_color = "#991B1B"
            status_level = "HIGH"
            
        # 5. Explanations (2-6 evidence reasons)
        reasons = []
        if is_impossible_date:
            reasons.append("⚠️ **CRITICAL TEMPORAL ANOMALY**: Impossible date sequence detected! Sanction date precedes recommendation date or target completion date precedes sanction date.")
            
        if abs(rec_sanc_ratio - 1.0) > 0.25:
            reasons.append(f"• Significant variance between Sanctioned Amount (₹{sanctioned_amount:,.2f}) and Recommended Amount (₹{recommended_amount:,.2f}) (Ratio: {rec_sanc_ratio:.2f}).")
            
        if cost_diff_pct > 25.0:
            reasons.append(f"• Proposed amount (₹{recommended_amount:,.2f}) is significantly higher ({cost_diff_pct:+.1f}%) than the historical median (₹{hist_median:,.2f}) for '{category}' works.")
        elif cost_diff_pct < -25.0:
            reasons.append(f"• Proposed amount (₹{recommended_amount:,.2f}) is lower ({cost_diff_pct:.1f}%) than the historical median (₹{hist_median:,.2f}) for '{category}' works.")
        else:
            reasons.append(f"• Proposed amount (₹{recommended_amount:,.2f}) is aligned with the historical median (₹{hist_median:,.2f}) for '{category}' works.")
            
        if max_sim >= 0.85:
            reasons.append(f"• Work description is highly similar ({max_sim*100.0:.1f}% cosine match) to {sim_count_85} existing historical project(s).")
        elif max_sim >= 0.60:
            reasons.append(f"• Work description exhibits moderate text similarity ({max_sim*100.0:.1f}% match) with existing historical projects.")
        else:
            reasons.append(f"• Work description exhibits a standard unique text pattern with low description overlap.")
            
        if iso_score_100 >= 55.0:
            reasons.append(f"• **Isolation Forest ML Model Flag**: Multi-dimensional feature pattern isolated as statistically rare (Isolation Score: {iso_score_100:.1f}/100 | Label: {iso_prediction_label}).")
        else:
            reasons.append(f"• **Isolation Forest ML Model**: Feature pattern is consistent with normal historical project clusters (Isolation Score: {iso_score_100:.1f}/100 | Label: {iso_prediction_label}).")
            
        if status_level in ["MEDIUM", "HIGH"]:
            reasons.append("• Overall risk evaluation flagged this proposed work — human verification recommended.")
            
        # Render Output Section
        st.subheader("Screening Result Telemetry")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="🌲 Isolation Forest Model Score",
                value=f"{iso_score_100:.1f} / 100",
                delta="ANOMALY DETECTED" if iso_prediction_code == -1 else "NORMAL PATTERN",
                delta_color="inverse" if iso_prediction_code == -1 else "normal"
            )
        with m_col2:
            st.metric(
                label="⚡ Composite AI Risk Score",
                value=f"{screening_score:.1f} / 100",
                delta=f"RISK LEVEL: {status_level}"
            )
        with m_col3:
            st.metric(
                label="💰 Sanction / Recommendation Ratio",
                value=f"{rec_sanc_ratio * 100.0:.1f}%",
                delta=f"{'EXCESSIVE' if rec_sanc_ratio > 1.25 else 'STANDARD'}"
            )

        st.markdown(
            f'<div style="margin-top: 1.0rem; background-color: {badge_color}; color: {text_color}; padding: 0.8rem 1.5rem; border-radius: 8px; font-weight: 700; font-size: 1.1rem; display: inline-block;">'
            f'OVERALL STATUS: {status_label} | ISOLATION FOREST CLASSIFICATION: {iso_prediction_label}</div>',
            unsafe_allow_html=True
        )
            
        st.divider()
        
        st.subheader("Why was it flagged?")
        for reason in reasons:
            st.markdown(reason)
            
        with st.expander("🔍 View Isolation Forest Feature Vector & Model Parameters"):
            st.markdown("Below are the 19 numerical feature values passed to `sklearn.ensemble.IsolationForest`:")
            st.json({
                "effective_amount": recommended_amount,
                "sanctioned_amount": sanctioned_amount,
                "recommendation_sanction_ratio": round(rec_sanc_ratio, 3),
                "is_impossible_date_sequence": is_impossible_date,
                "is_negative_duration": is_negative_dur,
                "is_covid_era": is_covid,
                "cost_deviation_zscore": round(cost_dev, 2),
                "state_cost_percentile": round(state_percentile, 1),
                "constituency_cost_percentile": round(const_percentile, 1),
                "nlp_description_length": len(work_description),
                "nlp_word_count": len(work_description.split()),
                "nearest_similarity_score": round(max_sim, 3),
                "similar_project_count_85pct": sim_count_85,
                "isolation_tree_anomaly_score": round(iso_score_100, 1),
                "isolation_forest_label": iso_prediction_label
            })

        st.divider()
        
        st.subheader("Similar Historical Projects")
        st.dataframe(top_5_display, width="stretch", hide_index=True)
        
        st.divider()
        
        st.subheader("Cost Comparison")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Proposed Amount", f"₹{recommended_amount:,.2f}")
        c2.metric("Historical Median Amount", f"₹{hist_median:,.2f}")
        c3.metric("Cost Difference %", f"{cost_diff_pct:+.1f}%")
        c4.metric("Comparable Projects", f"{num_comparable:,}")
        
        # Visual Cost Bar Chart
        fig_cost = io.Figure()
        fig_cost.add_trace(io.Bar(name="Proposed Amount", x=["Cost Benchmark"], y=[recommended_amount], marker_color="#EF4444" if cost_diff_pct > 25 else "#3B82F6"))
        fig_cost.add_trace(io.Bar(name="Historical Median", x=["Cost Benchmark"], y=[hist_median], marker_color="#10B981"))
        fig_cost.add_trace(io.Bar(name="Historical Mean", x=["Cost Benchmark"], y=[hist_mean], marker_color="#F59E0B"))
        fig_cost.update_layout(barmode="group", title=f"Proposed Amount vs Historical '{category}' Category Baseline", yaxis_title="Amount (₹)")
        st.plotly_chart(fig_cost, width="stretch")

# -----------------------------------------------------------------------------
# 8. RISK MAP SECTION
# -----------------------------------------------------------------------------
elif nav_selection == "🗺️ Risk Map":
    st.header("🗺️ MPLADS Risk Map")
    st.markdown("AI-based visualization of unusual project patterns across administrative areas")
    st.info("ℹ️ **Operational Policy**: Area Risk Scores represent aggregated multi-factor statistical deviations across projects in administrative regions. Designations indicate administrative verification priority, not guilt or fraud.")
    
    # 1. Search & Filter Controls
    with st.expander("🔻 Search & Filter Options", expanded=True):
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            rm_states = st.multiselect("Filter by State:", sorted(df["state"].dropna().unique().tolist()), key="rm_state_filter")
            rm_cats = st.multiselect("Filter by Category:", sorted(df["category"].dropna().unique().tolist()), key="rm_cat_filter")
        with f_c2:
            const_options = sorted(df["constituency"].dropna().unique().tolist())
            if rm_states:
                const_options = sorted(df[df["state"].isin(rm_states)]["constituency"].dropna().unique().tolist())
            rm_consts = st.multiselect("Filter by Constituency:", const_options, key="rm_const_filter")
            rm_risks = st.multiselect("Filter by Risk Level:", ["HIGH", "MEDIUM", "LOW"], key="rm_risk_filter")
        with f_c3:
            rm_status = st.multiselect("Filter by Project Status:", sorted(df["project_status"].dropna().unique().tolist()), key="rm_status_filter")
            rm_min_works = st.number_input("Minimum Works Threshold:", min_value=1, max_value=50, value=5, step=1, help="Areas with total works below this threshold are marked INSUFFICIENT DATA.")
        with f_c4:
            st.write(" ")
            st.write(" ")
            if st.button("🔄 Reset Filters", type="secondary"):
                for k in ["rm_state_filter", "rm_cat_filter", "rm_const_filter", "rm_risk_filter", "rm_status_filter"]:
                    if k in st.session_state:
                        st.session_state[k] = []
                st.rerun()

    # Apply Filters to Master Dataset
    rm_df = df.copy()
    if rm_states:
        rm_df = rm_df[rm_df["state"].isin(rm_states)]
    if rm_consts:
        rm_df = rm_df[rm_df["constituency"].isin(rm_consts)]
    if rm_cats:
        rm_df = rm_df[rm_df["category"].isin(rm_cats)]
    if rm_risks:
        rm_df = rm_df[rm_df["risk_category"].isin(rm_risks)]
    if rm_status:
        rm_df = rm_df[rm_df["project_status"].isin(rm_status)]
        
    if len(rm_df) == 0:
        st.warning("No projects match the selected filters. Please adjust filter criteria.")
        st.stop()

    # 2. Risk Level Legend Bar (Matching Screenshot)
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem 1.2rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <span style="font-weight: 700; color: #475569; font-size: 0.95rem;">Risk Level:</span>
        <span style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.9rem; font-weight: 600; color: #065F46;">🟢 Low (0–39)</span>
        <span style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.9rem; font-weight: 600; color: #D97706;">🟡 Moderate (40–54)</span>
        <span style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.9rem; font-weight: 600; color: #EA580C;">🟠 Elevated (55–69)</span>
        <span style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.9rem; font-weight: 600; color: #DC2626;">🔴 High (70–100)</span>
    </div>
    """, unsafe_allow_html=True)

    # 3. State & Constituency Aggregations
    state_map_agg = rm_df.groupby("state").agg(
        total_works=("work_id", "count"),
        high_risk_works=("risk_category", lambda s: (s == "HIGH").sum()),
        med_risk_works=("risk_category", lambda s: (s == "MEDIUM").sum()),
        low_risk_works=("risk_category", lambda s: (s == "LOW").sum()),
        median_score=("final_risk_score", "median")
    ).reset_index()

    state_map_agg["area_risk_score"] = state_map_agg["median_score"].round(1)
    state_map_agg["high_risk_pct"] = np.round((state_map_agg["high_risk_works"] / state_map_agg["total_works"]) * 100.0, 2)

    const_agg = rm_df.groupby(["state", "constituency"]).agg(
        total_works=("work_id", "count"),
        high_risk_works=("risk_category", lambda s: (s == "HIGH").sum()),
        medium_risk_works=("risk_category", lambda s: (s == "MEDIUM").sum()),
        low_risk_works=("risk_category", lambda s: (s == "LOW").sum()),
        median_risk_score=("final_risk_score", "median"),
        high_z_count=("cost_z_score", lambda s: (s > 2.5).sum()),
        high_sim_count=("nearest_similarity_score", lambda s: (s >= 0.85).sum()),
        high_iso_count=("isolation_forest_score", lambda s: (s >= 55.0).sum()),
        imp_date_count=("is_impossible_date_sequence", lambda s: (s == 1).sum())
    ).reset_index()

    const_agg["high_risk_pct"] = np.round((const_agg["high_risk_works"] / const_agg["total_works"]) * 100.0, 2)
    const_agg["area_risk_score"] = const_agg["median_risk_score"].round(1)

    def assign_area_status(row, min_works_val):
        if row["total_works"] < min_works_val:
            return "INSUFFICIENT DATA"
        elif row["area_risk_score"] >= 50.0 or row["high_risk_pct"] >= 10.0 or row["high_risk_works"] >= 10:
            return "HIGH"
        elif row["area_risk_score"] >= 40.0 or row["high_risk_pct"] >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"

    const_agg["area_status"] = const_agg.apply(lambda r: assign_area_status(r, rm_min_works), axis=1)

    # Summary Cards Row
    total_areas = len(const_agg)
    high_areas = (const_agg["area_status"] == "HIGH").sum()
    med_areas = (const_agg["area_status"] == "MEDIUM").sum()
    low_areas = (const_agg["area_status"] == "LOW").sum()
    total_works_cnt = len(rm_df)
    high_works_cnt = (rm_df["risk_category"] == "HIGH").sum()

    s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)
    s_col1.metric("Total Areas", f"{total_areas:,}")
    s_col2.metric("High-Risk Areas", f"{high_areas:,}", delta="Verification Priority", delta_color="inverse")
    s_col3.metric("Medium-Risk Areas", f"{med_areas:,}", delta_color="off")
    s_col4.metric("Low-Risk Areas", f"{low_areas:,}", delta_color="normal")
    s_col5.metric("Total Works Analyzed", f"{total_works_cnt:,}")
    s_col6.metric("High-Risk Works", f"{high_works_cnt:,}")

    st.divider()

    # 4. Main Split View: Map (Left) vs State Cards & Top Risk States (Right)
    map_col, info_col = st.columns([2.2, 1])

    with map_col:
        st.subheader("Administrative India Risk Map")
        geojson_path = os.path.join("data", "india_states.geojson")
        if os.path.exists(geojson_path):
            with open(geojson_path, "r", encoding="utf-8") as f:
                india_geojson = json.load(f)

            map_data = state_map_agg.copy()
            name_mapping = {
                "The Dadra And Nagar Haveli And Daman And Diu": "Dadra and Nagar Haveli",
                "Andaman And Nicobar Islands": "Andaman and Nicobar Islands",
                "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
                "Jammu And Kashmir": "Jammu and Kashmir",
                "Jammu & Kashmir": "Jammu and Kashmir"
            }
            map_data["geo_state"] = map_data["state"].replace(name_mapping)

            fig_map = px.choropleth(
                map_data,
                geojson=india_geojson,
                locations="geo_state",
                featureidkey="properties.st_nm",
                color="area_risk_score",
                color_continuous_scale=[
                    [0.0, "#10B981"],    # Low - Green
                    [0.4, "#F59E0B"],    # Moderate - Yellow
                    [0.65, "#F97316"],   # Elevated - Orange
                    [1.0, "#EF4444"]     # High - Red
                ],
                range_color=[20, 80],
                hover_name="state",
                hover_data={
                    "geo_state": False,
                    "area_risk_score": ":.1f",
                    "total_works": ":,",
                    "high_risk_works": ":,",
                    "high_risk_pct": ":.2f%"
                },
                labels={
                    "area_risk_score": "Area Risk Score",
                    "total_works": "Total Works",
                    "high_risk_works": "High Risk Works",
                    "high_risk_pct": "High Risk %"
                }
            )
            fig_map.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="rgba(0,0,0,0)"
            )
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                height=720,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(
                    title="Risk Score",
                    thicknessmode="pixels", thickness=14,
                    lenmode="pixels", len=280,
                    yanchor="middle", y=0.5
                )
            )
            
            map_event = st.plotly_chart(
                fig_map,
                width="stretch",
                on_select="rerun",
                selection_mode="points"
            )

            # Detect touch / click event on state polygon
            if map_event and "selection" in map_event and "points" in map_event["selection"] and len(map_event["selection"]["points"]) > 0:
                pt = map_event["selection"]["points"][0]
                clicked_geo_name = pt.get("location") or pt.get("hovertext")
                if clicked_geo_name:
                    rev_map = {
                        "Dadra and Nagar Haveli": "The Dadra And Nagar Haveli And Daman And Diu",
                        "Andaman and Nicobar Islands": "Andaman And Nicobar Islands",
                        "Jammu and Kashmir": "Jammu And Kashmir"
                    }
                    target_state_name = rev_map.get(clicked_geo_name, clicked_geo_name)
                    if target_state_name in state_map_agg["state"].values and st.session_state.get("inspect_state_select") != target_state_name:
                        st.session_state["inspect_state_select"] = target_state_name
                        st.rerun()
        else:
            st.warning("India GeoJSON missing. Rendering bar overview.")
            fig_fallback = px.bar(state_map_agg, x="state", y="area_risk_score", color="area_risk_score")
            st.plotly_chart(fig_fallback, width="stretch")

    with info_col:
        # Top Card: State Statistics Inspector
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.0rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">📍</div>
            <div style="font-weight: 700; color: #1E293B; font-size: 1.05rem;">State Statistics Inspector</div>
            <div style="font-size: 0.85rem; color: #64748B;">Select or hover a state on the map to view statistics</div>
        </div>
        """, unsafe_allow_html=True)

        selected_state_inspect = st.selectbox(
            "Select State to Inspect:",
            options=sorted(state_map_agg["state"].tolist()),
            key="inspect_state_select"
        )

        st_row = state_map_agg[state_map_agg["state"] == selected_state_inspect].iloc[0]

        st.markdown(f"#### **{st_row['state']}**")
        st1, st2 = st.columns(2)
        st1.metric("Area Risk Score", f"{st_row['area_risk_score']:.1f} / 100")
        st2.metric("High-Risk %", f"{st_row['high_risk_pct']:.2f}%")

        st.write(f"Total Works: **{st_row['total_works']:,}** | High Risk: **{st_row['high_risk_works']:,}** | Med Risk: **{st_row['med_risk_works']:,}**")

        st.divider()

        # Bottom Card: TOP RISK STATES List (Matching Screenshot)
        st.markdown('<div style="font-weight: 700; color: #475569; font-size: 0.95rem; margin-bottom: 0.8rem;">⚠️ TOP RISK STATES</div>', unsafe_allow_html=True)

        top_states_list = state_map_agg.sort_values("total_works", ascending=False).head(7)

        for _, ts in top_states_list.iterrows():
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.8rem; border-bottom: 1px solid #F1F5F9; font-size: 0.9rem;">
                <span style="font-weight: 600; color: #334155;">{ts['state']}</span>
                <span style="background-color: #F1F5F9; color: #475569; padding: 0.15rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.8rem;">{ts['total_works']:,}</span>
            </div>
            """, unsafe_allow_html=True)

    # 5. Separate Detailed Information Panels (State, Constituency & Works)
    st.divider()
    st.subheader(f"📊 Separate Audit Inspection Panels for: {selected_state_inspect}")
    
    det_tab1, det_tab2, det_tab3 = st.tabs([
        f"🏛️ State Audit Telemetry ({selected_state_inspect})",
        f"📌 Constituency Level Deep-Dive ({selected_state_inspect})",
        "📋 Top Flagged Works & Evidence Deep-Dive"
    ])

    state_works = rm_df[rm_df["state"] == selected_state_inspect]
    state_const_df = const_agg[const_agg["state"] == selected_state_inspect].sort_values("area_risk_score", ascending=False)

    with det_tab1:
        st.markdown(f"### 🏛️ State Overview & Telemetry: `{selected_state_inspect}`")
        
        st_h_cnt = (state_works["risk_category"] == "HIGH").sum()
        st_m_cnt = (state_works["risk_category"] == "MEDIUM").sum()
        st_l_cnt = (state_works["risk_category"] == "LOW").sum()
        st_tot = len(state_works)
        st_pct = (st_h_cnt / st_tot * 100.0) if st_tot > 0 else 0.0
        st_amt = state_works["effective_amount"].sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("State Median Risk Score", f"{st_row['area_risk_score']:.1f} / 100")
        m2.metric("Total Works", f"{st_tot:,}")
        m3.metric("High-Risk Works", f"{st_h_cnt:,}")
        m4.metric("High-Risk %", f"{st_pct:.2f}%")
        m5.metric("Total Effective Amount", f"₹{st_amt:,.2f}")

        st.markdown(f"Work Distribution in {selected_state_inspect}: **{st_h_cnt}** High Risk | **{st_m_cnt}** Medium Risk | **{st_l_cnt}** Low Risk")

        st.markdown("#### 🚨 State-Level Main Risk Signals")
        st_z = (state_works["cost_z_score"] > 2.5).sum()
        st_sim = (state_works["nearest_similarity_score"] >= 0.85).sum()
        st_iso = (state_works["isolation_forest_score"] >= 55.0).sum()
        st_imp = (state_works["is_impossible_date_sequence"] == 1).sum()

        st_signals = []
        if st_z > 0:
            st_signals.append(f"• **{st_z}** project(s) exhibit unusually high cost compared with historical benchmarks (Z-Score > 2.5).")
        if st_sim > 0:
            st_signals.append(f"• **{st_sim}** project(s) show high description text similarity (>= 85% cosine match) with existing works.")
        if st_iso > 0:
            st_signals.append(f"• **{st_iso}** project(s) exhibit multi-dimensional feature isolation flagged by Isolation Forest.")
        if st_imp > 0:
            st_signals.append(f"• **{st_imp}** project(s) contain invalid date chronology sequences.")

        if st_signals:
            for sig in st_signals:
                st.markdown(sig)
        else:
            st.markdown("• All projects in this state operate within standard baseline bounds.")

        st.markdown("#### 📋 Constituency Risk Ranking within State")
        st.dataframe(
            state_const_df[["constituency", "total_works", "high_risk_works", "high_risk_pct", "area_risk_score", "area_status"]].rename(columns={
                "constituency": "Constituency",
                "total_works": "Total Works",
                "high_risk_works": "High Risk Works",
                "high_risk_pct": "High Risk %",
                "area_risk_score": "Area Risk Score",
                "area_status": "Status"
            }),
            width="stretch",
            hide_index=True
        )

    with det_tab2:
        st.markdown(f"### 📌 Constituency Level Deep-Dive: `{selected_state_inspect}`")
        
        if len(state_const_df) == 0:
            st.warning("No constituencies found for this state under current filters.")
        else:
            selected_const_inspect = st.selectbox(
                f"Select Constituency in {selected_state_inspect} to Inspect:",
                options=state_const_df["constituency"].tolist(),
                key="const_inspect_tab_select"
            )

            c_data = state_const_df[state_const_df["constituency"] == selected_const_inspect].iloc[0]
            c_works = state_works[state_works["constituency"] == selected_const_inspect]

            st.markdown(f"#### Constituency Risk Summary: `{selected_const_inspect}`")

            ca1, ca2, ca3, ca4, ca5 = st.columns(5)

            if c_data["area_status"] == "INSUFFICIENT DATA":
                ca1.metric("Area Risk Score", "N/A")
                ca2.markdown('<div style="background-color: #E2E8F0; color: #475569; padding: 0.6rem; text-align: center; border-radius: 8px; font-weight: 700;">INSUFFICIENT DATA</div>', unsafe_allow_html=True)
            else:
                ca1.metric("Area Risk Score", f"{c_data['area_risk_score']:.0f} / 100")
                badge_bg = "#FEE2E2" if c_data["area_status"] == "HIGH" else ("#FEF3C7" if c_data["area_status"] == "MEDIUM" else "#D1FAE5")
                badge_fg = "#991B1B" if c_data["area_status"] == "HIGH" else ("#92400E" if c_data["area_status"] == "MEDIUM" else "#065F46")
                badge_txt = "HIGH PRIORITY FOR VERIFICATION" if c_data["area_status"] == "HIGH" else f"{c_data['area_status']} PRIORITY"
                ca2.markdown(f'<div style="background-color: {badge_bg}; color: {badge_fg}; padding: 0.6rem; text-align: center; border-radius: 8px; font-weight: 700;">{badge_txt}</div>', unsafe_allow_html=True)

            ca3.metric("Total Works", f"{c_data['total_works']:,}")
            ca4.metric("High-Risk Works", f"{c_data['high_risk_works']:,}")
            ca5.metric("High-Risk %", f"{c_data['high_risk_pct']:.2f}%")

            st.write(f"Constituency Breakdown: **{c_data['high_risk_works']}** High Risk | **{c_data['medium_risk_works']}** Medium Risk | **{c_data['low_risk_works']}** Low Risk")

            st.markdown("##### Constituency Risk Signals")
            c_sigs = []
            if c_data["high_z_count"] > 0:
                c_sigs.append(f"• **{c_data['high_z_count']}** project(s) exhibit unusually high cost compared with historical benchmarks.")
            if c_data["high_sim_count"] > 0:
                c_sigs.append(f"• **{c_data['high_sim_count']}** project(s) show high text similarity (>= 85%) with existing descriptions.")
            if c_data["high_iso_count"] > 0:
                c_sigs.append(f"• **{c_data['high_iso_count']}** project(s) exhibit unusual feature isolation patterns.")
            if c_data["imp_date_count"] > 0:
                c_sigs.append(f"• **{c_data['imp_date_count']}** project(s) contain invalid date chronology flags.")

            if c_sigs:
                for cs in c_sigs:
                    st.markdown(cs)
            else:
                st.markdown("• All projects in this constituency operate within standard baseline bounds.")

    with det_tab3:
        st.markdown(f"### 📋 Top Flagged Works in `{selected_state_inspect}`")
        st.markdown("Contributing projects sorted by risk score descending:")

        top_works_st = state_works.sort_values("final_risk_score", ascending=False).head(10).copy()

        top_works_display = pd.DataFrame({
            "Work ID": top_works_st["work_id"].values,
            "Constituency": top_works_st["constituency"].values,
            "Work Description": top_works_st["work_description"].values,
            "Category": top_works_st["category"].values,
            "Amount (₹)": [f"₹{amt:,.2f}" for amt in top_works_st["effective_amount"].values],
            "Risk Score": top_works_st["final_risk_score"].values,
            "Risk Level": top_works_st["risk_category"].values,
            "Main Reason": [str(exp).split("\n")[1] if "\n" in str(exp) else str(exp)[:100] for exp in top_works_st["explanation"].values]
        })

        st.dataframe(top_works_display, width="stretch", hide_index=True)

        if len(top_works_st) > 0:
            st.divider()
            st.markdown("#### 🔍 Project Evidence Deep-Dive Inspector")
            sel_w_id = st.selectbox(
                "Select Work ID to Inspect Full Evidence Vector:",
                options=top_works_st["work_id"].tolist(),
                format_func=lambda x: f"Work ID #{x} ({top_works_st[top_works_st['work_id']==x]['risk_category'].values[0]} Risk - Score: {top_works_st[top_works_st['work_id']==x]['final_risk_score'].values[0]:.2f})"
            )

            w_row = top_works_st[top_works_st["work_id"] == sel_w_id].iloc[0]

            w1, w2, w3 = st.columns(3)
            w1.write(f"**Work ID**: #{w_row['work_id']}")
            w1.write(f"**Constituency**: {w_row['constituency']}")
            w1.write(f"**MP Name**: {w_row['mp_name']}")

            w2.write(f"**Category**: {w_row['category']}")
            w2.write(f"**Effective Amount**: ₹{w_row['effective_amount']:,.2f}")
            w2.write(f"**Cost Z-Score**: +{w_row['cost_z_score']:.2f}")

            w3.metric("Final Risk Score", f"{w_row['final_risk_score']:.2f} / 100", delta=w_row['risk_category'])
            w3.write(f"**Isolation Forest Score**: {w_row['isolation_forest_score']:.1f}/100")

            st.info(f"**Work Description**: {w_row['work_description']}")
            st.text_area("Audit Explanation & Telemetry", value=w_row["explanation"], height=120, disabled=True)

    # 6. Overall Risk Distribution Charts
    st.divider()
    st.subheader("Risk Distribution Analytics")
    
    rd_c1, rd_c2 = st.columns(2)
    
    with rd_c1:
        area_cat_counts = const_agg["area_status"].value_counts().reset_index()
        area_cat_counts.columns = ["Status", "Count"]
        fig_area_dist = px.pie(
            area_cat_counts,
            names="Status",
            values="Count",
            color="Status",
            color_discrete_map={
                "HIGH": "#EF4444",
                "MEDIUM": "#F59E0B",
                "LOW": "#10B981",
                "INSUFFICIENT DATA": "#94A3B8"
            },
            title="Area Risk Level Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_area_dist, width="stretch")
        
    with rd_c2:
        work_cat_counts = rm_df["risk_category"].value_counts().reset_index()
        work_cat_counts.columns = ["Risk Level", "Count"]
        fig_work_dist = px.bar(
            work_cat_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
            title="Project Work Risk Distribution (Filtered Corpus)"
        )
        st.plotly_chart(fig_work_dist, width="stretch")



