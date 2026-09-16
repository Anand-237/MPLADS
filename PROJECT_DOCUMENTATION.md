# 🛡️ MPLADS AI Anomaly Monitoring & Human-in-the-Loop Audit System
## Complete System Documentation & Technical Blueprint

---

## 📌 Executive Summary

The **MPLADS AI Anomaly Monitoring & Human-in-the-Loop Audit System** is an end-to-end Machine Learning, Explainable AI (XAI), and administrative audit platform designed for oversight of the **Members of Parliament Local Area Development Scheme (MPLADS)**.

The system processes **127,263 official project records** across India, standardizes non-uniform dataset structures, extracts multi-domain features (Financial, Temporal, Vendor, and NLP Similarity), applies an **Isolation Forest** unsupervised anomaly detection algorithm, and computes a **Multi-Factor Risk Score (0–100)** to prioritize projects for human administrative audit.

> [!IMPORTANT]
> **Operational Compliance Policy**: The system identifies unusual statistical patterns and prioritizes projects for human administrative verification. It strictly **DOES NOT** automatically claim fraud, corruption, or illegal wrongdoing.

---

## 📊 Core Dataset Overview

The system ingests three primary data streams:

| Dataset | Dimensions | Key Attributes / Scope |
| :--- | :--- | :--- |
| **`data/raw/projects.csv`** | $127,263 \text{ rows} \times 14 \text{ cols}$ | Individual project recommendations, sanctioned values, recommendation & completion dates, work descriptions, MP IDs. |
| **`data/raw/work_by_type.csv`** | $127,263 \text{ rows} \times 15 \text{ cols}$ | Category breakdowns (Roads, Sanitation, Water, Education, Health), expenditure values, state/district tags. |
| **`data/raw/mp_summary.csv`** | $794 \text{ MPs} \times 10 \text{ cols}$ | MP-level metrics: Allocated Funds, Total Expenditure, Completed Works count, Utilization %, Completion Rate %. |

---

## 🏗️ System Architecture & Stack

```
 +---------------------------------------------------------------------------------------+
 |                                  OFFICIAL DATA SOURCES                                |
 |        data/raw/projects.csv  |  data/raw/work_by_type.csv  |  data/raw/mp_summary.csv |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
 +---------------------------------------------------------------------------------------+
 |                     STAGE 1 & 2: INGESTION, SCHEMA & CLEANING                          |
 |    - Strip currency symbols (₹, commas)  - Standardize column names                    |
 |    - Parse ISO dates (YYYY-MM-DD)        - Impute missing vendor / date values         |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
 +---------------------------------------------------------------------------------------+
 |                     STAGE 3A-3D: MULTI-DOMAIN FEATURE ENGINEERING                      |
 |  [Financial Features]    [Temporal Features]    [Vendor Features]    [NLP Similarity]  |
 |  - Sanction Variance     - Duration Days        - Max Share Ratio    - TF-IDF Vector   |
 |  - Peer Z-Score          - Date Sequence Flag   - Repeat Vendor Rate - Cosine Sim Mat   |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
 +---------------------------------------------------------------------------------------+
 |                     STAGE 4A-4C: MACHINE LEARNING & BASELINES                         |
 |  - 19-Feature Matrix Standard Scaling (StandardScaler)                                |
 |  - Isolation Forest Model (n_estimators=100, contamination=0.05)                      |
 |  - Statistical Baselines & Cohort Profiling (P50, μ, P99)                             |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
 +---------------------------------------------------------------------------------------+
 |                     STAGE 5: MULTI-FACTOR RISK SCORING ENGINE                         |
 |        Final Score = (30% Fin) + (20% Temp) + (25% Iso) + (15% Vend) + (10% MP)      |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
 +---------------------------------------------------------------------------------------+
 |                   STAGE 6 & 8: VALIDATION & MASTER ANOMALY COMPILER                   |
 |  - Save compiled dataset to data/processed/mplads_anomaly_results.csv & SQLite DB      |
 +---------------------------------------------------------------------------------------+
                                             │
                                             ▼
            +--------------------------------+--------------------------------+
            │                                                                 │
            ▼                                                                 ▼
+------------------------------------+             +------------------------------------+
| STAGE 9: Human Audit Service       |             | STAGE 10: Dual-Frontend Interfaces |
| - 7 Detailed Audit Tabs            |             | A. Streamlit App (app.py)          |
| - Decoupled Immutable Audit Log    | <─────────> | B. FastAPI (server.py) +           |
|   (data/audit/review_log.csv)      |             |    React + Vite (frontend/)        |
+------------------------------------+             +------------------------------------+
```

### Technology Stack Summary

- **Data Processing & ML Engine**: Python 3.12, `pandas`, `numpy`, `scikit-learn`, `joblib`, `scipy`.
- **Backend REST API**: FastAPI, Uvicorn, Pydantic, SQLite3 ([`server.py`](file:///d:/MPLADS%28SIH%29/server.py)).
- **Web User Interfaces**:
  1. **React 18 + Vite Frontend**: Modern web dashboard with TailwindCSS / Lucide icons ([`frontend/`](file:///d:/MPLADS%28SIH%29/frontend/)).
  2. **Streamlit Interactive Audit Dashboard**: Pure Python audit platform ([`app.py`](file:///d:/MPLADS%28SIH%29/app.py)).
- **Orchestrator & Testing**: [`scripts/run_pipeline.py`](file:///d:/MPLADS%28SIH%29/scripts/run_pipeline.py), Python `unittest` framework.

---

## 🔄 End-to-End 10-Stage Pipeline

The pipeline is executed sequentially by [`scripts/run_pipeline.py`](file:///d:/MPLADS%28SIH%29/scripts/run_pipeline.py):

| Stage | Module Script | Primary Objective | Key Outputs |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `scripts/data_ingestion/ingest_data.py` | Load raw CSVs, standardize column names & types | Verified raw DataFrames |
| **Stage 2** | `scripts/preprocessing/clean_mplads_data.py` | Parse currency, clean strings, calculate durations | `data/processed/projects_clean.csv` |
| **Stage 3A** | `scripts/feature_engineering/build_financial_features.py` | Extract cost overrun %, Z-scores, peer cohort ranks | `data/features/financial_features.csv` |
| **Stage 3B** | `scripts/feature_engineering/build_temporal_features.py` | Calculate execution days, flag negative dates | `data/features/temporal_features.csv` |
| **Stage 3C** | `scripts/feature_engineering/build_vendor_features.py` | Compute vendor concentration & repeat rates | `data/features/vendor_features.csv` |
| **Stage 3D** | `scripts/feature_engineering/build_nlp_features.py` | Compute TF-IDF text similarity to detect template texts | `data/features/nlp_features.csv` |
| **Stage 4A** | `scripts/feature_engineering/build_isolation_forest_features.py` | Assemble 19 normalized numerical features | `data/features/isolation_forest_features.csv` |
| **Stage 4B** | `scripts/models/train_isolation_forest.py` | Fit `IsolationForest(n_estimators=100, contamination=0.05)` | `models/isolation_forest.joblib` |
| **Stage 4C** | `scripts/models/build_statistical_baselines.py` | Profile peer cohorts (State, District, Category baselines) | `data/processed/statistical_anomaly_results.csv` |
| **Stage 5** | `scripts/evaluation/build_risk_scores.py` | Apply multi-factor risk weighting formula | `data/processed/final_risk_scores.csv` |
| **Stage 6** | `scripts/evaluation/evaluate_anomaly_model.py` | Validate score stability & evaluation metrics | `reports/model_evaluation.md` |
| **Stage 8** | `scripts/data_processing/build_master_anomaly_results.py` | Compile unified dataset & seed SQLite DB | `data/processed/mplads_anomaly_results.csv` |
| **Stage 9** | `scripts/audit/human_verification_service.py` | Initialize audit trail storage service | `data/audit/review_log.csv` |

---

## 🧠 Machine Learning & Risk Scoring Engine

### 1. Isolation Forest Unsupervised Model
- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Parameters**: `n_estimators=100`, `contamination=0.05`, `random_state=42`
- **Feature Set (19 Numerical Features)**: Financial overrun ratio, Z-score, recommendation-to-sanction days, sanction-to-completion days, repeat vendor ratio, max vendor concentration, work completion rate, NLP max similarity score, etc.
- **Output**: Binary Isolation Label ($-1$ Anomaly vs. $1$ Normal) and scaled Isolation Risk Score ($0-100$).

### 2. Multi-Factor Explainable Risk Scoring Formula

$$\text{Final Risk Score} = (0.30 \times S_{\text{financial}}) + (0.20 \times S_{\text{temporal}}) + (0.25 \times S_{\text{isolation}}) + (0.15 \times S_{\text{vendor}}) + (0.10 \times S_{\text{mp\_works}})$$

- **Financial Risk Score ($30\%$)**: Percentile rank and Z-score deviation against peer category baseline cohort ($P_{50}, \mu, P_{99}$).
- **Temporal Risk Score ($20\%$)**: Flags negative date sequences ($100.0$) or extreme execution delay percentiles.
- **Isolation Forest Score ($25\%$)**: Normalized isolation depth severity ($0-100$).
- **Vendor Risk Score ($15\%$)**: Max vendor share, pending payment ratio, and repeat vendor rate.
- **MP Works Risk Score ($10\%$)**: $\text{CLIP}_{0}^{100}(100.0 - \text{completion\_rate\_pct})$.

### 3. Risk Tiers & Distribution

- **`LOW`** ($0.00 – 39.99$): $83,604$ projects ($65.69\%$) — Standard baseline execution.
- **`MEDIUM`** ($40.00 – 69.99$): $43,506$ projects ($34.19\%$) — Moderate statistical variation.
- **`HIGH`** ($70.00 – 100.00$): $153$ projects ($0.12\%$) — **High priority for human audit**.

---

## 🛡️ Human Verification & Decoupled Audit Trail

Reviewers audit flagged high-risk projects through **7 Investigation Tabs**:
1. **📌 Project Details**: Core metadata, MP name, constituency, work description.
2. **💰 Financial Anomaly Explanation**: Sanctioned vs. spent variance & peer Z-scores.
3. **⏱️ Temporal Anomaly Explanation**: Timeline breakdown and date sequence flags.
4. **🏪 Vendor Anomaly Explanation**: Vendor concentration & payment ratio.
5. **📄 NLP Similarity Results**: Flagged duplicate or template descriptions.
6. **📈 Historical Benchmark**: Comparison with sector and district averages.
7. **🤖 Isolation Forest Breakdown**: Feature contribution to tree isolation depth.

### Decoupled Audit Storage
- Review actions (`Verified Normal`, `Needs Further Review`, `Evidence Insufficient`, `Manual Investigation Required`) and auditor notes are appended directly to [`data/audit/review_log.csv`](file:///d:/MPLADS%28SIH%29/data/audit/review_log.csv).
- **Immutability Guarantee**: Model predictions and raw datasets remain untouched, preserving an unalterable ML audit baseline.

---

## 🔌 FastAPI REST API Reference (`server.py`)

The system exposes RESTful endpoints for web applications:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status & loaded records count |
| `GET` | `/api/stats/overview` | Macro metrics: Total projects, total expenditure, risk breakdown |
| `GET` | `/api/projects` | Paginated project list with filtering by state, district, risk level, search query |
| `GET` | `/api/projects/{project_id}` | Detailed project record with feature breakdowns |
| `POST`| `/api/audit/submit` | Save human reviewer decision & notes into audit log |
| `GET` | `/api/audit/logs` | Fetch review audit history |
| `POST`| `/api/predict/new-project` | Run real-time risk prediction on proposed new project input |
| `GET` | `/api/analytics/states` | State-wise aggregate expenditure and high-risk count |

---

## 🚀 Execution & Operating Instructions

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn plotly streamlit pyyaml joblib fastapi uvicorn pydantic
```

### 2. Run Complete 10-Stage Pipeline
```powershell
python scripts/run_pipeline.py
```

### 3. Run Automated Unit Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch Streamlit Dashboard
```powershell
python -m streamlit run app.py
```

### 5. Launch FastAPI Backend Server
```powershell
python -m uvicorn server:app --reload --port 8000
```

---

## ⚠️ System Limitations & Operational Guidance

1. **Statistical Nature**: The system highlights statistical anomalies relative to historical cohorts; it does not constitute proof of illegality or corruption.
2. **Missing Data Imputation**: Missing vendor names or completion dates receive neutral baseline defaults.
3. **External Disruption Factors**: Surge in delays during 2020–2021 reflects COVID-19 pandemic administrative suspensions.
4. **Human Final Authority**: Every high-risk flag requires human administrative verification prior to initiating any formal inquiry.
