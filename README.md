# 🛡️ MPLADS AI Anomaly Monitoring & Human-in-the-Loop Audit System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-v1.62-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Objective

The **MPLADS AI Anomaly Monitoring & Human-in-the-Loop Audit System** is an end-to-end Machine Learning and Explainable AI platform built for the **Members of Parliament Local Area Development Scheme (MPLADS)**.

It processes **127,263 official project records**, standardizes irregular data formats, extracts domain-specific financial, temporal, vendor, and NLP features, and applies **unsupervised anomaly detection** combined with **multi-factor risk scoring** to prioritize projects for human administrative verification.

> [!IMPORTANT]
> **Operational Compliance Policy**: The system identifies unusual statistical patterns and prioritizes projects for human administrative verification. It strictly **DOES NOT** automatically claim fraud, corruption, or illegal wrongdoing.

---

## 📊 Data Sources

1. **`data/raw/projects.csv`** ($127,263$ rows $\times$ 14 columns): Individual project recommendations, sanctioned values, completion dates, and work descriptions.
2. **`data/raw/work_by_type.csv`** ($127,263$ rows $\times$ 15 columns): Project work category breakdowns and expenditure values.
3. **`data/raw/mp_summary.csv`** ($794$ MPs $\times$ 10 columns): MP-level summary metrics (Allocated Funds, Total Expenditure, Completed Works, Recommended Works, Utilization %, Completion Rate %).

---

## 🔄 End-to-End Pipeline Architecture

```
                                +-----------------------------------+
                                |    Official MPLADS Data Sources   |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 1: Ingestion & Schema Std.  |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                |  Stage 2: Cleaning & Integration  |
                                +-----------------------------------+
                                                  │
                                                  ▼
              +-----------------------------------+-----------------------------------+
              │                                   │                                   │
              ▼                                   ▼                                   ▼
    +-------------------+               +-------------------+               +-------------------+
    | Stage 3A: Financial|              | Stage 3B: Temporal|               | Stage 3C: Vendor  |
    | Feature Extract.  |               | Feature Extract.  |               | Feature Extract.  |
    +-------------------+               +-------------------+               +-------------------+
              │                                   │                                   │
              +-----------------------------------+-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 3D: NLP Text Similarity     |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 4A: Feature Matrix Build    |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 4B: Isolation Forest Model  |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 4C: Statistical Baselines   |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 5: Multi-Factor Risk Score  |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 6: Model Validation & Eval. |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 8: Master Anomaly Compiler  |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 9: Human Verification Audit |
                                +-----------------------------------+
                                                  │
                                                  ▼
                                +-----------------------------------+
                                | Stage 10: Streamlit Dashboard App |
                                +-----------------------------------+
```

---

## 🧠 Machine Learning & Risk Scoring Methodology

### 1. Isolation Forest Unsupervised Anomaly Detection (Stage 4B)
- **Model**: `sklearn.ensemble.IsolationForest` ($n\_estimators=100$, $contamination=0.05$, $random\_state=42$)
- **Preprocessing**: `StandardScaler` fitted on 19 numerical features (financial deviations, durations, growth rates, vendor concentration ratios).
- **Output**: Binary anomaly labels ($-1$ Anomaly vs $1$ Normal) and 0-100 normalized severity scores.

### 2. Multi-Factor Explainable Risk Scoring Formula (Stage 5)
Projects are evaluated using a 5-component weighted risk model:

$$\text{Final Risk Score} = (0.30 \times S_{\text{financial}}) + (0.20 \times S_{\text{temporal}}) + (0.25 \times S_{\text{isolation}}) + (0.15 \times S_{\text{vendor}}) + (0.10 \times S_{\text{works}})$$

- **Financial Score ($30\%$)**: Percentile rank and Z-score deviation against peer baseline cohort ($P_{50}, \mu, P_{99}$).
- **Temporal Score ($20\%$)**: Negative duration sequence flag ($100.0$) or YoY recommendation volume growth.
- **Isolation Forest Score ($25\%$)**: Normalized tree isolation depth score ($0-100$).
- **Vendor Risk Score ($15\%$)**: Max vendor share, pending payment ratio, and repeat vendor rate.
- **MP Works Risk Score ($10\%$)**: $\text{CLIP}_{0}^{100}(100.0 - \text{completion\_rate\_pct})$.

### 3. Risk Categories & Distribution
- **`LOW`** ($0.00 – 39.99$): $83,604$ projects ($65.69\%$) — Baseline normal execution.
- **`MEDIUM`** ($40.00 – 69.99$): $43,506$ projects ($34.19\%$) — Moderate statistical deviation.
- **`HIGH`** ($70.00 – 100.00$): $153$ projects ($0.12\%$) — **High priority for human verification**.

---

## 🛡️ Stage 9: Human Verification & Audit Trail Storage

Reviewers audit flagged projects in the interactive Streamlit workspace ([`app.py`](file:///d:/MPLADS%28SIH%29/app.py)):

### 7 Investigation Tabs Per Flagged Project:
1. **📌 Project Details**
2. **💰 Financial Anomaly Explanation**
3. **⏱️ Temporal Anomaly Explanation**
4. **🏪 Vendor Anomaly Explanation**
5. **📄 NLP Similarity Results**
6. **📈 Historical Expenditure Benchmark**
7. **🤖 Isolation Forest Score**

### 4 Reviewer Actions & Audit Trail Storage:
- **Actions**: `Verified Normal`, `Needs Further Review`, `Evidence Insufficient`, `Manual Investigation Required`.
- **Decoupled Audit Log**: Review decisions are saved separately in [`data/audit/review_log.csv`](file:///d:/MPLADS%28SIH%29/data/audit/review_log.csv) with timestamp and reviewer ID. Model predictions remain 100% immutable.

---

## 🚀 How to Run the System

### 1. Requirements & Environment Setup
Python 3.12+ recommended. Activate virtual environment and install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn plotly streamlit pyyaml joblib
```

### 2. Run Master Pipeline Orchestrator (Stage 1 to Stage 10)
Execute the complete data processing, feature engineering, model training, risk scoring, and audit log initialization in one command:
```powershell
python scripts/run_pipeline.py
```
*Logs are output to console and saved in [`logs/pipeline.log`](file:///d:/MPLADS%28SIH%29/logs/pipeline.log).*

### 3. Run Automated Unit Test Suite
Verify data loading, currency parsing, date sequence parsing, feature matrix, model prediction ranges, and risk scoring logic:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch Streamlit Audit Dashboard
Launch the interactive 6-view audit dashboard:
```powershell
python -m streamlit run app.py
```
*Access in browser at `http://localhost:8501`.*

---

## ⚠️ System Limitations & Operational Guidance

1. **Unsupervised Nature**: Isolation Forest detects statistical rarity relative to historical cohorts; it does not constitute proof of corruption.
2. **Data Gaps**: Projects with missing vendor names or missing completion dates receive neutral baseline defaults.
3. **COVID-19 Pandemic Impact**: Recommendation surges or low completion rates during 2020-2021 reflect scheme suspension and macro administrative delays.
4. **Human Final Authority**: Every high-risk flag requires human administrative review before any official inquiry is initiated.
