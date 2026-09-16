# MPLADS Human-in-the-Loop Audit Dashboard Methodology (Stage 8)

**Pipeline Version:** 8.0  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 8 — Interactive Human-in-the-Loop Audit Dashboard Prototype  

---

## 1. Overview & Objectives

Stage 8 provides an interactive Streamlit web dashboard application (`app.py`) for human administrative review of multi-factor anomaly signals and explainable risk scores across **127,263 MPLADS projects**.

The primary objectives of this stage are to:
1. Provide macro-level executive dashboards showing corpus-wide risk distribution and high-risk state heatmaps.
2. Enable granular multi-criteria filtering across State, Constituency, Category, Year, and Risk Category.
3. Present detailed single-project deep-dives with financial, temporal, vendor, Isolation Forest, and elaborated bullet-point explanations.
4. Visualize cohort historical benchmark comparisons (Median, Mean, $P_{99}$ ceiling).
5. Display near-duplicate text similarity candidates from NLP vectorization.
6. Provide a persistent Human-in-the-Loop verification manager to record official audit statuses.

> [!IMPORTANT]
> **Neutral Operational Framing**: The application strictly **PROHIBITS** automatic labeling of fraud or corruption. All high/medium risk projects are designated as **"Priority for verification"** to guide human administrative oversight.

---

## 2. Dashboard Architecture & 6 Key Views

```
+-----------------------------------------------------------------------------------+
|                        MPLADS AUDIT DASHBOARD MODULES (app.py)                    |
+-----------------------------------------------------------------------------------+
|  1. 📊 Overview           - Metric Cards, Risk Categories & State Heatmaps        |
|  2. 🔎 Project Explorer   - Multi-Criteria Filter Table & Data Downloads          |
|  3. 📋 Project Detail     - Deep-Dive Cards & Elaborated Bullet Explanations      |
|  4. 📈 Historical Baseline- Selected Project vs Cohort P50/P99 Bar Charts         |
|  5. 📄 Similar Projects   - NLP Cosine Similarity Match Candidate Table           |
|  6. 🛡️ Human Verification - Record Official Audit Decision & Save Notes           |
+-----------------------------------------------------------------------------------+
```

---

## 3. Human-in-the-Loop Verification Framework

Reviewers can record official human audit decisions per project:

| Verification Status | Operational Meaning | System Workflow Action |
| :--- | :--- | :--- |
| **`Not Reviewed`** | Default state upon system initialization | Queue for administrative screening |
| **`Verified Normal`** | Document audit confirms standard project execution | Clear from active verification queue |
| **`Needs Further Review`** | Substantial deviation warrants physical site inspection | Escalate to District Collector / Auditor General |
| **`Evidence Insufficient`** | Missing invoice/sanction documentation | Request supplementary documentation from agency |

Audit decisions and reviewer notes are saved to [`data/processed/human_verifications.csv`](file:///d:/MPLADS%28SIH%29/data/processed/human_verifications.csv).

---

## 4. Input Dataset & Artifact Manifest

### 1. [`data/processed/mplads_anomaly_results.csv`](file:///d:/MPLADS%28SIH%29/data/processed/mplads_anomaly_results.csv)
- **Scale**: 127,263 rows $\times$ 35 columns
- **Fields**: `work_id`, `mp_name`, `project_status`, `state`, `constituency`, `category`, `work_description`, `recommended_amount`, `sanctioned_amount`, `effective_amount`, `recommendation_date`, `sanction_date`, `completion_date`, `duration_days`, `recommendation_year`, `is_impossible_date_sequence`, `constituency_max_vendor_share`, `constituency_pending_payment_ratio`, `isolation_forest_score`, `baseline_group`, `group_mean`, `group_median`, `group_p99`, `cost_z_score`, `percentile`, `statistical_flags`, `nearest_similarity_score`, `similar_project_count`, `financial_score`, `temporal_score`, `vendor_score`, `final_risk_score`, `risk_category`, `explanation`, `verification_status`.

### 2. Application Entrypoint
- **File**: [`app.py`](file:///d:/MPLADS%28SIH%29/app.py)
- **Execution Command**: `python -m streamlit run app.py`
