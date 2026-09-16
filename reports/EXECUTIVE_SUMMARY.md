# Executive Summary & Presentation Synthesis
## MPLADS AI Multi-Factor Anomaly Detection & Explainable Risk System

**Project:** MPLADS AI Anomaly Detection System (SIH Edition)  
**Corpus Scale:** 127,263 Projects | 36 States/UTs | 537 Constituencies  
**Author:** AI Pair Programming Agent (Antigravity Team)  
**Date:** September 1, 2026  
**Pipeline Version:** 6.0 (Production Release)  

---

## 1. Executive Summary

The **Members of Parliament Local Area Development Scheme (MPLADS)** facilitates local infrastructure and developmental works across India. Monitoring project execution at scale presents significant administrative challenges due to data heterogeneity, regional variance, and complex multi-vendor ecosystems.

This project delivers an end-to-end, scalable, machine learning and statistical framework designed to detect anomalous patterns and prioritize projects for administrative verification. By integrating **Financial, Temporal, Vendor Concentration, NLP Text Similarity, and Unsupervised Isolation Forest** signals into a single, explainable **Multi-Factor Risk Score ($0.00 – 100.00$)**, the system highlights high-priority projects while upholding strict non-accusatory governance standards.

```
+-----------------------------------------------------------------------------------+
|                            MPLADS AI PIPELINE ARCHITECTURE                        |
+-----------------------------------------------------------------------------------+
|  Stage 1: Cleaned 127,263 Projects (0 NaN IDs, Standardized States/Constituencies)|
|  Stage 2-3D: Engineered 21 Financial, Temporal, Vendor & NLP Similarity Features |
|  Stage 4A-4B: Trained Isolation Forest Model (100 Trees, 5.0% Contamination)      |
|  Stage 4C: Calculated Hierarchical Statistical Baselines (Z-Scores & Quantiles)   |
|  Stage 5: Synthesized Explainable Multi-Factor Risk Scores & Bullet Explanations  |
|  Stage 6: Evaluated Model Stability (Seed ρ=0.9426) & Feature Effect Sizes       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Empirical Findings & System Metrics

### 2.1 Risk Priority Distribution
Out of **127,263 projects** evaluated across India:

| Risk Category | Score Range | Project Count | Percentage of Corpus | Operational Action Required |
| :--- | :---: | :---: | :---: | :--- |
| **`HIGH`** | $70.00 – 100.00$ | **204** | **0.16%** | **Immediate Administrative Audit & Field Inspection** |
| **`MEDIUM`** | $40.00 – 69.99$ | **36,856** | **28.96%** | Secondary Document Verification & Milestone Review |
| **`LOW`** | $0.00 – 39.99$ | **90,203** | **70.88%** | Standard Baseline Monitoring |

> [!NOTE]
> Out of 127,263 projects, only **204 projects ($0.16\%$)** meet the **`HIGH`** risk threshold ($\ge 70.00$), enabling audit teams to focus resources on top priority items.

---

### 2.2 Model Reliability & Validation Metrics

| Validation Metric Dimension | Measured Result | Operational Significance |
| :--- | :---: | :--- |
| **Multi-Seed Rank Stability ($\rho$)** | **`0.9426`** | High rank correlation across 5 random seeds confirms non-random splits |
| **Top-5% Anomaly Consensus (Jaccard)** | **`0.6378`** | Strong alignment on top flagged outliers across random tree initializations |
| **Contamination Hierarchy Retention** | **100.0%** | Core 1% outliers are 100% retained within 3%, 5%, and 10% parameter sets |
| **Top Feature Effect Size (State Cost %)** | **`d = +0.87`** | Strong statistical separation between normal and anomalous cohorts |
| **Data Integrity & Completeness** | **100.0%** | 0 nulls, 0 infinite values across entire training and risk matrices |

---

## 3. High-Priority Case Studies (Real Project Demonstrations)

Below are real project examples flagged by the system, featuring evidence-grounded bullet-point explanations:

### Case Study 1: Work ID `61773` (Tamil Nadu / Sitting Rajya Sabha)
- **Final Risk Score**: **`80.18 / 100`** (`HIGH Priority`)
- **Component Scores**: Financial = `99.92` | Temporal = `100.00` | Isolation Forest = `67.98` | Vendor = `21.41`
- **Elaborated Bullet Explanation**:
  > **High priority for verification (Final Risk Score: 80.18/100):**  
  > • **Financial Benchmark Deviation**: Expenditure of ₹10,976,847.00 is unusually high compared with Category: Normal/Others benchmark mean of ₹542,801.52 ($Z$-Score: $+10.20$, Cohort Percentile: $99.9\%$).  
  > • **Constituency Project Volume Spike**: Annual recommendation volume grew by $+1818.1\%$ YoY, indicating an unusual concentration of project approvals.  
  > • **Isolation Forest Machine Learning Pattern**: Multi-dimensional feature isolation detected across combined financial, temporal, and vendor space (Model Severity: $68.0/100$).

---

### Case Study 2: Work ID `7438` (Punjab / Sitting Rajya Sabha)
- **Final Risk Score**: **`73.04 / 100`** (`HIGH Priority`)
- **Component Scores**: Financial = `82.44` | Temporal = `100.00` | Isolation Forest = `69.97` | Vendor = `11.31`
- **Elaborated Bullet Explanation**:
  > **High priority for verification (Final Risk Score: 73.04/100):**  
  > • **Financial Benchmark Deviation**: Expenditure of ₹6,322,296.00 is unusually high compared with Category: Normal/Others benchmark mean of ₹542,801.52 ($Z$-Score: $+5.65$, Cohort Percentile: $99.7\%$).  
  > • **Constituency Project Volume Spike**: Annual recommendation volume grew by $+1177.7\%$ YoY, indicating an unusual concentration of project approvals.  
  > • **Isolation Forest Machine Learning Pattern**: Multi-dimensional feature isolation detected across combined financial, temporal, and vendor space (Model Severity: $70.0/100$).

---

### Case Study 3: Work ID `149` (Rajasthan / Sitting Rajya Sabha)
- **Final Risk Score**: **`45.33 / 100`** (`MEDIUM Priority`)
- **Component Scores**: Financial = `59.40` | Temporal = `39.77` | Isolation Forest = `50.89` | Vendor = `12.50`
- **Elaborated Bullet Explanation**:
  > **Priority for verification (Final Risk Score: 45.33/100):**  
  > • **Constituency Project Volume Spike**: Annual recommendation volume grew by $+397.7\%$ YoY, indicating an unusual concentration of project approvals.  
  > • **Isolation Forest Machine Learning Pattern**: Multi-dimensional feature isolation detected across combined financial, temporal, and vendor space (Model Severity: $50.9/100$).  
  > • **Text Similarity Candidate**: Work description matches 1 other project(s) in constituency with high text similarity ($100.0\%$ cosine match score).

---

## 4. Operational Governance & Neutral Compliance Framework

1. **Non-Accusatory Classification**:
   - The system strictly avoids labeling projects as "fraud" or "corruption". All outputs use neutral operational framing: **"Priority for verification"**.
2. **Audit Action Workflow**:
   - **`HIGH` Risk (204 projects)**: Flagged for mandatory physical site verification and comprehensive invoice audit.
   - **`MEDIUM` Risk (36,856 projects)**: Flagged for administrative desk audit and vendor contract review.
   - **`LOW` Risk (90,203 projects)**: Approved for standard automated progress tracking.
3. **Data Quality Flags vs. Operational Anomalies**:
   - Historical sanity checks revealed that $0.91\%$ of anomalies ($58$ projects) stem from data input errors (e.g. completion date preceding recommendation date). These are segregated into data cleaning queues rather than vendor audit queues.

---

## 5. Presentation Synthesis Slide Deck Outline

This outline synthesizes the project into a **10-slide presentation deck** ready for presentation to ministry officials, SIH judges, or administrative evaluators.

---

### Slide 1: Title & Overview
- **Title**: MPLADS AI Multi-Factor Anomaly Detection & Risk System
- **Subtitle**: Machine Learning & Statistical Governance Framework for Infrastructure Oversight
- **Scope**: 127,263 Projects Across 36 States/UTs and 537 Constituencies

---

### Slide 2: The Challenge
- **Problem Statement**: High volume of MPLADS works makes manual inspection of every project impossible.
- **Key Bottlenecks**: Regional cost variances, missing vendor histories, and duplicate work descriptions.
- **Solution Goal**: Data-driven priority scoring to direct human audit resources where statistical deviations are highest.

---

### Slide 3: End-to-End System Architecture
- **Stage 1**: Automated Cleaning & Normalization ($127,263 \times 12$ clean records).
- **Stage 2–3D**: Multi-Domain Feature Engineering (Financial, Temporal, Vendor, NLP Similarity).
- **Stage 4A–4C**: Unsupervised Isolation Forest Modeling & Hierarchical Statistical Baselines.
- **Stage 5–6**: Explainable Multi-Factor Risk Scoring & Model Validation.

---

### Slide 4: Feature Engineering & Domain Signals
- **Financial**: Cost Z-scores, recommendation-to-sanction ratios, state/constituency percentiles.
- **Temporal**: Date sequence logic checks, duration deviations, YoY recommendation growth spikes.
- **Vendor**: Constituency vendor expenditure concentration, pending payment ratios, repeat transaction rates.
- **NLP Similarity**: TF-IDF cosine similarity vectorization ($\ge 0.85$ near-duplicate detection).

---

### Slide 5: Unsupervised Isolation Forest Modeling (Stage 4B)
- **Algorithm**: `IsolationForest` ensemble ($100$ trees, $5\%$ contamination rate).
- **Matrix Dimensions**: $127,263 \times 19$ numerical features (scaled via `StandardScaler`).
- **Results**: Identified $6,363$ statistical outlier projects ($5.00\%$) in $3.40$ seconds.

---

### Slide 6: Hierarchical Statistical Baselines (Stage 4C)
- **Hierarchy**: Category $\rightarrow$ Constituency $\rightarrow$ State (Minimum group size $N \ge 10$).
- **Parametric & Non-Parametric Metrics**: Mean ($\mu$), Std ($\sigma$), Median ($P_{50}$), $P_{75}, P_{90}, P_{95}, P_{99}$.
- **Multi-Thresholding**: Parametric $Z > 2.5$ ($2,409$ projects) & Quantile $P_{95}$ ($6,148$ projects).

---

### Slide 7: Explainable Multi-Factor Risk Scoring (Stage 5)
- **Transparent Formula**:
  $$\text{Risk Score} = 0.35 \times \text{Financial} + 0.25 \times \text{Temporal} + 0.25 \times \text{Isolation Forest} + 0.15 \times \text{Vendor}$$
- **Risk Categorization**:
  - `LOW` ($0 – 39$): $90,203$ projects ($70.88\%$)
  - `MEDIUM` ($40 – 69$): $36,856$ projects ($28.96\%$)
  - `HIGH` ($70 – 100$): **$204$ projects ($0.16\%$)**

---

### Slide 8: Explainability & Automated Evidence Generation
- **Automated Bullet Explanations**: Generated for every `MEDIUM` and `HIGH` project.
- **Sample High-Risk Explanation**:
  > *Expenditure ₹10.98M vs Category Mean ₹542k (Z=+10.20) | YoY Volume Spike +1818.1% | Isolation Forest Severity 68.0/100*
- **Governance Standard**: Neutral framing ("Priority for verification").

---

### Slide 9: Model Validation & Empirical Stability (Stage 6)
- **Multi-Seed Stability**: Spearman rank correlation $\rho = 0.9426$ across 5 random seeds.
- **Top Feature Effect Sizes**: State Cost Percentile ($d = +0.87$), Cost Deviation ($d = +0.76$), Effective Amount ($d = +0.71$).
- **Contamination Sensitivity**: $100\%$ consensus retention of core $1\%$ outliers across parameter bands.

---

### Slide 10: Operational Impact & Future Roadmap
- **Immediate Administrative Impact**: Audit teams can focus field inspections on the top **$204$ High-Risk projects** rather than sorting through 127,263 records manually.
- **Future Enhancements**:
  - Web UI / Dashboard integration (React + Vite / Streamlit).
  - Automated GIS / Spatial coordinate mapping for physical project verification.

---

## 6. Output Dataset & Code Base Manifest

- **Executive Summary Report**: [`reports/EXECUTIVE_SUMMARY.md`](file:///d:/MPLADS%28SIH%29/reports/EXECUTIVE_SUMMARY.md)
- **Final Risk Scores CSV**: [`data/processed/final_risk_scores.csv`](file:///d:/MPLADS%28SIH%29/data/processed/final_risk_scores.csv) (127,263 rows $\times$ 13 columns)
- **Statistical Baselines CSV**: [`data/processed/statistical_anomaly_results.csv`](file:///d:/MPLADS%28SIH%29/data/processed/statistical_anomaly_results.csv) (127,263 rows $\times$ 21 columns)
- **Isolation Forest Results CSV**: [`data/processed/isolation_forest_results.csv`](file:///d:/MPLADS%28SIH%29/data/processed/isolation_forest_results.csv) (127,263 rows $\times$ 8 columns)
- **Model Pipeline Objects**: [`models/isolation_forest.joblib`](file:///d:/MPLADS%28SIH%29/models/isolation_forest.joblib) & [`models/scaler.joblib`](file:///d:/MPLADS%28SIH%29/models/scaler.joblib)
- **Evaluation Report & Figures**: [`reports/model_evaluation.md`](file:///d:/MPLADS%28SIH%29/reports/model_evaluation.md) & [`reports/figures/`](file:///d:/MPLADS%28SIH%29/reports/figures/)
