# MPLADS Isolation Forest Unsupervised Anomaly Detection Methodology (Stage 4B)

**Pipeline Version:** 4.2  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 4B — Isolation Forest Model Training & Anomaly Scoring  

---

## 1. Overview & Objectives

Stage 4B trains an unsupervised `IsolationForest` machine learning model on the 19 active numerical features engineered across 127,263 MPLADS project records.

The primary objectives of this stage are to:
1. Perform pre-flight data audits ensuring zero target leakage, zero missing values, and zero infinite values.
2. Standardize feature vectors using `StandardScaler` to prevent high-magnitude financial features from dominating tree partitioning logic.
3. Fit an unsupervised `IsolationForest` ensemble (`n_estimators = 100`, `contamination = 0.05`, `random_state = 42`).
4. Generate decision function anomaly scores, standard binary labels (`-1` vs `1`), and a UI-ready `normalized_anomaly_score` ($[0.0, 1.0]$).
5. Serialize trained model and scaler pipelines to disk (`models/isolation_forest.joblib`, `models/scaler.joblib`).
6. Export project-level anomaly scores to `data/processed/isolation_forest_results.csv`.

> [!IMPORTANT]
> **Neutral Terminology Compliance**: Isolation Forest flags statistical outliers (records isolated quickly due to extreme values or rare feature combinations). Flagged records are **statistical anomalies requiring human verification**, NOT proof of fraud, corruption, or intentional wrongdoing.

---

## 2. Model Architecture & Hyperparameters

### 2.1 Feature Scaling Protocol (`StandardScaler`)
- **Fitted Scope**: `StandardScaler` is fitted exclusively on the 19 active numerical feature columns of feature matrix $X$.
- **Transformation**: Centers each feature to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$).
- **Artifact**: Serialized to `models/scaler.joblib`.

### 2.2 Model Configuration (`IsolationForest`)
- **Ensemble Estimators (`n_estimators`)**: `100` isolation trees.
- **Contamination Rate (`contamination`)**: `0.05` ($5.00\%$ target anomaly proportion).
- **Sub-sampling**: Standard $256$ samples per tree (`max_samples='auto'`).
- **Random State**: `42` (reproducibility across runs).
- **Multi-threading (`n_jobs`)**: `-1` (parallelized CPU execution).
- **Artifact**: Serialized to `models/isolation_forest.joblib`.

---

## 3. Anomaly Scoring & Dashboard Normalization

Each project record is assigned 4 scoring metrics:

1. **`raw_anomaly_score`**: Output of `decision_function(X_scaled)`.
   - Positive scores indicate normal observations ($\text{score} > 0$).
   - Negative scores indicate isolated anomalies ($\text{score} < 0$).
2. **`anomaly_label`**: Standard scikit-learn label:
   - `-1`: Flagged anomaly (top $5.00\%$ most isolated records).
   - `1`: Normal observation.
3. **`is_anomaly`**: Binary indicator ($1$ for anomaly, $0$ for normal).
4. **`normalized_anomaly_score`**: Min-max scaled score ($[0.0, 1.0]$) for frontend dashboard presentation:
   $$\text{normalized\_score} = \frac{-\text{raw\_score} - \min(-\text{raw\_score})}{\max(-\text{raw\_score}) - \min(-\text{raw\_score})}$$
   - `0.0000`: Least anomalous project (most normal).
   - `1.0000`: Most anomalous project (highest risk severity).

---

## 4. Score Distribution & Audit Results

### 4.1 Global Anomaly Summary
- **Total Observations Processed**: $127,263$ projects
- **Active Model Features Used**: $19$ numerical features
- **Total Flagged Anomalies**: $6,363$ projects
- **Anomaly Contamination Rate**: $5.00\%$
- **Pipeline Execution Time**: $3.40$ seconds

### 4.2 Score Distribution Statistics

| Metric | Raw Decision Function Score | Normalized Anomaly Score $[0, 1]$ |
| :--- | :---: | :---: |
| **Mean** | $+0.0778$ | $0.2008$ |
| **Std Dev** | $0.0422$ | $0.1177$ |
| **Min (Most Anomalous)** | $-0.2085$ | **$1.0000$** |
| **25th Percentile** | $+0.0547$ | $0.2652$ |
| **50th Percentile (Median)** | $+0.0857$ | $0.1787$ |
| **75th Percentile** | $+0.1092$ | $0.1131$ |
| **Max (Least Anomalous)** | $+0.1497$ | **$0.0000$** |

### 4.3 Anomaly Breakdown by Project Status Cohort

| Project Status Cohort | Total Projects | Anomalies Flagged | Cohort Anomaly Rate |
| :--- | :---: | :---: | :---: |
| **Completed** | 43,295 | 1,658 | 3.83% |
| **Recommended** | 83,528 | 4,639 | 5.55% |
| **Recommended & Completed (Linked)** | 440 | 66 | **15.00%** |
| **Total Corpus** | **127,263** | **6,363** | **5.00%** |

---

## 5. Output Datasets & Model Manifest

1. [`models/isolation_forest.joblib`](file:///d:/MPLADS%28SIH%29/models/isolation_forest.joblib)
   - Trained scikit-learn `IsolationForest` model.
2. [`models/scaler.joblib`](file:///d:/MPLADS%28SIH%29/models/scaler.joblib)
   - Fitted scikit-learn `StandardScaler` pipeline.
3. [`data/processed/isolation_forest_results.csv`](file:///d:/MPLADS%28SIH%29/data/processed/isolation_forest_results.csv)
   - Scale: 127,263 rows $\times$ 8 columns (`work_id`, `project_status`, `state`, `constituency`, `raw_anomaly_score`, `normalized_anomaly_score`, `anomaly_label`, `is_anomaly`).
