# MPLADS Statistical Anomaly Baselines Methodology (Stage 4C)

**Pipeline Version:** 4.3  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 4C — Interpretable Parametric & Quantile Statistical Baselines  

---

## 1. Overview & Objectives

Stage 4C constructs interpretable statistical anomaly baselines to complement the unsupervised `IsolationForest` model.

While Isolation Forest detects complex multi-dimensional feature isolation, **Statistical Baselines** provide transparent, human-auditable metrics ($Z$-scores and quantile percentiles) that explain *why* specific project expenditures deviate significantly from peer benchmark cohorts.

The primary objectives of this stage are to:
1. Group project records into hierarchical cohorts (Category $\rightarrow$ Constituency $\rightarrow$ State) enforcing a minimum group size threshold ($N \ge 10$).
2. Compute parametric ($\mu, \sigma$) and non-parametric quantile benchmarks ($P_{50}, P_{75}, P_{90}, P_{95}, P_{99}$) for every cohort.
3. Calculate project-level $Z$-scores and cohort percentile ranks across all 127,263 projects.
4. Evaluate multi-threshold anomaly flags ($Z > 2.5$, $P_{95}$, $P_{99}$).
5. Export results to `data/processed/statistical_anomaly_results.csv`.

> [!IMPORTANT]
> **Neutral Terminology Compliance**: High $Z$-scores or top-percentile ranks reflect extreme monetary magnitude or unusual statistical variance. Flagged records are **statistical anomalies requiring verification**, NOT proof of fraud, corruption, or intentional misallocation.

---

## 2. Hierarchical Cohort Baseline Architecture

To prevent misleading or noisy statistical signals (e.g. 100th percentile scores assigned to isolated projects in categories with $N = 1$), the pipeline enforces `MIN_GROUP_SIZE = 10` and assigns baselines hierarchically:

1. **Priority 1 — Category Cohort**:
   - Applied to completed projects ($N = 43,729$, $34.36\%$ of corpus) where explicit sub-category labels exist and $N_{\text{cat}} \ge 10$.
2. **Priority 2 — Constituency Cohort**:
   - Applied to recommended-only projects ($N = 83,505$, $65.62\%$ of corpus) or categories with $N < 10$, where $N_{\text{const}} \ge 10$.
3. **Priority 3 — State Fallback**:
   - Applied to projects in tiny constituencies with $N < 10$ ($N = 29$, $0.02\%$ of corpus).

---

## 3. Parametric & Quantile Formulas

### 3.1 Parametric $Z$-Score
$$Z = \frac{x - \mu_{\text{cohort}}}{\sigma_{\text{cohort}}}$$
- **Formula**: Measures how many standard deviations a project's cost ($x$) lies above or below the cohort mean ($\mu$).
- **Threshold**: $Z > 2.5$ flags projects exceeding $+2.5 \sigma$ from cohort baseline.

### 3.2 Non-Parametric Quantile Ranks ($P_{50}, P_{75}, P_{90}, P_{95}, P_{99}$)
$$\text{Percentile Rank} = \frac{\text{Rank}(x)}{N_{\text{cohort}}} \times 100.0$$
- **Formula**: Computes exact quantile placement of project expenditure within its cohort.
- **Thresholds**: $P_{95}$ ($\ge 95.0\%$) flags the top 5% upper-tail expenditures; $P_{99}$ ($\ge 99.0\%$) flags the top 1% extreme upper-tail expenditures.

---

## 4. Statistical Anomaly Threshold Audit Summary

Across the **127,263 projects** evaluated against **537 unique cohort groups**:

| Anomaly Threshold Criteria | Flagged Projects | Proportion of Corpus | Operational Significance |
| :--- | :---: | :---: | :--- |
| **$Z$-Score Threshold ($Z > 2.5$)** | **2,409** | **1.89%** | Severe parametric upper-tail deviation ($+2.5\sigma$) |
| **95th Percentile ($P_{95} \ge 95.0\%$)** | **6,148** | **4.83%** | Upper 5th percentile cost cohort within group |
| **99th Percentile ($P_{99} \ge 99.0\%$)** | **1,410** | **1.11%** | Extreme top 1st percentile cost cohort within group |
| **Combined Flag ($Z > 2.5 \text{ OR } P_{95}$)** | **6,168** | **4.85%** | Union of parametric & quantile statistical anomalies |

---

## 5. Top Extreme Statistical Anomaly Case Studies

| Case | Work ID | State / Constituency | Baseline Cohort Group | Effective Amount | Group Mean ($\mu$) | Group $P_{99}$ | $Z$-Score | Percentile Rank |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1** | **141045** | Uttar Pradesh / AONLA | Category: Normal/Others | ₹46,470,400.00 | ₹542,801.52 | ₹3,069,927.00 | **$+44.91$** | **100.00%** |
| **#2** | **194809** | Uttar Pradesh / SHAHJAHANPUR | Category: Normal/Others | ₹40,095,000.00 | ₹542,801.52 | ₹3,069,927.00 | **$+38.68$** | **100.00%** |
| **#3** | **65656** | Tamil Nadu / Sitting Rajya Sabha | Category: Normal/Others | ₹39,955,654.00 | ₹542,801.52 | ₹3,069,927.00 | **$+38.54$** | **100.00%** |
| **#4** | **134984** | Uttar Pradesh / SHAHJAHANPUR | Category: Normal/Others | ₹37,908,000.00 | ₹542,801.52 | ₹3,069,927.00 | **$+36.54$** | **99.99%** |
| **#5** | **171199** | West Bengal / Sitting Rajya Sabha | Constituency: Sitting Rajya Sabha | ₹73,500,000.00 | ₹949,283.71 | ₹7,000,000.00 | **$+36.30$** | **100.00%** |

---

## 6. Output Dataset Manifest

### [`data/processed/statistical_anomaly_results.csv`](file:///d:/MPLADS%28SIH%29/data/processed/statistical_anomaly_results.csv)
- **Scale**: 127,263 rows $\times$ 21 columns
- **Fields**: `work_id`, `project_status`, `state`, `constituency`, `category`, `effective_amount`, `baseline_group`, `baseline_type`, `group_mean`, `group_std`, `group_median`, `group_p75`, `group_p90`, `group_p95`, `group_p99`, `z_score`, `percentile`, `statistical_flag`, `z_flag`, `p95_flag`, `p99_flag`.
