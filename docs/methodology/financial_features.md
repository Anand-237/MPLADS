# MPLADS Financial Feature Engineering Methodology (Stage 3A)

**Pipeline Version:** 3.1  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 3A - Numerical Financial Feature Engineering  

---

## 1. Overview & Objectives

Stage 3A extracts numerical financial features designed to expose monetary anomalies, fund inflation, cost deviations, and utilization inefficiencies in MPLADS project execution.

The feature engineering pipeline operates dynamically on cleaned datasets (`data/processed/projects_clean.csv` and `data/processed/mp_summary_clean.csv`) and enforces statistical validity through minimum group-size thresholding.

---

## 2. Feature Specifications & Formulas

### 2.1 Ratio & Balance Features

#### 1. `expenditure_ratio`
- **Formula:** $\frac{\text{final\_amount}}{\text{recommended\_amount}}$
- **Scope:** Computed for linked projects where both recommendation and completion costs exist ($N = 440$).
- **Purpose:** Identifies cost overruns ($\text{ratio} > 1.0$) or massive budget underutilization.

#### 2. `release_ratio`
- **Formula:** $\frac{\text{total\_expenditure}}{\text{allocated\_amount}}$ (from MP Summary)
- **Scope:** Computed for all 127,263 projects via MP/Constituency lookup ($N = 127,263$).
- **Purpose:** Measures overall financial disbursal efficiency of the sponsoring Member of Parliament.

#### 3. `recommendation_sanction_ratio`
- **Formula:** $\frac{\text{recommended\_amount}}{\mu_{\text{State}}(\text{recommended\_amount})}$
- **Scope:** Projects with valid `recommended_amount` ($N = 83,968$).
- **Purpose:** Highlights projects recommended for amounts significantly higher than state baseline averages.

#### 4. `unspent_balance`
- **Formula:** $\text{recommended\_amount} - \text{final\_amount}$
- **Scope:** Linked recommended and completed projects ($N = 440$).
- **Purpose:** Quantifies net leftover funds or excess expenditure per project.

---

### 2.2 Group Deviation & Percentile Features

#### 5. `expenditure_deviation`
- **Formula:** $Z = \frac{\text{final\_amount} - \mu_{\text{Category}}(\text{final\_amount})}{\sigma_{\text{Category}}(\text{final\_amount})}$
- **Threshold:** Evaluated only for categories with group size $N \ge 10$.
- **Usable Records:** $43,729$ (34.36%).

#### 6. `cost_deviation`
- **Formula:** $Z = \frac{\text{effective\_amount} - \mu_{\text{State}}(\text{effective\_amount})}{\sigma_{\text{State}}(\text{effective\_amount})}$
- **Threshold:** Evaluated for States with group size $N \ge 10$.
- **Usable Records:** $127,263$ (100.00%).

#### 7. `cost_percentile`
- **Formula:** $\text{Percentile Rank of effective\_amount within Category Cohort}$
- **Threshold:** Evaluated only for categories with group size $N \ge 10$.
- **Usable Records:** $43,729$ (34.36%).

#### 8. `state_cost_percentile`
- **Formula:** $\text{Percentile Rank of effective\_amount within State Cohort}$
- **Threshold:** Evaluated for States with group size $N \ge 10$.
- **Usable Records:** $127,263$ (100.00%).

#### 9. `constituency_cost_percentile`
- **Formula:** $\text{Percentile Rank of effective\_amount within Constituency Cohort}$
- **Threshold:** Evaluated for Constituencies with group size $N \ge 10$.
- **Usable Records:** $127,225$ (99.97% — 38 records in 9 small constituencies with $N < 10$ assigned `NaN`).

---

## 3. Minimum Group-Size Thresholding Rule

To prevent misleading or noisy statistical signals (e.g. 100th percentile scores assigned to isolated projects in categories or constituencies with only 1 observation), the pipeline enforces `MIN_GROUP_SIZE = 10`.

- If a group has fewer than 10 observations, all group-relative percentiles and Z-scores evaluate to `NaN`.
- Example: Category `Bar and Associations` ($N=1$) receives `NaN` for `cost_percentile` and `expenditure_deviation`.

---

## 4. Summary Statistics of Feature Matrix

| Feature Name | Usable Records | Null Records | Mean | Std | Min | 50% (Median) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `expenditure_ratio` | 440 | 126,823 | 1.0690 | 0.9769 | 0.0972 | 0.9996 | 14.5000 |
| `release_ratio` | 127,263 | 0 | 0.3849 | 0.1945 | 0.0000 | 0.3798 | 0.9509 |
| `recommendation_sanction_ratio` | 83,968 | 43,295 | 1.0000 | 1.7443 | 0.0000 | 0.6808 | 126.9237 |
| `unspent_balance` (₹) | 440 | 126,823 | ₹67,337 | ₹384,742 | -₹1.35M | ₹206.50 | ₹2.25M |
| `expenditure_deviation` (Z) | 43,729 | 83,534 | 0.0000 | 1.0000 | -0.9108 | -0.2374 | 44.9131 |
| `cost_deviation` (Z) | 127,263 | 0 | 0.0000 | 0.9999 | -1.5204 | -0.2132 | 78.1276 |
| `cost_percentile` | 43,729 | 83,534 | 50.0034 | 28.8647 | 0.0023 | 49.9107 | 100.0000 |
| `state_cost_percentile` | 127,263 | 0 | 50.0141 | 28.7513 | 0.0105 | 50.4658 | 100.0000 |
| `constituency_cost_percentile` | 127,225 | 38 | 50.2075 | 28.1033 | 0.0063 | 50.6579 | 100.0000 |

---

## 5. Output Datasets

- [`data/features/financial_features.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/features/financial_features.csv) (127,263 rows, 17 columns)
