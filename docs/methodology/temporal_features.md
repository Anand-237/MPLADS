# MPLADS Temporal Feature Engineering Methodology (Stage 3B)

**Pipeline Version:** 3.2  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 3B - Temporal Feature Engineering  

---

## 1. Executive Summary

Stage 3B converts raw MPLADS recommendation, completion, and expenditure dates into structured numerical features, calculates project duration metrics, and performs date sequence anomaly detection.

Zero dates were fabricated or silently overwritten. Impossible date sequences (e.g. completion date preceding recommendation date) were preserved, flagged (`is_impossible_date_sequence`), and reported transparently.

---

## 2. Feature Specifications & Mathematical Definitions

### 2.1 Extract Components
- `recommendation_year`: Calendar year extracted from `recommendation_date`.
- `recommendation_month`: Calendar month (1–12) extracted from `recommendation_date`.
- `expenditure_year`: Median expenditure year for the sponsoring MP/Constituency.
- `expenditure_month`: Month of payment disbursemnt (recorded in expenditures dataset).

### 2.2 Duration & Delay Metrics
- `completion_delay_days`: $\text{completed\_date} - \text{recommendation\_date}$
- `completion_delay_months`: $\frac{\text{completion\_delay\_days}}{30.4375}$
- `project_duration_days` & `project_duration_months`: Computed for linked projects where completion dates exist.
- `recommendation_to_start_days`: Unfabricated feature set to `NaN` because `Start Date` is absent in raw MPLADS CSV schemas.

### 2.3 Baseline Deviation & YoY Trends
- `duration_deviation_from_historical_baseline`: Z-score deviation of valid positive project completion delay relative to State baseline mean and std ($Z = \frac{\text{delay} - \mu_{\text{State}}}{\sigma_{\text{State}}}$).
- `year_over_year_change`: Percentage change in recommended project volume per State between 2024 and 2025.

---

## 3. Date Sequence Anomaly & Validation Report

### 3.1 Date Validation Findings
- **Unparseable / Invalid ISO Dates:** 0 (All dates in input datasets are valid ISO 8601 strings).
- **Missing `recommendation_date`:** 43,295 (Structural — completed-only records).
- **Missing `completed_date`:** 83,528 (Structural — recommended-only projects).
- **Linked Projects with Both Dates:** 440 records.

### 3.2 Impossible Date Sequences & Negative Durations
- **Impossible Date Sequences (`completed_date < recommendation_date`):** **124 records** (28.18% of linked records).
- **Negative Durations:** **124 records** (Minimum duration: -842 days).
- **Handling Protocol:** These 124 records were flagged (`is_impossible_date_sequence = True`, `is_negative_duration = True`) without silent date modification. These represent severe administrative logging anomalies ideal for downstream anomaly detection.

### 3.3 COVID-Era Flagging
- **`is_covid_era`:** Boolean flag for 2020-21 and 2021-22 financial periods. (0 records in current 2023-2026 dataset; schema flag provided for longitudinal comparability).

---

## 4. Summary Statistics of Temporal Feature Matrix

| Feature Name | Usable Records | Null Records | Mean | Std | Min | 50% (Median) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `recommendation_year` | 83,968 | 43,295 | 2025.38 | 0.64 | 2023.00 | 2025.00 | 2026.00 |
| `recommendation_month` | 83,968 | 43,295 | 6.21 | 3.16 | 1.00 | 6.00 | 12.00 |
| `expenditure_year` | 126,617 | 646 | 2025.57 | 0.51 | 2024.00 | 2026.00 | 2026.00 |
| `completion_delay_days` | 440 | 126,823 | 103.63 | 439.68 | -842.00 | 265.00 | 767.00 |
| `completion_delay_months` | 440 | 126,823 | 3.40 | 14.45 | -27.66 | 8.71 | 25.20 |
| `project_duration_days` | 440 | 126,823 | 103.63 | 439.68 | -842.00 | 265.00 | 767.00 |
| `project_duration_months` | 440 | 126,823 | 3.40 | 14.45 | -27.66 | 8.71 | 25.20 |
| `recommendation_to_start_days` | 0 | 127,263 | N/A | N/A | N/A | N/A | N/A |
| `duration_deviation` (Z) | 312 | 126,951 | -0.00 | 0.97 | -2.13 | -0.27 | 3.65 |
| `year_over_year_change` (%) | 126,822 | 441 | +685.75% | 459.46 | +60.00% | +571.43% | +3100.00% |

---

## 5. Output Datasets

- [`data/features/temporal_features.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/features/temporal_features.csv) (127,263 rows, 19 columns)
