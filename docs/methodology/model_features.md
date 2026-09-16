# MPLADS Isolation Forest Model Feature Matrix Methodology (Stage 4A)

**Pipeline Version:** 4.1  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 4A — Isolation Forest Unsupervised Model Feature Matrix Engineering  

---

## 1. Overview & Objectives

Stage 4A constructs a unified, standardized, numerical feature matrix (`data/features/isolation_forest_features.csv`) for downstream unsupervised anomaly detection via Isolation Forest.

The primary objectives of this stage are to:
1. Integrate domain feature vectors across financial, temporal, text/NLP, and vendor administrative risk dimensions for all 127,263 projects.
2. Enforce strict isolation between metadata identifiers (`work_id`, `project_status`, `state`, `constituency`) and model training features.
3. Audit for infinite values, extreme outliers, zero-variance features, and collinear feature pairs.
4. Impute missing values using domain-grounded median standards without fabricating synthetic records.
5. Export a machine-readable metadata dictionary (`data/features/feature_dictionary.csv`) documenting every feature's provenance, formula, missing percentage, and model inclusion status.

---

## 2. Feature Selection & Domain Representation

The feature matrix combines 21 numerical features spanning 4 core operational domains:

### 2.1 Financial Features
- **`effective_amount`**: Sanctioned or completed project expenditure in Rupees ($N = 127,263$).
- **`cost_deviation`**: Z-score of project cost relative to State baseline cohort ($N = 127,263$).
- **`state_cost_percentile`**: Percentile rank of effective cost within State cohort ($N = 127,263$).
- **`constituency_cost_percentile`**: Percentile rank of effective cost within Constituency cohort ($N = 127,263$, 38 missing imputed to median `50.6579`).
- **`release_ratio`**: Disbursal ratio of total expenditure against allocated funds for the sponsoring MP ($N = 127,263$).
- **`recommendation_sanction_ratio`**: Ratio of recommended cost relative to state average recommendation ($N = 127,263$, 43,295 missing imputed to `1.0` baseline).

### 2.2 Temporal Features
- **`is_impossible_date_sequence`**: Binary indicator ($0/1$) flagging completion dates occurring before recommendation dates ($N = 127,263$).
- **`is_negative_duration`**: Binary indicator ($0/1$) flagging negative completion durations (duplicate of date sequence check; flagged `used_for_model = False`).
- **`is_covid_era`**: Binary indicator ($0/1$) for COVID window (zero variance in corpus; flagged `used_for_model = False`).
- **`year_over_year_change`**: Percentage change in constituency project recommendation volume ($N = 127,263$, 441 missing imputed to `0.0`).

### 2.3 Natural Language Processing (NLP) Features
- **`description_length`**: Raw character length of work description text ($N = 127,263$).
- **`word_count`**: Total word count of raw work description text ($N = 127,263$).
- **`tfidf_mean`**: Mean TF-IDF vector weight across 10,000 unigram/bigram terms ($N = 127,263$).
- **`tfidf_max`**: Maximum TF-IDF feature weight for dominant description term ($N = 127,263$).
- **`nearest_similarity_score`**: Maximum cosine text similarity score with any other project in constituency cohort ($N = 127,263$).
- **`similar_project_count`**: Total count of cohort projects exhibiting cosine text similarity $\ge 0.85$ ($N = 127,263$).

### 2.4 Vendor Cohort Risk Features
- **`constituency_max_vendor_share`**: Maximum vendor expenditure share in constituency ($N = 127,263$, 772 missing imputed to median `0.1676`).
- **`constituency_avg_vendor_transaction`**: Mean vendor transaction size in constituency ($N = 127,263$, 772 missing imputed to median `₹446,440.89`).
- **`constituency_vendor_count`**: Total active unique vendors in constituency ($N = 127,263$, 772 missing imputed to `0.0`).
- **`constituency_avg_vendor_repeat_rate`**: Mean vendor repeat transaction rate in constituency ($N = 127,263$, 772 missing imputed to `0.0`).
- **`constituency_pending_payment_ratio`**: Mean pending payment ratio across constituency vendors ($N = 127,263$, 772 missing imputed to `0.0`).

---

## 3. Data Cleaning, Infinite Values & Imputation Audit

1. **Infinite Value Audit**: Zero `inf` or `-inf` values were found across all numerical feature vectors.
2. **Missing Value Imputation**: Pre-imputation missingness was resolved deterministically using domain standards:
   - `recommendation_sanction_ratio`: $34.02\%$ missing ($43,295$ projects) $\rightarrow$ Imputed with `1.0` (state baseline norm).
   - `constituency_cost_percentile`: $0.03\%$ missing ($38$ projects) $\rightarrow$ Imputed with cohort median `50.6579`.
   - `year_over_year_change`: $0.35\%$ missing ($441$ projects) $\rightarrow$ Imputed with `0.0` (zero growth baseline).
   - Constituency vendor metrics: $0.61\%$ missing ($772$ projects) $\rightarrow$ Imputed with overall cohort medians / 0 count.
3. **Target Leakage Prevention**: No target labels, post-hoc fraud indicators, or future audit results are included.

---

## 4. Multicollinearity & Correlation Analysis

High correlation pairs ($|r| \ge 0.85$) were audited across the 21 numerical features:

1. **`effective_amount` $\leftrightarrow$ `cost_deviation` ($r = 0.8855$)**:
   - *Rationale*: Cost deviation is Z-score scaled relative to State baseline. Retained as decision trees handle monotonic linear scalings gracefully.
2. **`is_impossible_date_sequence` $\leftrightarrow$ `is_negative_duration` ($r = 1.0000$)**:
   - *Rationale*: Exact duplicate binary flag. `is_negative_duration` flagged `used_for_model = False`.
3. **`description_length` $\leftrightarrow$ `word_count` ($r = 0.9671$)**:
   - *Rationale*: Character vs word length representation. Both retained as tree splits partition non-linearly.

---

## 5. Feature Dictionary Manifest

| Feature Name | Source Column | Description | Formula | Data Type | Missing % | Used For Model |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `work_id` | `work_id` | Unique project identifier | Raw identifier string | string | 0.00% | False |
| `project_status` | `project_status` | Project status (Recommended vs Completed) | Raw category string | string | 0.00% | False |
| `state` | `state` | State / Union Territory name | Raw geographical string | string | 0.00% | False |
| `constituency` | `constituency` | Constituency name | Raw geographical string | string | 0.00% | False |
| `effective_amount` | `final_amount / recommended_amount` | Sanctioned or completed cost in Rupees | COALESCE(final, recommended) | float64 | 0.00% | **True** |
| `cost_deviation` | `effective_amount` | Z-score of cost within State cohort | (cost - mean_state) / std_state | float64 | 0.00% | **True** |
| `state_cost_percentile` | `effective_amount` | Percentile rank within State cohort | Percentile(cost \| State) | float64 | 0.00% | **True** |
| `constituency_cost_percentile` | `effective_amount` | Percentile rank within Constituency cohort | Percentile(cost \| Constituency) | float64 | 0.03% | **True** |
| `release_ratio` | `total_expenditure / allocated_amount` | Sponsoring MP fund utilization ratio | total_expenditure / allocated | float64 | 0.00% | **True** |
| `recommendation_sanction_ratio` | `recommended_amount` | Ratio of recommendation cost to state mean | recommended / mean_state(rec) | float64 | 34.02% | **True** |
| `is_impossible_date_sequence` | `completed_date, recommendation_date` | Binary flag for invalid date sequence | IF(completed < recommended, 1, 0) | int64 | 0.00% | **True** |
| `is_negative_duration` | `completed_date, recommendation_date` | Binary flag for negative duration | IF(duration < 0, 1, 0) | int64 | 0.00% | False |
| `is_covid_era` | `recommendation_year / completed_year` | Binary flag for COVID window (zero var) | IF(year IN (2020, 2021), 1, 0) | int64 | 0.00% | False |
| `year_over_year_change` | `project_count` | YoY change in constituency project volume | ((count_t - count_t-1) / count_t-1) * 100 | float64 | 0.35% | **True** |
| `description_length` | `work_description` | Character length of work description | LEN(work_description) | int64 | 0.00% | **True** |
| `word_count` | `work_description` | Word count of work description | WORD_COUNT(description) | int64 | 0.00% | **True** |
| `tfidf_mean` | `work_description` | Mean TF-IDF weight across vocabulary | MEAN(TFIDF_Vector) | float64 | 0.00% | **True** |
| `tfidf_max` | `work_description` | Max TF-IDF weight for dominant term | MAX(TFIDF_Vector) | float64 | 0.00% | **True** |
| `nearest_similarity_score` | `work_description` | Max cosine similarity with cohort project | MAX_j(CosineSim(d_i, d_j)) | float32 | 0.00% | **True** |
| `similar_project_count` | `work_description` | Count of cohort projects with similarity $\ge 0.85$ | SUM_j(CosineSim >= 0.85) | int32 | 0.00% | **True** |
| `constituency_max_vendor_share` | `vendor_share_of_constituency_expenditure` | Max vendor share in constituency | MAX(vendor_const_share) | float64 | 0.61% | **True** |
| `constituency_avg_vendor_transaction` | `average_vendor_transaction` | Mean vendor transaction size in constituency | MEAN(vendor_avg_tx) | float64 | 0.61% | **True** |
| `constituency_vendor_count` | `vendor` | Total unique active vendors in constituency | COUNT(DISTINCT vendor) | float64 | 0.61% | **True** |
| `constituency_avg_vendor_repeat_rate` | `vendor_repeat_rate` | Mean vendor repeat rate in constituency | MEAN(repeat_rate) | float64 | 0.61% | **True** |
| `constituency_pending_payment_ratio` | `pending_payment_ratio` | Mean pending payment ratio in constituency | MEAN(pending_ratio) | float64 | 0.61% | **True** |

---

## 6. Output Datasets Manifest

1. [`data/features/isolation_forest_features.csv`](file:///d:/MPLADS%28SIH%29/data/features/isolation_forest_features.csv)
   - **Scale**: 127,263 rows $\times$ 25 columns (4 metadata + 21 numerical model features).
2. [`data/features/feature_dictionary.csv`](file:///d:/MPLADS%28SIH%29/data/features/feature_dictionary.csv)
   - **Scale**: 25 rows $\times$ 7 columns metadata dictionary.
