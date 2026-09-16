# MPLADS Data Cleaning & ETL Methodology Documentation (Stage 2)

**Pipeline Version:** 2.0  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 2 - Data Cleaning, Missingness Classification & ETL  

---

## 1. Executive Summary

Stage 2 transforms raw MPLADS datasets into cleaned, analysis-ready formats stored in `data/processed/`. The ETL pipeline enforces strict data governance rules: zero synthetic data fabrication, zero artificial imputation of missing values, strict byte-level duplicate removal, and transparent classification of data quality anomalies.

---

## 2. Data Cleaning Rules & Transformations

### 2.1 Currency & Monetary Field Cleaning
Raw financial values across all datasets contain currency formatting artifacts such as the Rupee symbol (`₹`), thousands separators (commas `,`), and trailing whitespace.

**Transformation Rules:**
- Stripped `₹` symbols and commas using string regex replacement.
- Parsed monetary attributes directly into 64-bit floating-point numeric types (`float64`).
- Standardized field names (e.g. `Recommended Amount (₹)` -> `recommended_amount`, `Final Amount (₹)` -> `final_amount`, `Expenditure Amount (₹)` -> `expenditure_amount`, `Allocated Amount (₹)` -> `allocated_amount`, `Total Expenditure (₹)` -> `total_expenditure`, `Unspent Amount (₹)` -> `unspent_amount`).

### 2.2 Date Standardization & Temporal Parsing
Raw date columns contain ISO timestamp strings formatted with millisecond and UTC indicators (e.g., `'2025-02-14T00:00:00.000Z'`).

**Transformation Rules:**
- Converted date attributes (`Recommendation Date`, `Completed Date`, `Expenditure Date`) into ISO 8601 calendar strings (`YYYY-MM-DD`).
- Derived calendar year fields (`recommendation_year`, `completed_year`, `expenditure_year`).
- Evaluated temporal coverage: datasets span active observations from mid-2023 to mid-2026.

---

## 3. Duplicate Detection & Removal Strategy

| Dataset | Total Rows | Exact Duplicates Identified | Action Taken | Rationale |
| :--- | :---: | :---: | :--- | :--- |
| `expenditures.csv` | 107,683 | 32,185 | **32,185 Rows Deleted** | Confirmed exact byte-for-byte duplicate payment transaction records. |
| `recommended_works.csv` | 83,968 | 0 (15 duplicate `Work ID`s) | **0 Rows Deleted** | 15 duplicate `Work ID`s represent separate recommendations recorded under identical legacy IDs. Retained without deletion per raw immutability rules. |
| `completed_works.csv` | 43,735 | 0 | **0 Rows Deleted** | No exact row duplicates or duplicate `Work ID`s detected. |
| `mp_summary.csv` | 774 | 0 | **0 Rows Deleted** | Clean primary key entity set per MP. |

---

## 4. Missingness Analysis & Classification Framework

Rather than blindly imputing missing values with zeros or arbitrary strings, missing values were systematically categorized into four statistical missingness mechanisms:

1. **MCAR (Missing Completely At Random):**
   - *Example:* Minor text missingness in `work_description` (50 in recommended works, 85 in completed works).
   - *Mechanism:* Sporadic data entry omission independent of project parameters.
2. **MAR (Missing At Random):**
   - *Example:* Missing `category` in completed works (5 records).
   - *Mechanism:* Missingness correlated with administrative data entry protocols across specific IDAs.
3. **MNAR (Missing Not At Random):**
   - *Example:* `average_rating` in `completed_works.csv` (missing in 43,731 out of 43,735 records / 99.99%).
   - *Mechanism:* Rating scores are voluntarily submitted by citizens or inspectors upon completion; absence reflects unsubmitted feedback rather than zero rating.
4. **Structural Missingness:**
   - *Example:* `completed_date` and `final_amount` in `projects_clean.csv` (missing for 83,528 recommended-only works).
   - *Mechanism:* Inherent to project lifecycle status. Uncompleted recommended projects naturally lack completion dates and final amounts.

Detailed missingness percentages and classifications for all 84 audited columns are published in [`data/processed/data_quality_report.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/processed/data_quality_report.csv).

---

## 5. Dataset Relationships & Unification Strategy

### 5.1 Unified Projects Dataset (`projects_clean.csv`)
- **Primary Key:** `work_id`
- **Linking Strategy:** Full outer join between `recommended_works` and `completed_works`.
- **Status Classification:**
  - `Recommended` (83,528 records): Present only in recommended works.
  - `Completed` (43,295 records): Present only in completed works.
  - `Recommended & Completed` (440 records): Successfully matched across recommendation and completion logs.
- **Total Unified Rows:** 127,263

### 5.2 Expenditure Relationship Boundaries (`expenditures_clean.csv`)
- **Key Schema Observation:** Raw expenditure records do not contain a `Work ID` foreign key.
- **Data Governance Boundary:** To avoid creating false linkages or arbitrary fuzzy matching, expenditure records are preserved at their native grain (`MP Name`, `Constituency`, `State`, `Vendor`, `IDA`). Aggregated vendor and MP level financial linking will be performed downstream in Stage 3 feature engineering.

---

## 6. COVID-Era (2020–2022) Handling

During the COVID-19 pandemic (specifically Fiscal Years 2020-21 and 2021-22), MPLADS fund allocations were temporarily suspended/diverted by the Government of India for pandemic response.

**Handling Protocol:**
- Added boolean schema flag `is_covid_era` across processed datasets.
- Current active dataset observations cover mid-2023 through mid-2026.
- Future longitudinal or historical analyses must isolate COVID-era periods to prevent false anomaly signals caused by administrative scheme suspension.

---

## 7. Processed Datasets Manifest

- [`projects_clean.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/processed/projects_clean.csv) (127,263 rows, 17 columns, 31.5 MB)
- [`expenditures_clean.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/processed/expenditures_clean.csv) (75,498 rows, 11 columns, 18.8 MB)
- [`mp_summary_clean.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/processed/mp_summary_clean.csv) (774 rows, 13 columns, 83.5 KB)
- [`data_quality_report.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/processed/data_quality_report.csv) (84 audited columns, 6.8 KB)
