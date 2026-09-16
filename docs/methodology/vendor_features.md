# MPLADS Vendor and Administrative Feature Engineering Methodology (Stage 3C)

**Pipeline Version:** 3.3  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 3C - Vendor & Administrative Feature Engineering  

---

## 1. Executive Summary & Terminology Governance

Stage 3C quantifies vendor transaction volume, concentration ratios, payment disbursal distributions, and administrative patterns across **27,927 unique vendors** in the MPLADS payment records.

### ⚠️ Strict Non-Accusatory Terminology Policy
Per analytical governance rules, **no vendor, Member of Parliament, or administrative body is labeled as fraudulent, corrupt, or illegal**. 

All analytical signals use strictly neutral terminology:
- `anomaly`
- `unusual pattern`
- `requires verification`

---

## 2. Feature Specifications & Mathematical Definitions

### 2.1 Transaction Volume & Repeat Metrics
1. **`transaction_count`**: Total number of payment transactions recorded per vendor.
2. **`total_vendor_expenditure`**: Total monetary sum ($\sum \text{Expenditure Amount}$) disbursed to vendor.
3. **`average_vendor_transaction`**: $\mu(\text{Expenditure Amount})$ per vendor.
4. **`median_vendor_transaction`**: $\text{Median}(\text{Expenditure Amount})$ per vendor.
5. **`vendor_repeat_rate`**: $\frac{\text{transaction\_count} - 1}{\text{transaction\_count}}$ (quantifies transaction recurrence beyond single one-off disbursements).

### 2.2 Payment Status & Disbursal Distributions
6. **`pending_payment_ratio`**: $\frac{\text{Count}(\text{Payment In-Progress})}{\text{transaction\_count}}$
7. **`payment_status_distribution`**: $\frac{\text{Count}(\text{Payment Success})}{\text{transaction\_count}}$ (Ratio of successfully settled payments).

### 2.3 Concentration & Market Share Ratios
8. **`vendor_share_of_constituency_expenditure`**: Vendor's maximum share of total expenditure within any single constituency ($\max \frac{\text{Vendor Exp in Constituency}}{\text{Total Constituency Exp}}$).
9. **`vendor_share_of_MP_expenditure`**: Vendor's maximum share of total expenditure under any single MP ($\max \frac{\text{Vendor Exp under MP}}{\text{Total MP Exp}}$).

### 2.4 Frequency & YoY Trends
10. **`transaction_frequency`**: $\frac{\text{transaction\_count}}{\text{active\_months\_count}}$ (Transactions per active calendar month).
11. **`year_over_year_vendor_change`**: $\frac{\text{Exp}_{2026} - \text{Exp}_{2025}}{\text{Exp}_{2025}} \times 100\%$ (Percentage annual spending change).

---

## 3. Audited Entity Counts & Entity Breakdown

- **Total Unique Vendors:** 27,927
- **Total Unique MPs:** 699
- **Total Unique Constituencies:** 530
- **Total Unique IDAs:** 763
- **Total Payment Transactions Processed:** 75,498

---

## 4. Summary Statistics of Vendor Feature Matrix ($N = 27,927$ Vendors)

| Feature Name | Usable Records | Null Records | Mean | Std | Min | 50% (Median) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `transaction_count` | 27,927 | 0 | 2.70 | 6.56 | 1.00 | 1.00 | 268.00 |
| `total_vendor_expenditure` (₹) | 27,927 | 0 | ₹1,160,272 | ₹4,645,865 | ₹58.00 | ₹250,000 | ₹202,457,098 |
| `average_vendor_transaction` (₹) | 27,927 | 0 | ₹420,550 | ₹728,041 | ₹58.00 | ₹250,000 | ₹32,562,250 |
| `median_vendor_transaction` (₹) | 27,927 | 0 | ₹414,816 | ₹726,911 | ₹58.00 | ₹250,000 | ₹32,562,250 |
| `vendor_repeat_rate` | 27,927 | 0 | 0.28 | 0.33 | 0.00 | 0.00 | 1.00 |
| `pending_payment_ratio` | 27,927 | 0 | 0.03 | 0.17 | 0.00 | 0.00 | 1.00 |
| `payment_status_distribution` | 27,927 | 0 | 0.97 | 0.17 | 0.00 | 1.00 | 1.00 |
| `vendor_share_of_constituency` | 27,927 | 0 | 0.02 (2.0%) | 0.05 | 0.00 | 0.004 | 1.00 (100%) |
| `vendor_share_of_MP` | 27,927 | 0 | 0.02 (2.0%) | 0.06 | 0.00 | 0.004 | 1.00 (100%) |
| `transaction_frequency` | 27,927 | 0 | 1.31 | 0.95 | 1.00 | 1.00 | 41.00 |
| `year_over_year_vendor_change` (%) | 13,867 | 14,060 | +274.19% | 31,716.96 | -100.00% | +12.50% | +3,725,525% |

---

## 5. Output Feature Matrix Location

- [`data/features/vendor_features.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/features/vendor_features.csv) (27,927 rows, 16 columns)
