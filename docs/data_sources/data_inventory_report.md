# MPLADS Dataset Inventory Report (Stage 1)

**Acquisition Date (UTC):** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 1 - Data Ingestion & Project Structure  

---

## 1. Overview & Raw File Manifest

All raw datasets were copied into `data/raw/` preserving original contents without modification. Every ingestion run generates cryptographic checksums and records audit logs in `data/raw/ingestion.log` and metadata in `data/raw/ingestion_manifest.json`.

| Filename | File Size | Row Count | Column Count | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| [`recommended_works.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/raw/recommended_works.csv) | 18,957,933 bytes (18.08 MB) | 83,968 | 9 | `761b3661a463f1d589d5e44498888ede201bb8fe085f2623ff0c0a34b12e2a65` |
| [`completed_works.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/raw/completed_works.csv) | 10,536,497 bytes (10.05 MB) | 43,735 | 11 | `66fcdbfeb6f859629c0c94850fb1efc654a0f440340a4ec55515681ae686b164` |
| [`expenditures.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/raw/expenditures.csv) | 23,247,025 bytes (22.17 MB) | 107,683 | 9 | `8fba3f0dc7870ec00681638c39550b301f0b0923e9e3e6b9bbfb453351ca77d9` |
| [`mp_summary.csv`](file:///c:/Users/sharm/OneDrive/Desktop/MPLADS(SIH)/data/raw/mp_summary.csv) | 79,313 bytes (0.08 MB) | 774 | 13 | `d66751d4b579df95d372f08e037967a493ee1e49829598ef9e930f0ce887b4f0` |

---

## 2. Dataset Schema & Column Details

### 2.1 `recommended_works.csv`
- **Total Rows:** 83,968
- **Total Columns:** 9

| # | Column Name | Data Type | Missing Count | Missing % | Description / Notes |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Work ID` | `int64` | 0 | 0.00% | Unique identifier for recommended work item |
| 2 | `Work Description` | `object` (string) | 50 | 0.06% | Text description of the proposed project |
| 3 | `MP Name` | `object` (string) | 0 | 0.00% | Name of recommending Member of Parliament |
| 4 | `Constituency` | `object` (string) | 0 | 0.00% | Parliamentary constituency |
| 5 | `State` | `object` (string) | 0 | 0.00% | State / Union Territory |
| 6 | `Recommended Amount (₹)` | `float64` | 0 | 0.00% | Financial amount recommended in INR |
| 7 | `Recommendation Date` | `object` (string) | 0 | 0.00% | Date of recommendation (YYYY-MM-DD) |
| 8 | `Has Images` | `bool` | 0 | 0.00% | Flag indicating attached proof images |
| 9 | `IDA` | `object` (string) | 0 | 0.00% | Implementing Agency identifier/name |

---

### 2.2 `completed_works.csv`
- **Total Rows:** 43,735
- **Total Columns:** 11

| # | Column Name | Data Type | Missing Count | Missing % | Description / Notes |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Work ID` | `int64` | 0 | 0.00% | Foreign key / Unique identifier matching recommended works |
| 2 | `Work Description` | `object` (string) | 85 | 0.19% | Text description of completed work |
| 3 | `Category` | `object` (string) | 5 | 0.01% | Sector classification (e.g. Roads, Water, Education) |
| 4 | `MP Name` | `object` (string) | 0 | 0.00% | Name of Member of Parliament |
| 5 | `Constituency` | `object` (string) | 0 | 0.00% | Parliamentary constituency |
| 6 | `State` | `object` (string) | 0 | 0.00% | State / Union Territory |
| 7 | `Final Amount (₹)` | `float64` | 0 | 0.00% | Actual cost incurred upon completion |
| 8 | `Completed Date` | `object` (string) | 0 | 0.00% | Date of completion (YYYY-MM-DD) |
| 9 | `Has Images` | `bool` | 0 | 0.00% | Image proof availability |
| 10 | `Average Rating` | `float64` | 43,731 | 99.99% | Rating score (sparsely populated field) |
| 11 | `IDA` | `object` (string) | 0 | 0.00% | Implementing Agency identifier/name |

---

### 2.3 `expenditures.csv`
- **Total Rows:** 107,683
- **Total Columns:** 9

| # | Column Name | Data Type | Missing Count | Missing % | Description / Notes |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `MP Name` | `object` (string) | 0 | 0.00% | Member of Parliament name |
| 2 | `Constituency` | `object` (string) | 0 | 0.00% | Parliamentary constituency |
| 3 | `State` | `object` (string) | 0 | 0.00% | State / Union Territory |
| 4 | `Work Description` | `object` (string) | 0 | 0.00% | Specific work item description |
| 5 | `Vendor` | `object` (string) | 0 | 0.00% | Recipient vendor / contractor name |
| 6 | `IDA` | `object` (string) | 0 | 0.00% | Implementing Agency |
| 7 | `Expenditure Amount (₹)` | `float64` | 0 | 0.00% | Disbursed payment amount |
| 8 | `Expenditure Date` | `object` (string) | 0 | 0.00% | Date of transaction |
| 9 | `Payment Status` | `object` (string) | 0 | 0.00% | Status (e.g. Paid, Pending) |

---

### 2.4 `mp_summary.csv`
- **Total Rows:** 774
- **Total Columns:** 13

| # | Column Name | Data Type | Missing Count | Missing % | Description / Notes |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `MP Name` | `object` (string) | 0 | 0.00% | Name of Member of Parliament |
| 2 | `Constituency` | `object` (string) | 0 | 0.00% | Parliamentary constituency |
| 3 | `State` | `object` (string) | 0 | 0.00% | State / Union Territory |
| 4 | `Allocated Amount (₹)` | `float64` | 0 | 0.00% | Total fund allocated to MP |
| 5 | `Total Expenditure (₹)` | `float64` | 0 | 0.00% | Total expenditure disbursed |
| 6 | `Utilization %` | `float64` | 0 | 0.00% | Fund utilization percentage |
| 7 | `Completed Works` | `int64` | 0 | 0.00% | Count of completed works |
| 8 | `Recommended Works` | `int64` | 0 | 0.00% | Count of recommended works |
| 9 | `Completion Rate %` | `float64` | 0 | 0.00% | Work completion percentage |
| 10 | `Unspent Amount (₹)` | `float64` | 0 | 0.00% | Unutilized balance |
| 11 | `Transaction Count` | `int64` | 0 | 0.00% | Total payment transactions recorded |
| 12 | `Successful Payments` | `int64` | 0 | 0.00% | Completed payment count |
| 13 | `Pending Payments` | `int64` | 0 | 0.00% | Pending payment count |

---

## 3. Data Ingestion & Audit Rules Applied

1. **Raw Data Preservation:** Files in `data/raw/` are exact byte-for-byte copies of source files. No rows were removed, imputed, or modified.
2. **Missing Values Preserved:** Missing entries (such as 50 missing descriptions in `recommended_works.csv`, 85 in `completed_works.csv`, 5 in `Category`, and 43,731 missing values in `Average Rating`) were retained without artificial imputation.
3. **Audit Trails:** Ingestion events are logged with UTC timestamps, file hashes, and schema metadata in `data/raw/ingestion.log` and `data/raw/ingestion_manifest.json`.
