# MPLADS Human-in-the-Loop Verification Methodology (Stage 9)

**Pipeline Version:** 9.0  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 9 — Human-in-the-Loop Verification & Audit Trail Management  

---

## 1. Overview & Objectives

Stage 9 establishes a formal **Human-in-the-Loop (HITL) audit verification framework** allowing human administrative auditors to investigate AI-generated anomaly signals, record official audit decisions, and build a permanent audit trail.

The primary objectives of this stage are to:
1. Provide human reviewers with a 7-panel investigation workspace covering all domain anomaly vectors.
2. Enable 4 official human reviewer actions: `Verified Normal`, `Needs Further Review`, `Evidence Insufficient`, `Manual Investigation Required`.
3. Capture mandatory auditor metadata: `review_status`, `reviewer_notes`, `reviewer_id`, and `review_timestamp`.
4. Guarantee **strict immutability of AI predictions**: Model scores (`final_risk_score`, `isolation_forest_score`, `risk_category`) remain untouched in system pipeline outputs.
5. Persist human review records in a dedicated, decoupled audit log: [`data/audit/review_log.csv`](file:///d:/MPLADS%28SIH%29/data/audit/review_log.csv).

> [!IMPORTANT]
> **Neutral Policy Compliance**: AI model outputs serve strictly as administrative prioritizations for human review. The application strictly **PROHIBITS** automated labeling of fraud or illegal behavior.

---

## 2. 7-Panel Investigation Workspace Architecture

For every flagged project selected in Section 6 of the Streamlit Audit Dashboard (`app.py`), the auditor is presented with 7 comprehensive evidence panels:

```
+-----------------------------------------------------------------------------------+
|               FLAGGED PROJECT AUDITOR INVESTIGATION WORKSPACE (app.py)            |
+-----------------------------------------------------------------------------------+
|  1. 📌 Project Details    - Work ID, MP Name, State, Constituency, Status, Dates  |
|  2. 💰 Financial Anomaly  - Effective Amount, Cohort μ/P50, Z-Score, Percentile   |
|  3. ⏱️ Temporal Anomaly   - Duration, Date Sequence Flags, YoY Volume Growth %    |
|  4. 🏪 Vendor Anomaly     - Max Vendor Share %, Pending Payment %, Repeat Rate %  |
|  5. 📄 NLP Similarity     - Nearest Cosine Similarity %, Similar Candidate Count  |
|  6. 📈 Benchmark Chart    - Selected Value vs Cohort Median & P99 Ceiling         |
|  7. 🤖 Isolation Forest   - ML Tree Isolation Severity Score (0-100)              |
+-----------------------------------------------------------------------------------+
```

---

## 3. Human Reviewer Action Taxonomy

Auditors can record 4 official review decisions:

| Reviewer Action | Operational Meaning | Recommended Administrative Action |
| :--- | :--- | :--- |
| **`Verified Normal`** | Document audit confirms standard execution | Clear from active verification queue |
| **`Needs Further Review`** | Substantial statistical deviation warrants physical inspection | Escalate to District Collector / Auditor General |
| **`Evidence Insufficient`** | Missing invoice, sanction, or completion records | Request supplementary documentation from agency |
| **`Manual Investigation Required`** | Severe multi-factor anomaly detected | Initiate formal administrative inquiry |

---

## 4. Audit Log Data Schema (`data/audit/review_log.csv`)

Human reviewer decisions are appended to [`data/audit/review_log.csv`](file:///d:/MPLADS%28SIH%29/data/audit/review_log.csv):

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `log_id` | String | Unique audit entry identifier (e.g. `AUDIT_000001`) |
| `work_id` | Integer | Project identifier |
| `review_status` | String | Official reviewer action chosen by auditor |
| `reviewer_notes` | Text | Free-text auditor findings and justification notes |
| `reviewer_id` | String | Auditor officer name or badge ID |
| `review_timestamp` | String | UTC ISO 8601 timestamp of review entry |
| `ai_risk_score_at_review` | Float | Immutable snapshot of AI `final_risk_score` at time of review |
| `ai_risk_category_at_review` | String | Immutable snapshot of AI `risk_category` at time of review |

---

## 5. System Execution Verification

- **Service Script**: [`scripts/audit/human_verification_service.py`](file:///d:/MPLADS%28SIH%29/scripts/audit/human_verification_service.py)
- **Streamlit Application**: [`app.py`](file:///d:/MPLADS%28SIH%29/app.py)
- **Audit Log Output**: [`data/audit/review_log.csv`](file:///d:/MPLADS%28SIH%29/data/audit/review_log.csv)
