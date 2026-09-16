# MPLADS Explainable Multi-Factor Risk Scoring Methodology (Stage 5)

**Pipeline Version:** 5.2  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 5 — Explainable Multi-Factor Risk Scoring & Priority Categorization  

---

## 1. Overview & Objectives

Stage 5 synthesizes independent anomaly signals across domain feature vectors (Financial, Temporal, Vendor, Isolation Forest, Statistical Baselines, NLP Text Similarity, and **MP Works Completion Risk**) into a unified, transparent, and explainable **Project Priority Risk Score** ($0.00 – 100.00$).

The primary objectives of this stage are to:
1. Normalize domain risk components onto a standardized $0 – 100$ scale before weighting.
2. Apply an explicit, fully transparent 5-factor weighted combination formula:
   $$\text{Final Risk Score} = (0.30 \times \text{Financial}) + (0.20 \times \text{Temporal}) + (0.25 \times \text{Isolation Forest}) + (0.15 \times \text{Vendor Risk}) + (0.10 \times \text{MP Works Risk})$$
3. Categorize projects into configurable risk priority bands (`LOW`: $0–39$, `MEDIUM`: $40–69$, `HIGH`: $70–100$).
4. Generate **elaborated, evidence-grounded bullet-point explanations** detailing specific anomaly triggers (including low completion rate warnings).
5. Export results to [`data/processed/final_risk_scores.csv`](file:///d:/MPLADS%28SIH%29/data/processed/final_risk_scores.csv).

---

## 2. Component Normalization Architecture ($0 – 100$)

Each domain risk signal is mapped onto a uniform $0.00 – 100.00$ continuous scale:

1. **`financial_score`**:
   $$\text{financial\_score} = \text{CLIP}_{0}^{100}\left(0.6 \times \text{Percentile} + 0.4 \times \text{CLIP}_{0}^{100}(Z \times 10)\right)$$
2. **`temporal_score`**:
   $$\text{temporal\_score} = \begin{cases} 100.0 & \text{if invalid date sequence} \\ \text{CLIP}_{0}^{100}\left(\frac{\text{YoY Change}}{10.0}\right) & \text{otherwise} \end{cases}$$
3. **`vendor_score`**:
   $$\text{vendor\_score} = \text{CLIP}_{0}^{100}\left(0.5 \times S_{\text{max\_vendor}} + 0.3 \times S_{\text{pending}} + 0.2 \times S_{\text{repeat}}\right)$$
4. **`isolation_forest_score`**:
   $$\text{isolation\_forest\_score} = \text{CLIP}_{0}^{100}\left(\text{Normalized Anomaly Score} \times 100.0\right)$$
5. **`mp_works_risk_score` (MP Execution Risk)**:
   $$\text{mp\_works\_risk\_score} = \text{CLIP}_{0}^{100}\left(100.0 - \text{Completion Rate \%}\right)$$
   - Inverse of constituency completion rate: low completion rates (e.g. $< 50\%$) reflect elevated project backlog risk.

---

## 3. Transparent Weighted Combination & Risk Categories

### 3.1 Aggregation Formula
$$\text{Final Risk Score} = (0.30 \times \text{Financial}) + (0.20 \times \text{Temporal}) + (0.25 \times \text{Isolation Forest}) + (0.15 \times \text{Vendor Risk}) + (0.10 \times \text{MP Works Risk})$$

### 3.2 Configurable Risk Categories

| Risk Category | Score Threshold Band | Operational Meaning |
| :--- | :---: | :--- |
| **`LOW`** | $0.00 \le \text{Score} \le 39.99$ | Standard project execution within normal baseline bounds ($83,604$ projects / $65.7\%$) |
| **`MEDIUM`** | $40.00 \le \text{Score} \le 69.99$ | Moderate multi-factor statistical deviation ($43,506$ projects / $34.2\%$) |
| **`HIGH`** | $70.00 \le \text{Score} \le 100.00$ | Severe multi-factor anomaly requiring priority administrative audit ($153$ projects / $0.12\%$) |
