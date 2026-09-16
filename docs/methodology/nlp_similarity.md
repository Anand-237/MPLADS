# MPLADS Work Description NLP & Text Similarity Methodology (Stage 3D)

**Pipeline Version:** 3.4  
**Execution Date:** 2026-09-01  
**Project:** MPLADS AI Anomaly Detection System  
**Pipeline Stage:** Stage 3D — MPLADS Work Description Natural Language Processing (NLP) & Text Similarity Analysis  

---

## 1. Overview & Objectives

Stage 3D extracts numerical text features and detects potentially duplicated or highly similar MPLADS project work descriptions across historical project records.

The primary objectives of this stage are to:
1. Normalize and clean unstructured free-text work descriptions across 127,263 project entries.
2. Build a high-dimensional TF-IDF (Term Frequency - Inverse Document Frequency) vector space model representing text patterns.
3. Compute cosine similarity across projects within constituency/state cohorts using memory-efficient sparse block matrix multiplication.
4. Extract text scale metrics and similarity features (`nearest_similarity_score`, `similar_project_count`) for downstream anomaly detection models.
5. Retrieve candidate project pairs exhibiting text similarity $\ge 0.85$ into an isolated analytical table titled **"Potentially Similar Projects"**.

> [!IMPORTANT]
> **Neutral Terminology Compliance**: High text similarity reflects common phrasing, standardized tender templates, or repeated works. High similarity score is **NOT** proof of duplicate funding, fraud, or intentional misallocation.

---

## 2. Text Preprocessing & Cleaning Pipeline

Raw work descriptions are processed through a deterministic text cleaning pipeline:

1. **Lowercasing**: Converts all characters to lowercase to eliminate casing discrepancies (e.g. `Construction` vs `construction`).
2. **Character Filtering**: Retains alphanumeric characters and spaces (`isalnum() | isspace()`), stripping irrelevant punctuation, bullet points, special characters, and formatting symbols.
3. **Whitespace Normalization**: Collapses multi-space sequences, leading, and trailing whitespace into single spaces.
4. **Preservation of Meaningful Words**: Retains domain-specific terminology (e.g. `paver`, `borewell`, `community hall`, `solar light`, `cc road`).

---

## 3. TF-IDF Vector Space Model

The cleaned text corpus is transformed into a high-dimensional sparse matrix using `sklearn.feature_extraction.text.TfidfVectorizer`:

- **Vocabulary Limit (`max_features`)**: 10,000 top terms ranked by term frequency.
- **N-gram Range**: Unigrams and Bigrams (`ngram_range=(1, 2)`), capturing both single terms and two-word phrases (e.g. `solar_light`, `community_hall`, `cc_road`).
- **Stop Words Removal**: English stop words removed to filter out generic function words (`and`, `the`, `in`, `for`, `of`).
- **Minimum Document Frequency (`min_df`)**: 2 occurrences minimum across the corpus.

### Corpus Summary Metrics
- **Total Projects Loaded**: 127,263
- **Non-Null Work Descriptions**: 127,128 (135 empty/null descriptions assigned zero vectors)
- **Unique Work Descriptions**: 114,948
- **TF-IDF Matrix Shape**: $127,263 \times 10,000$ sparse CSR matrix

---

## 4. Scalable Cosine Similarity & Sparse Block Multiplication

To avoid an impractical $O(n^2)$ global matrix comparison ($127,263 \times 127,263 \approx 16.2 \text{ billion pairs}$), the pipeline employs **Sparse Cohort Matrix Multiplication**:

1. **Cohort Grouping**: Projects are grouped by composite cohort key (`state` + `constituency`).
2. **Contiguous CSR Slicing**: Dataframe is sorted by `group_key`, enabling $O(1)$ contiguous slicing (`X[b_start:b_end]`) without expensive sparse index lookups.
3. **Chunk Capping (`MAX_BLOCK_SIZE = 1000`)**: Larger cohorts (e.g. `Sitting Rajya Sabha`) are sub-chunked into windows of max 1,000 rows.
4. **Sparse Dot Product**: Computes local similarity matrix $S_g = X_g X_g^T$.
5. **Diagonal Masking**: $S_g.\text{setdiag}(0.0)$ masks self-similarity.
6. **Threshold Filtering**: Candidate pairs matching $S_{i, j} \ge 0.85$ are extracted vectorially via COO format.

---

## 5. Feature Specifications & Summary Statistics

| Feature Name | Type | Description | Usable Records | Missing Records | Mean | Std | Min | 50% (Median) | Max |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `description_length` | Integer | Raw character length of work description | 127,263 | 0 | 90.5433 | 48.9877 | 0.0000 | 84.0000 | 500.0000 |
| `word_count` | Integer | Total count of words in raw work description | 127,263 | 0 | 14.1163 | 7.7017 | 0.0000 | 13.0000 | 101.0000 |
| `tfidf_mean` | Float | Mean TF-IDF feature weight across all vocabulary terms | 127,263 | 0 | 0.0003 | 0.0001 | 0.0000 | 0.0003 | 0.0009 |
| `tfidf_max` | Float | Maximum TF-IDF feature weight for dominant term | 127,263 | 0 | 0.4603 | 0.1368 | 0.0000 | 0.4438 | 1.0000 |
| `nearest_similarity_score` | Float | Cosine similarity score with most similar project in cohort | 127,263 | 0 | 0.7734 | 0.2252 | 0.0000 | 0.8250 | 1.0000 |
| `similar_project_count` | Integer | Count of projects in cohort with similarity score $\ge 0.85$ | 127,263 | 0 | 10.8818 | 34.4102 | 0.0000 | 0.0000 | 357.0000 |

---

## 6. Output Dataset Manifest

### 1. [`data/features/nlp_features.csv`](file:///d:/MPLADS%28SIH%29/data/features/nlp_features.csv)
- **Rows**: 127,263
- **Columns**: `work_id`, `project_status`, `state`, `constituency`, `description_length`, `word_count`, `tfidf_mean`, `tfidf_max`, `nearest_similarity_score`, `similar_project_count`.

### 2. [`data/processed/potentially_similar_projects.csv`](file:///d:/MPLADS%28SIH%29/data/processed/potentially_similar_projects.csv)
- **Rows**: 692,427 candidate pairs
- **Columns**: `work_id_1`, `work_id_2`, `constituency`, `state`, `mp_name`, `description_1`, `description_2`, `similarity_score`.
- **Filtering Criteria**: Cosine similarity $\ge 0.85$.
