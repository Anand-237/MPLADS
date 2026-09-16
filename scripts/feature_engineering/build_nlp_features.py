"""
MPLADS Work Description NLP Similarity Feature Engineering Script (Stage 3D)

This script performs text preprocessing, TF-IDF vectorization, and sparse cosine similarity candidate retrieval for project descriptions:
1. Cleans text: lowercase, remove punctuation, normalize whitespace.
2. Extracts text length and word count metrics.
3. Builds TF-IDF representation (max_features=10000, unigrams & bigrams).
4. Computes sparse cosine similarity within state-constituency cohort blocks.
5. Identifies candidate pairs with similarity score >= 0.85.
6. Enforces strict neutral non-accusatory terminology ('Potentially Similar Projects').

Outputs:
- data/features/nlp_features.csv
- data/processed/potentially_similar_projects.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, coo_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for candidate identification
MAX_BLOCK_SIZE = 1000       # Maximum rows per matrix multiplication chunk for O(1) memory & ultra-fast compute

def clean_text(text):
    """Normalize text: lowercase, strip punctuation/symbols, normalize whitespace."""
    if pd.isnull(text):
        return ""
    text_str = str(text).lower()
    clean_chars = [c if c.isalnum() or c.isspace() else " " for c in text_str]
    return " ".join("".join(clean_chars).split())

def build_nlp_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("=" * 80, flush=True)
    print("            MPLADS NLP WORK DESCRIPTION FEATURE PIPELINE (STAGE 3D)        ", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    proj_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
    proj = pd.read_csv(proj_path, low_memory=False)
    
    total_projects = len(proj)
    non_null_descriptions = int(proj["work_description"].notnull().sum())
    unique_descriptions = int(proj["work_description"].nunique())
    
    print(f"\n[1] LOADED PROJECTS DATASET SUMMARY:", flush=True)
    print(f"  - Total Projects:           {total_projects:,}", flush=True)
    print(f"  - Non-Null Descriptions:    {non_null_descriptions:,}", flush=True)
    print(f"  - Unique Work Descriptions: {unique_descriptions:,}", flush=True)
    
    # -------------------------------------------------------------------------
    # 1. TEXT CLEANING & BASIC TEXT METRICS
    # -------------------------------------------------------------------------
    print("\n[2] CLEANING TEXT & EXTRACTING TEXT SCALE METRICS...", flush=True)
    proj["clean_description"] = proj["work_description"].apply(clean_text)
    
    # Create composite cohort key (State + Constituency)
    proj["group_key"] = proj["state"].fillna("UNKNOWN_STATE") + "___" + proj["constituency"].fillna("UNKNOWN_CONST")
    proj["orig_order"] = np.arange(total_projects)
    
    # Sort dataframe by group_key to enable O(1) contiguous CSR submatrix slicing
    proj = proj.sort_values("group_key").reset_index(drop=True)
    
    description_length_sorted = proj["work_description"].fillna("").astype(str).str.len()
    word_count_sorted = proj["work_description"].fillna("").astype(str).str.split().str.len()

    # -------------------------------------------------------------------------
    # 2. TF-IDF VECTORIZATION
    # -------------------------------------------------------------------------
    print("\n[3] BUILDING TF-IDF VECTOR REPRESENTATION...", flush=True)
    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )
    X = vec.fit_transform(proj["clean_description"])
    vocab_size = len(vec.vocabulary_)
    print(f"  - TF-IDF Matrix Shape: {X.shape[0]:,} rows x {X.shape[1]:,} features", flush=True)
    print(f"  - Vocabulary Size:     {vocab_size:,} terms (unigrams & bigrams)", flush=True)
    
    tfidf_mean_sorted = np.array(X.mean(axis=1)).ravel()
    tfidf_max_sorted = np.array(X.max(axis=1).toarray()).ravel()

    # -------------------------------------------------------------------------
    # 3. SPARSE COSINE SIMILARITY & CANDIDATE RETRIEVAL
    # -------------------------------------------------------------------------
    print(f"\n[4] COMPUTING SPARSE COSINE SIMILARITY (Threshold >= {SIMILARITY_THRESHOLD}, Chunk Size <= {MAX_BLOCK_SIZE})...", flush=True)
    nearest_similarity_score_sorted = np.zeros(total_projects, dtype=np.float32)
    similar_project_count_sorted = np.zeros(total_projects, dtype=np.int32)
    
    candidate_chunks = []
    
    work_id_vals = proj["work_id"].values
    state_vals = proj["state"].values
    mp_name_vals = proj["mp_name"].values
    desc_vals = proj["work_description"].values
    const_vals = proj["constituency"].values
    
    # Identify contiguous slice ranges for each cohort group
    group_series = proj["group_key"]
    change_locations = np.where(group_series.values[:-1] != group_series.values[1:])[0] + 1
    slice_starts = np.concatenate(([0], change_locations))
    slice_ends = np.concatenate((change_locations, [total_projects]))
    
    for start_idx, end_idx in zip(slice_starts, slice_ends):
        group_len = end_idx - start_idx
        if group_len <= 1:
            continue
            
        # Sub-chunk larger cohorts into MAX_BLOCK_SIZE windows for ultra-fast dot products
        for b_start in range(start_idx, end_idx, MAX_BLOCK_SIZE):
            b_end = min(b_start + MAX_BLOCK_SIZE, end_idx)
            chunk_len = b_end - b_start
            if chunk_len <= 1:
                continue
                
            X_g = X[b_start:b_end]
            S_g = X_g.dot(X_g.T).tocsr()
            S_g.setdiag(0.0)
            S_g.eliminate_zeros()
            
            if S_g.nnz == 0:
                continue
                
            max_sims = S_g.max(axis=1).toarray().ravel()
            S_thresh = S_g >= SIMILARITY_THRESHOLD
            counts = np.array(S_thresh.sum(axis=1)).ravel()
            
            nearest_similarity_score_sorted[b_start:b_end] = np.maximum(
                nearest_similarity_score_sorted[b_start:b_end], max_sims
            )
            similar_project_count_sorted[b_start:b_end] += counts
            
            coo = S_g.tocoo()
            pair_mask = (coo.row < coo.col) & (coo.data >= SIMILARITY_THRESHOLD)
            
            if pair_mask.any():
                r_idx = b_start + coo.row[pair_mask]
                c_idx = b_start + coo.col[pair_mask]
                scores = np.round(coo.data[pair_mask], 4)
                
                chunk_df = pd.DataFrame({
                    "work_id_1": work_id_vals[r_idx],
                    "work_id_2": work_id_vals[c_idx],
                    "constituency": const_vals[r_idx],
                    "state": state_vals[r_idx],
                    "mp_name": mp_name_vals[r_idx],
                    "description_1": desc_vals[r_idx],
                    "description_2": desc_vals[c_idx],
                    "similarity_score": scores
                })
                candidate_chunks.append(chunk_df)

    if len(candidate_chunks) > 0:
        candidates_df = pd.concat(candidate_chunks, ignore_index=True)
    else:
        candidates_df = pd.DataFrame(columns=[
            "work_id_1", "work_id_2", "constituency", "state",
            "mp_name", "description_1", "description_2", "similarity_score"
        ])

    candidates_path = os.path.join(PROCESSED_DIR, "potentially_similar_projects.csv")
    candidates_df.to_csv(candidates_path, index=False, encoding="utf-8")
    
    # Restore original row ordering
    proj["description_length"] = description_length_sorted
    proj["word_count"] = word_count_sorted
    proj["tfidf_mean"] = tfidf_mean_sorted
    proj["tfidf_max"] = tfidf_max_sorted
    proj["nearest_similarity_score"] = nearest_similarity_score_sorted
    proj["similar_project_count"] = similar_project_count_sorted
    
    proj_restored = proj.sort_values("orig_order").reset_index(drop=True)
    
    # Assemble NLP Features DataFrame
    nlp_features_df = pd.DataFrame()
    nlp_features_df["work_id"] = proj_restored["work_id"]
    nlp_features_df["project_status"] = proj_restored["project_status"]
    nlp_features_df["state"] = proj_restored["state"]
    nlp_features_df["constituency"] = proj_restored["constituency"]
    nlp_features_df["description_length"] = proj_restored["description_length"]
    nlp_features_df["word_count"] = proj_restored["word_count"]
    nlp_features_df["tfidf_mean"] = proj_restored["tfidf_mean"]
    nlp_features_df["tfidf_max"] = proj_restored["tfidf_max"]
    nlp_features_df["nearest_similarity_score"] = proj_restored["nearest_similarity_score"]
    nlp_features_df["similar_project_count"] = proj_restored["similar_project_count"]
    
    nlp_features_path = os.path.join(FEATURES_DIR, "nlp_features.csv")
    nlp_features_df.to_csv(nlp_features_path, index=False, encoding="utf-8")
    
    t1 = time.time()
    
    # -------------------------------------------------------------------------
    # PRINT MANDATORY STAGE 3D REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("               ENGINEERED NLP WORK DESCRIPTION REPORT                 ", flush=True)
    print("=" * 80, flush=True)
    print("1. PIPELINE PARAMETERS & AUDIT METRICS:", flush=True)
    print(f"   - Number of Work Descriptions Processed: {non_null_descriptions:,} (Total Projects: {total_projects:,})", flush=True)
    print(f"   - TF-IDF Vocabulary Size:                {vocab_size:,} terms", flush=True)
    print(f"   - Cosine Similarity Threshold Used:       {SIMILARITY_THRESHOLD}", flush=True)
    print(f"   - Number of Similarity Candidates (>= {SIMILARITY_THRESHOLD}): {len(candidates_df):,} potentially similar project pairs", flush=True)
    print(f"   - Projects with >= 1 Similar Candidate:  {(proj_restored['similar_project_count'] > 0).sum():,} projects", flush=True)
    print(f"   - Total Execution Time:                 {t1 - t0:.2f} seconds", flush=True)

    feature_cols = [
        "description_length",
        "word_count",
        "tfidf_mean",
        "tfidf_max",
        "nearest_similarity_score",
        "similar_project_count"
    ]

    print(f"\n2. FEATURE DESCRIPTIVE STATISTICS across {total_projects:,} Projects:", flush=True)
    print(f"  {'Feature Name':<30} | {'Usable':<8} | {'Missing':<8} | {'Mean':<10} | {'Std':<10} | {'Min':<8} | {'50%':<8} | {'Max':<10}", flush=True)
    print("  " + "-" * 105, flush=True)
    
    for col in feature_cols:
        s = nlp_features_df[col]
        non_null_cnt = int(s.notnull().sum())
        null_cnt = int(s.isnull().sum())
        mean_val = f"{s.mean():.4f}" if non_null_cnt > 0 else "N/A"
        std_val = f"{s.std():.4f}" if non_null_cnt > 0 else "N/A"
        min_val = f"{s.min():.4f}" if non_null_cnt > 0 else "N/A"
        med_val = f"{s.median():.4f}" if non_null_cnt > 0 else "N/A"
        max_val = f"{s.max():.4f}" if non_null_cnt > 0 else "N/A"
        
        print(f"  {col:<30} | {non_null_cnt:<8,} | {null_cnt:<8,} | {mean_val:<10} | {std_val:<10} | {min_val:<8} | {med_val:<8} | {max_val:<10}", flush=True)

    print("\n3. OUTPUT DATASET MANIFEST:", flush=True)
    print(f"   - Feature Matrix: {nlp_features_path} ({len(nlp_features_df):,} rows)", flush=True)
    print(f"   - Candidates Table: {candidates_path} ({len(candidates_df):,} pairs)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    build_nlp_features()
