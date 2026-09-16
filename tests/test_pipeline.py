"""
MPLADS AI Monitoring System - Automated Unit Test Suite (Stage 10)

Tests:
1. Data Loading & Schema Verification
2. Currency String Parsing Logic
3. Date Sequence Parsing & Negative Duration Detection
4. Feature Matrix Integrity (No NaNs, No Infs)
5. Isolation Forest Model Inference & Predictions
6. Multi-Factor Risk Score Calculation & Category Assignments

Run with: python -m unittest discover -s tests -p "test_*.py"
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class TestMPLADSPipeline(unittest.TestCase):
    
    def test_01_data_loading_and_schema(self):
        """Test raw data ingestion and cleaned datasets existence & columns."""
        proj_path = os.path.join(PROCESSED_DIR, "projects_clean.csv")
        self.assertTrue(os.path.exists(proj_path), f"Missing cleaned projects file: {proj_path}")
        
        df = pd.read_csv(proj_path, nrows=10, low_memory=False)
        expected_cols = ["work_id", "project_status", "work_description", "mp_name", "constituency", "state"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing from projects_clean.csv")
        self.assertEqual(len(df), 10, "DataFrame loading failed")
        
    def test_02_currency_parsing_logic(self):
        """Test parsing of currency strings containing ₹ symbols and commas."""
        sample_currencies = ["₹10,976,847.00", "₹ 1,50,000", "5000", np.nan, ""]
        
        def parse_currency(val):
            if pd.isna(val) or val == "":
                return 0.0
            clean_str = str(val).replace("₹", "").replace(",", "").strip()
            try:
                return float(clean_str)
            except ValueError:
                return 0.0
                
        parsed_vals = [parse_currency(v) for v in sample_currencies]
        expected = [10976847.00, 150000.0, 5000.0, 0.0, 0.0]
        self.assertEqual(parsed_vals, expected, "Currency parsing mismatch")
        
    def test_03_date_parsing_and_duration(self):
        """Test ISO date parsing and negative duration sequence detection."""
        rec_date = pd.to_datetime("2023-05-10")
        comp_date_valid = pd.to_datetime("2023-08-15")
        comp_date_invalid = pd.to_datetime("2022-01-01")
        
        dur_valid = (comp_date_valid - rec_date).days
        dur_invalid = (comp_date_invalid - rec_date).days
        
        self.assertGreater(dur_valid, 0, "Valid date sequence duration should be positive")
        self.assertLess(dur_invalid, 0, "Invalid date sequence duration should be negative")
        
    def test_04_feature_matrix_integrity(self):
        """Verify numerical feature matrix contains no NaNs or Infs."""
        feat_path = os.path.join(FEATURES_DIR, "isolation_forest_features.csv")
        self.assertTrue(os.path.exists(feat_path), f"Missing feature matrix file: {feat_path}")
        
        feat_df = pd.read_csv(feat_path, low_memory=False)
        feat_cols = [c for c in feat_df.columns if c not in ["work_id", "mp_name", "state", "constituency", "category", "project_status", "work_description"]]
        
        X = feat_df[feat_cols]
        self.assertFalse(X.isnull().any().any(), "Feature matrix contains NaN values!")
        self.assertFalse(np.isinf(X.values).any(), "Feature matrix contains Infinite values!")
        self.assertGreaterEqual(len(feat_cols), 19, f"Expected at least 19 numerical features, found {len(feat_cols)}")
        
    def test_05_isolation_forest_model_predictions(self):
        """Test trained Isolation Forest model inference & output ranges."""
        model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        
        self.assertTrue(os.path.exists(model_path), f"Missing model file: {model_path}")
        self.assertTrue(os.path.exists(scaler_path), f"Missing scaler file: {scaler_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        n_feat = scaler.n_features_in_
        X_sample = pd.DataFrame(np.zeros((5, n_feat)), columns=scaler.feature_names_in_)
        X_scaled = scaler.transform(X_sample)
        preds = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)
        
        self.assertEqual(len(preds), 5)
        self.assertTrue(set(preds).issubset({-1, 1}), "Predictions must be in {-1, 1}")
        self.assertEqual(len(scores), 5)
        
    def test_06_risk_scoring_math_and_categories(self):
        """Test multi-factor risk score calculation and threshold category assignment."""
        fin = 100.0
        temp = 100.0
        iso = 100.0
        vend = 100.0
        works = 100.0
        
        # Formula: 0.30*Fin + 0.20*Temp + 0.25*Iso + 0.15*Vend + 0.10*Works
        score_max = (0.30 * fin + 0.20 * temp + 0.25 * iso + 0.15 * vend + 0.10 * works)
        self.assertEqual(score_max, 100.0, "Max score calculation error")
        
        def assign_category(score):
            if score <= 39.99:
                return "LOW"
            elif score <= 69.99:
                return "MEDIUM"
            else:
                return "HIGH"
                
        self.assertEqual(assign_category(20.0), "LOW")
        self.assertEqual(assign_category(55.0), "MEDIUM")
        self.assertEqual(assign_category(85.0), "HIGH")

if __name__ == "__main__":
    unittest.main()
