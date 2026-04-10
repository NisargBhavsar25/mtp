"""Tests for src/analysis/data_analysis.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np


# ===================================================================
# DDSEDataAnalyzer initialization
# ===================================================================

class TestDDSEDataAnalyzerInit:
    def test_loads_csv(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        assert len(analyzer.df) > 0

    def test_target_cols_only_ic(self, tmp_csv):
        """After Ea removal, only Ionic_Conductivity should be a target."""
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        assert analyzer.target_cols == ['Ionic_Conductivity']
        assert 'Ea_eV' not in analyzer.target_cols

    def test_ea_excluded_from_features(self, tmp_csv):
        """Ea_eV should be excluded from feature columns."""
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        assert 'Ea_eV' not in analyzer.feature_cols

    def test_column_types_identified(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        assert analyzer.numeric_cols is not None
        assert analyzer.categorical_cols is not None
        assert analyzer.feature_cols is not None


# ===================================================================
# Correlation analysis (hardcoded values)
# ===================================================================

class TestCorrelationAnalysis:
    def test_returns_dict_and_df(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        pearson_dict, corr_df = analyzer.correlation_analysis()
        assert isinstance(pearson_dict, dict)
        assert isinstance(corr_df, pd.DataFrame)

    def test_expected_features_present(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        pearson_dict, _ = analyzer.correlation_analysis()
        assert 'Temperature' in pearson_dict
        assert 'Li_fraction' in pearson_dict
        assert 'Average_ionic_radius' in pearson_dict

    def test_correlation_values_in_range(self, tmp_csv):
        """All Pearson correlations must be in [-1, 1]."""
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        pearson_dict, _ = analyzer.correlation_analysis()
        for feat, r in pearson_dict.items():
            assert -1 <= r <= 1, f"{feat} has r={r} outside [-1, 1]"

    def test_sorted_by_absolute_value(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        _, corr_df = analyzer.correlation_analysis()
        abs_vals = corr_df['|r|'].values
        # Should be sorted descending
        assert all(abs_vals[i] >= abs_vals[i+1] for i in range(len(abs_vals)-1))


# ===================================================================
# SHAP analysis (hardcoded values)
# ===================================================================

class TestShapAnalysis:
    def test_returns_dict_and_df(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        shap_dict, shap_df = analyzer.shap_analysis()
        assert isinstance(shap_dict, dict)
        assert isinstance(shap_df, pd.DataFrame)

    def test_shap_values_non_negative(self, tmp_csv):
        """Mean |SHAP| values must be >= 0."""
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        shap_dict, _ = analyzer.shap_analysis()
        for feat, val in shap_dict.items():
            assert val >= 0, f"{feat} has negative SHAP value: {val}"

    def test_temperature_is_top_feature(self, tmp_csv):
        """Temperature should have the highest SHAP value."""
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        _, shap_df = analyzer.shap_analysis()
        assert shap_df.iloc[0]['Feature'] == 'Temperature'

    def test_mat2vec_aggregate_present(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        shap_dict, _ = analyzer.shap_analysis()
        assert 'Mat2Vec_Aggregate_Sum' in shap_dict


# ===================================================================
# Outlier detection
# ===================================================================

class TestOutlierDetection:
    def test_returns_dataframe(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        outlier_df = analyzer.outlier_detection()
        assert isinstance(outlier_df, pd.DataFrame)

    def test_has_expected_columns(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        outlier_df = analyzer.outlier_detection()
        assert 'IQR_outliers' in outlier_df.columns
        assert 'Z_score_outliers' in outlier_df.columns

    def test_outlier_counts_non_negative(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        outlier_df = analyzer.outlier_detection()
        assert (outlier_df['IQR_outliers'] >= 0).all()
        assert (outlier_df['Z_score_outliers'] >= 0).all()


# ===================================================================
# Data quality assessment
# ===================================================================

class TestDataQualityAssessment:
    def test_returns_score_and_issues(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        score, issues = analyzer.data_quality_assessment()
        assert isinstance(score, (int, float))
        assert isinstance(issues, list)

    def test_score_in_valid_range(self, tmp_csv):
        from src.analysis.data_analysis import DDSEDataAnalyzer
        analyzer = DDSEDataAnalyzer(tmp_csv)
        score, _ = analyzer.data_quality_assessment()
        assert 0 <= score <= 100
