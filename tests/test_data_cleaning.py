"""Tests for src/data_cleaning.py — deduplication, conflict resolution, leakage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import normalize_formula, aggregate_conflicts


# ===================================================================
# normalize_formula
# ===================================================================

class TestNormalizeFormula:
    def test_strips_whitespace(self):
        assert normalize_formula("  Li3InCl6  ") == "Li3InCl6"

    def test_removes_internal_spaces(self):
        assert normalize_formula("Li 3 In Cl 6") == "Li3InCl6"

    def test_nan_returns_empty(self):
        assert normalize_formula(np.nan) == ""
        assert normalize_formula(None) == ""

    def test_numeric_input(self):
        result = normalize_formula(123)
        assert result == "123"

    def test_already_clean(self):
        assert normalize_formula("Li2S") == "Li2S"


# ===================================================================
# aggregate_conflicts
# ===================================================================

class TestAggregateConflicts:
    def test_no_conflicts(self):
        df = pd.DataFrame({
            'formula': ['Li2S', 'NaCl'],
            'temp': [300, 350],
            'value': [1.0, 2.0],
        })
        result, n_groups, n_removed = aggregate_conflicts(
            df, 'formula', 'temp', ['value'])
        assert len(result) == 2
        assert n_groups == 0
        assert n_removed == 0

    def test_exact_duplicates_merged(self):
        df = pd.DataFrame({
            'formula': ['Li2S', 'Li2S', 'NaCl'],
            'temp': [300, 300, 350],
            'value': [1.0, 3.0, 2.0],
        })
        result, n_groups, n_removed = aggregate_conflicts(
            df, 'formula', 'temp', ['value'])
        assert len(result) == 2  # Li2S merged, NaCl stays
        assert n_groups == 1
        assert n_removed == 1

    def test_median_aggregation(self):
        """Conflicting values should be aggregated by median."""
        df = pd.DataFrame({
            'formula': ['Li2S', 'Li2S', 'Li2S'],
            'temp': [300, 300, 300],
            'value': [1.0, 3.0, 5.0],
        })
        result, _, _ = aggregate_conflicts(df, 'formula', 'temp', ['value'])
        merged_val = result[result['formula'] == 'Li2S']['value'].values[0]
        assert merged_val == pytest.approx(3.0)  # median of 1, 3, 5

    def test_different_temps_not_merged(self):
        """Same formula at different temperatures should NOT merge."""
        df = pd.DataFrame({
            'formula': ['Li2S', 'Li2S'],
            'temp': [300, 400],
            'value': [1.0, 3.0],
        })
        result, n_groups, n_removed = aggregate_conflicts(
            df, 'formula', 'temp', ['value'])
        assert len(result) == 2
        assert n_groups == 0

    def test_preserves_metadata_first(self):
        """Non-numeric metadata columns should use 'first'."""
        df = pd.DataFrame({
            'formula': ['Li2S', 'Li2S'],
            'temp': [300, 300],
            'value': [1.0, 3.0],
            'source': ['paper_A', 'paper_B'],
        })
        result, _, _ = aggregate_conflicts(
            df, 'formula', 'temp', ['value'], extra_keep_cols=['source'])
        assert len(result) == 1
        assert result['source'].iloc[0] == 'paper_A'

    def test_multiple_value_columns(self):
        """Multiple value columns should all be aggregated by median."""
        df = pd.DataFrame({
            'formula': ['Li2S', 'Li2S'],
            'temp': [300, 300],
            'ic': [1e-3, 3e-3],
            'ea': [0.3, 0.5],
        })
        result, _, _ = aggregate_conflicts(
            df, 'formula', 'temp', ['ic', 'ea'])
        assert result['ic'].iloc[0] == pytest.approx(2e-3)
        assert result['ea'].iloc[0] == pytest.approx(0.4)


# ===================================================================
# Cross-dataset leakage detection
# ===================================================================

class TestLeakageDetection:
    def test_overlap_detection(self):
        """Formulas in both training and validation sets should be flagged."""
        train_formulas = ['Li2S', 'NaCl', 'LiCl', 'Li3InCl6']
        val_formulas = ['Li2S', 'LiCl']

        train_norm = set(normalize_formula(f) for f in train_formulas)
        val_norm = set(normalize_formula(f) for f in val_formulas)

        overlap = train_norm & val_norm
        assert overlap == {'Li2S', 'LiCl'}

    def test_no_overlap_when_clean(self):
        train_formulas = ['Li3InCl6', 'Li6PS5Cl']
        val_formulas = ['Li2S', 'NaCl']

        train_norm = set(normalize_formula(f) for f in train_formulas)
        val_norm = set(normalize_formula(f) for f in val_formulas)

        assert len(train_norm & val_norm) == 0

    def test_whitespace_normalized_before_comparison(self):
        """'Li2S ' and 'Li2S' should be considered the same formula."""
        assert normalize_formula('Li2S ') == normalize_formula('Li2S')
        assert normalize_formula(' Li2S') == normalize_formula('Li2S')
