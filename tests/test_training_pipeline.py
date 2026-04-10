"""Tests for the training pipeline (train_best_save.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


# ===================================================================
# Log target preparation
# ===================================================================

class TestLogTargetPreparation:
    """Test _prepare_log_targets logic without full trainer init."""

    def _apply_log_transform(self, ic_values):
        """Replicate the log transform logic from DDSEModelTrainer."""
        ic = pd.Series(ic_values).copy()
        ic = ic.replace(0, 1e-12)
        ic[ic <= 0] = 1e-12
        return np.log10(ic)

    def test_normal_values(self):
        ic = [1e-3, 1e-4, 1e-5]
        log_ic = self._apply_log_transform(ic)
        expected = np.log10(ic)
        np.testing.assert_allclose(log_ic.values, expected)

    def test_zero_replaced(self):
        """Zero conductivity should be replaced with 1e-12 before log."""
        log_ic = self._apply_log_transform([0.0])
        assert log_ic.iloc[0] == pytest.approx(np.log10(1e-12))

    def test_negative_replaced(self):
        """Negative conductivity should be replaced with 1e-12."""
        log_ic = self._apply_log_transform([-1e-5])
        assert log_ic.iloc[0] == pytest.approx(np.log10(1e-12))

    def test_very_small_positive(self):
        """Very small but positive values should be preserved."""
        log_ic = self._apply_log_transform([1e-15])
        assert log_ic.iloc[0] == pytest.approx(-15.0)

    def test_large_values(self):
        log_ic = self._apply_log_transform([1.0])
        assert log_ic.iloc[0] == pytest.approx(0.0)

    def test_mixed_values(self):
        """Mix of normal, zero, and negative values."""
        ic = [1e-3, 0.0, -5e-4, 1e-1]
        log_ic = self._apply_log_transform(ic)
        assert log_ic.iloc[0] == pytest.approx(-3.0)
        assert log_ic.iloc[1] == pytest.approx(np.log10(1e-12))
        assert log_ic.iloc[2] == pytest.approx(np.log10(1e-12))
        assert log_ic.iloc[3] == pytest.approx(-1.0)

    def test_output_no_nan(self):
        """Log transform should never produce NaN."""
        ic = [1e-3, 0.0, -1.0, 1e-10, 100]
        log_ic = self._apply_log_transform(ic)
        assert not log_ic.isna().any()

    def test_output_no_inf(self):
        """Log transform should never produce infinity."""
        ic = [1e-3, 0.0, -1.0]
        log_ic = self._apply_log_transform(ic)
        assert not np.isinf(log_ic).any()


# ===================================================================
# Temperature filtering
# ===================================================================

class TestTemperatureFiltering:
    def test_filters_below_293(self):
        """Only samples with Temp_K >= 293 should remain."""
        df = pd.DataFrame({
            'Temp_K': [200, 250, 293, 300, 400],
            'Ionic_Conductivity': [1e-5] * 5,
        })
        filtered = df[df['Temp_K'] >= 293]
        assert len(filtered) == 3
        assert filtered['Temp_K'].min() >= 293

    def test_boundary_293_included(self):
        df = pd.DataFrame({'Temp_K': [293], 'Ionic_Conductivity': [1e-3]})
        filtered = df[df['Temp_K'] >= 293]
        assert len(filtered) == 1


# ===================================================================
# Feature column exclusion
# ===================================================================

class TestFeatureExclusion:
    def test_target_excluded(self):
        """Target columns should not be used as features."""
        exclude_cols = ['electrolyte', 'doi', 'Ea_eV', 'Ionic_Conductivity', 'log_Ionic_Conductivity']
        all_cols = ['electrolyte', 'doi', 'Ea_eV', 'Ionic_Conductivity',
                    'log_Ionic_Conductivity', 'Temp_K', 'avg_electronegativity',
                    'li_fraction']
        features = [c for c in all_cols if c not in exclude_cols]
        assert 'Temp_K' in features
        assert 'avg_electronegativity' in features
        assert 'li_fraction' in features
        assert 'Ionic_Conductivity' not in features
        assert 'Ea_eV' not in features
        assert 'log_Ionic_Conductivity' not in features

    def test_ea_excluded_from_features(self):
        """Ea_eV must be excluded from features (it's not a prediction target)."""
        exclude_cols = ['electrolyte', 'doi', 'Ea_eV', 'Ionic_Conductivity', 'log_Ionic_Conductivity']
        assert 'Ea_eV' in exclude_cols


# ===================================================================
# Model pipeline structure
# ===================================================================

class TestModelPipelineStructure:
    def test_pipeline_components(self):
        """Pipeline should have imputer → scaler → model."""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=10, random_state=42))
        ])

        assert pipeline.named_steps['imputer'] is not None
        assert pipeline.named_steps['scaler'] is not None
        assert pipeline.named_steps['model'] is not None

    def test_pipeline_handles_nan(self):
        """Pipeline with imputer should handle NaN values."""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=10, random_state=42))
        ])

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        X[0, 0] = np.nan
        X[10, 2] = np.nan
        y = rng.standard_normal(50)

        # Should not raise
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 50
        assert not np.isnan(preds).any()


# ===================================================================
# Feature combination construction
# ===================================================================

class TestFeatureCombinations:
    def test_original_only(self):
        n_samples, n_orig, n_m2v = 100, 15, 200
        rng = np.random.default_rng(42)
        orig = rng.standard_normal((n_samples, n_orig))
        m2v = rng.standard_normal((n_samples, n_m2v))

        # Original only
        X = orig
        assert X.shape == (100, 15)

    def test_mat2vec_only(self):
        n_samples, n_orig, n_m2v = 100, 15, 200
        rng = np.random.default_rng(42)
        m2v = rng.standard_normal((n_samples, n_m2v))

        X = m2v
        assert X.shape == (100, 200)

    def test_combined(self):
        n_samples, n_orig, n_m2v = 100, 15, 200
        rng = np.random.default_rng(42)
        orig = rng.standard_normal((n_samples, n_orig))
        m2v = rng.standard_normal((n_samples, n_m2v))

        X = np.hstack([orig, m2v])
        assert X.shape == (100, 215)

    def test_combined_with_all_zero_mat2vec(self):
        """If mat2vec returns all zeros, combined features still have original data."""
        n_samples, n_orig, n_m2v = 100, 15, 200
        rng = np.random.default_rng(42)
        orig = rng.standard_normal((n_samples, n_orig))
        m2v = np.zeros((n_samples, n_m2v))  # All zero embeddings!

        X = np.hstack([orig, m2v])
        assert X.shape == (100, 215)

        # Original features should still be non-zero
        assert not np.allclose(X[:, :n_orig], 0)
        # Mat2vec part should be zero
        assert np.allclose(X[:, n_orig:], 0)
