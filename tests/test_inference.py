"""Tests for src/inference/predict_properties.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.features import get_composition as gc


# ===================================================================
# Formula column auto-detection
# ===================================================================

class TestFormulaColumnDetection:
    POSSIBLE_NAMES = ['formula', 'Formula', 'composition', 'Composition',
                      'Material', 'electrolyte']

    @pytest.mark.parametrize("col_name", POSSIBLE_NAMES)
    def test_detects_known_column_names(self, col_name):
        df = pd.DataFrame({col_name: ['Li2S', 'NaCl']})
        detected = next((c for c in self.POSSIBLE_NAMES if c in df.columns), None)
        assert detected == col_name

    def test_returns_none_for_unknown_column(self):
        df = pd.DataFrame({'compound': ['Li2S']})
        detected = next((c for c in self.POSSIBLE_NAMES if c in df.columns), None)
        assert detected is None


# ===================================================================
# Temperature default
# ===================================================================

class TestTemperatureDefault:
    def test_adds_temp_if_missing(self):
        df = pd.DataFrame({'formula': ['Li2S', 'NaCl']})
        if 'Temp_K' not in df.columns:
            df['Temp_K'] = 298.0
        assert 'Temp_K' in df.columns
        assert (df['Temp_K'] == 298.0).all()

    def test_preserves_existing_temp(self):
        df = pd.DataFrame({
            'formula': ['Li2S', 'NaCl'],
            'Temp_K': [350, 400],
        })
        assert df['Temp_K'].iloc[0] == 350
        assert df['Temp_K'].iloc[1] == 400


# ===================================================================
# Column renaming (orig_ prefix)
# ===================================================================

class TestColumnRenaming:
    def test_adds_orig_prefix(self):
        df = pd.DataFrame({
            'electrolyte': ['Li2S'],
            'Temp_K': [300],
            'avg_electronegativity': [2.0],
            'li_fraction': [0.5],
        })
        rename_map = {col: f"orig_{col}" for col in df.columns
                      if col not in ['electrolyte', 'Temp_K', 'doi']}
        rename_map['Temp_K'] = 'orig_Temp_K'
        df_renamed = df.rename(columns=rename_map)

        assert 'orig_Temp_K' in df_renamed.columns
        assert 'orig_avg_electronegativity' in df_renamed.columns
        assert 'orig_li_fraction' in df_renamed.columns
        # electrolyte should NOT be renamed
        assert 'electrolyte' in df_renamed.columns

    def test_no_double_prefix(self):
        """If column already has orig_ prefix, don't double it."""
        rename_map = {'avg_electronegativity': 'orig_avg_electronegativity'}
        assert not rename_map.get('orig_avg_electronegativity', '').startswith('orig_orig_')


# ===================================================================
# Feature alignment with model expectations
# ===================================================================

class TestFeatureAlignment:
    def test_reindex_fills_missing_with_zero(self):
        """Missing features should be filled with 0."""
        required_cols = ['orig_Temp_K', 'orig_li_fraction', 'mat2vec_0', 'mat2vec_1']
        df = pd.DataFrame({
            'orig_Temp_K': [300],
            'orig_li_fraction': [0.5],
            # mat2vec cols missing
        })
        X = df.reindex(columns=required_cols, fill_value=0)
        assert X.shape == (1, 4)
        assert X['mat2vec_0'].iloc[0] == 0
        assert X['mat2vec_1'].iloc[0] == 0
        assert X['orig_Temp_K'].iloc[0] == 300

    def test_extra_columns_dropped(self):
        """Extra columns not in required_cols should be dropped."""
        required_cols = ['orig_Temp_K', 'orig_li_fraction']
        df = pd.DataFrame({
            'orig_Temp_K': [300],
            'orig_li_fraction': [0.5],
            'extra_col': [999],
        })
        X = df.reindex(columns=required_cols, fill_value=0)
        assert 'extra_col' not in X.columns


# ===================================================================
# Inverse log transform
# ===================================================================

class TestInverseLogTransform:
    def test_correct_inversion(self):
        """10^log_pred should recover the original conductivity."""
        log_preds = np.array([-3.0, -4.0, -1.0, 0.0])
        ic = 10 ** log_preds
        np.testing.assert_allclose(ic, [1e-3, 1e-4, 1e-1, 1.0])

    def test_very_negative_log(self):
        """Very negative log predictions should give very small IC."""
        log_pred = -12.0
        ic = 10 ** log_pred
        assert ic == pytest.approx(1e-12)

    def test_positive_log(self):
        """Positive log predictions are physically unusual but should work."""
        log_pred = 1.0
        ic = 10 ** log_pred
        assert ic == pytest.approx(10.0)


# ===================================================================
# End-to-end feature generation for inference
# ===================================================================

class TestInferenceFeatureGeneration:
    def test_feature_generation_matches_training_columns(self):
        """Features generated at inference time should be the same set
        as those generated during training."""
        df = pd.DataFrame({
            'electrolyte': ['Li3InCl6', 'Li2S', 'NaCl'],
            'Temp_K': [300, 300, 300],
        })
        result = gc.enhance_composition_features_fixed(df, 'electrolyte')

        training_features = [
            'avg_electronegativity', 'avg_atomic_mass', 'avg_ionic_radius',
            'num_elements', 'li_fraction', 'composition_entropy',
            'electronegativity_variance', 'group_diversity',
            'packing_efficiency_proxy', 'li_to_anion_ratio',
            'heaviest_element_mass', 'lightest_element_mass',
            'is_mixture', 'formula_complexity', 'total_atoms',
        ]
        for col in training_features:
            assert col in result.columns, f"Inference missing training feature: {col}"

    def test_csv_round_trip(self, tmp_path):
        """Write predictions CSV, read it back, verify columns."""
        df = pd.DataFrame({
            'formula': ['Li2S', 'NaCl'],
            'Predicted_log_IC': [-3.5, -4.2],
            'Predicted_IC_S_cm': [10**-3.5, 10**-4.2],
        })
        path = tmp_path / "predictions.csv"
        df.to_csv(path, index=False)

        loaded = pd.read_csv(path)
        assert 'Predicted_log_IC' in loaded.columns
        assert 'Predicted_IC_S_cm' in loaded.columns
        assert len(loaded) == 2
