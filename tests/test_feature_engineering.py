"""Tests for compositional feature calculation in get_composition.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.features.get_composition import (
    calculate_weighted_property,
    calculate_composition_entropy,
    calculate_electronegativity_variance,
    get_element_group_diversity,
    calculate_packing_efficiency_proxy,
    enhance_composition_features_fixed,
    PERIODIC_TABLE,
)


# ===================================================================
# calculate_weighted_property
# ===================================================================

class TestCalculateWeightedProperty:
    def test_single_element(self):
        comp = {'Li': 1}
        result = calculate_weighted_property(comp, 'atomic_mass')
        assert result == pytest.approx(PERIODIC_TABLE['Li']['atomic_mass'])

    def test_weighted_average(self):
        comp = {'Li': 3, 'Cl': 6}
        total = 3 + 6
        expected = (3 * PERIODIC_TABLE['Li']['atomic_mass'] +
                    6 * PERIODIC_TABLE['Cl']['atomic_mass']) / total
        assert calculate_weighted_property(comp, 'atomic_mass') == pytest.approx(expected)

    def test_unknown_element_returns_none(self):
        comp = {'Li': 1, 'Xx': 2}
        assert calculate_weighted_property(comp, 'atomic_mass') is None

    def test_empty_composition_returns_none(self):
        assert calculate_weighted_property({}, 'atomic_mass') is None

    def test_zero_total_atoms_returns_none(self):
        # edge case: composition with all-zero counts
        comp = {'Li': 0, 'Cl': 0}
        assert calculate_weighted_property(comp, 'atomic_mass') is None


# ===================================================================
# calculate_composition_entropy
# ===================================================================

class TestCompositionEntropy:
    def test_single_element_zero_entropy(self):
        """A single-element compound has zero configurational entropy."""
        comp = {'Li': 5}
        assert calculate_composition_entropy(comp) == pytest.approx(0.0)

    def test_binary_equal_fractions(self):
        """50/50 binary has ln(2) entropy."""
        comp = {'Li': 1, 'Cl': 1}
        assert calculate_composition_entropy(comp) == pytest.approx(np.log(2))

    def test_more_elements_higher_entropy(self):
        binary = calculate_composition_entropy({'Li': 1, 'Cl': 1})
        ternary = calculate_composition_entropy({'Li': 1, 'Cl': 1, 'O': 1})
        assert ternary > binary

    def test_empty_returns_zero(self):
        assert calculate_composition_entropy({}) == 0

    def test_entropy_non_negative(self):
        comp = {'Li': 3, 'In': 1, 'Cl': 6}
        assert calculate_composition_entropy(comp) >= 0


# ===================================================================
# calculate_electronegativity_variance
# ===================================================================

class TestElectronegativityVariance:
    def test_single_element_zero_variance(self):
        comp = {'Li': 5}
        assert calculate_electronegativity_variance(comp) == pytest.approx(0.0)

    def test_two_element_positive_variance(self):
        comp = {'Li': 1, 'F': 1}
        var = calculate_electronegativity_variance(comp)
        assert var > 0

    def test_identical_en_zero_variance(self):
        """Elements with the same EN should give ~0 variance."""
        # Ni and Cu have very similar EN (1.91 vs 1.90)
        comp = {'Ni': 1, 'Cu': 1}
        var = calculate_electronegativity_variance(comp)
        assert var < 0.01


# ===================================================================
# get_element_group_diversity
# ===================================================================

class TestGroupDiversity:
    def test_same_group(self):
        comp = {'Li': 1, 'Na': 1, 'K': 1}  # All group 1
        assert get_element_group_diversity(comp) == 1

    def test_different_groups(self):
        comp = {'Li': 3, 'In': 1, 'Cl': 6}  # Groups 1, 13, 17
        assert get_element_group_diversity(comp) == 3

    def test_empty(self):
        assert get_element_group_diversity({}) == 0


# ===================================================================
# calculate_packing_efficiency_proxy
# ===================================================================

class TestPackingEfficiency:
    def test_single_element(self):
        comp = {'Li': 1}
        assert calculate_packing_efficiency_proxy(comp) == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        comp = {'Li': 3, 'La': 3, 'Zr': 2, 'O': 12}
        pe = calculate_packing_efficiency_proxy(comp)
        assert 0 < pe <= 1.0

    def test_empty_returns_none(self):
        assert calculate_packing_efficiency_proxy({}) is None


# ===================================================================
# enhance_composition_features_fixed (integration)
# ===================================================================

class TestEnhanceCompositionFeatures:
    def test_adds_all_expected_columns(self):
        df = pd.DataFrame({
            'electrolyte': ['Li3InCl6', 'Li2S', 'NaCl'],
            'Temp_K': [300, 300, 300],
        })
        result = enhance_composition_features_fixed(df, 'electrolyte')

        expected_cols = [
            'avg_electronegativity', 'avg_atomic_mass', 'avg_ionic_radius',
            'num_elements', 'li_fraction', 'composition_entropy',
            'electronegativity_variance', 'group_diversity',
            'packing_efficiency_proxy', 'li_to_anion_ratio',
            'heaviest_element_mass', 'lightest_element_mass',
            'is_mixture', 'formula_complexity', 'total_atoms',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_li_fraction_correct(self):
        df = pd.DataFrame({'electrolyte': ['Li2S']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        # Li2S: 2 Li out of 3 total atoms → 2/3
        assert result['li_fraction'].iloc[0] == pytest.approx(2 / 3)

    def test_no_li_gives_zero_fraction(self):
        df = pd.DataFrame({'electrolyte': ['NaCl']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        assert result['li_fraction'].iloc[0] == pytest.approx(0.0)

    def test_num_elements_correct(self):
        df = pd.DataFrame({'electrolyte': ['Li3InCl6']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        assert result['num_elements'].iloc[0] == 3  # Li, In, Cl

    def test_handles_nan_formula(self):
        df = pd.DataFrame({'electrolyte': [np.nan, 'Li2S']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        # First row should be None/NaN, second should be valid
        assert pd.isna(result['avg_electronegativity'].iloc[0])
        assert not pd.isna(result['avg_electronegativity'].iloc[1])

    def test_is_mixture_flag(self):
        df = pd.DataFrame({'electrolyte': ['Li3InCl6', '0.75LiBH4-0.25Ca(BH4)2']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        assert result['is_mixture'].iloc[0] == 0
        # The mixture should be flagged as 1
        assert result['is_mixture'].iloc[1] == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({'formula': ['Li2S']})
        with pytest.raises(ValueError, match="not found"):
            enhance_composition_features_fixed(df, 'electrolyte')

    def test_total_atoms_positive(self):
        df = pd.DataFrame({'electrolyte': ['Li3InCl6', 'Li7La3Zr2O12']})
        result = enhance_composition_features_fixed(df, 'electrolyte')
        assert (result['total_atoms'] > 0).all()

    def test_li_to_anion_ratio_no_anion(self):
        """Compound with Li but no anions → ratio = 0."""
        df = pd.DataFrame({'electrolyte': ['LiAl']})  # No O, S, Cl, F, Br, I
        result = enhance_composition_features_fixed(df, 'electrolyte')
        assert result['li_to_anion_ratio'].iloc[0] == pytest.approx(0.0)
