"""Shared fixtures for MTP test suite."""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ddse_df():
    """Minimal DDSE-like DataFrame for unit tests."""
    return pd.DataFrame({
        'electrolyte': [
            'Li3InCl6', 'Li6PS5Cl', 'Li7La3Zr2O12', 'NaTaOCl4',
            'Li2S', 'LiPF6', 'Li3OCl', 'Li10GeP2S12',
        ],
        'Temp_K': [300, 350, 298, 293, 400, 298, 310, 298],
        'Ea_eV': [0.31, 0.22, 0.35, 0.31, 0.45, 0.50, 0.28, 0.20],
        'Ionic_Conductivity': [1e-3, 5e-3, 1e-4, 8.9e-4, 1e-6, 1e-5, 2e-3, 1.2e-2],
        'doi': ['10.1/a', '10.1/b', '10.1/c', '10.1/d',
                '10.1/e', '10.1/f', '10.1/g', '10.1/h'],
        'Material_Type': ['Halide', 'Sulfide', 'Oxide', 'Oxyhalide',
                          'Sulfide', 'Organic', 'Oxide', 'Sulfide'],
    })


@pytest.fixture
def sample_ddse_with_features(sample_ddse_df):
    """DDSE DataFrame with compositional features already attached."""
    df = sample_ddse_df.copy()
    n = len(df)
    rng = np.random.default_rng(42)
    df['avg_electronegativity'] = rng.uniform(1.0, 3.0, n)
    df['avg_atomic_mass'] = rng.uniform(10, 100, n)
    df['avg_ionic_radius'] = rng.uniform(0.3, 2.0, n)
    df['num_elements'] = rng.integers(2, 6, n).astype(float)
    df['li_fraction'] = rng.uniform(0, 0.6, n)
    df['composition_entropy'] = rng.uniform(0.5, 2.0, n)
    df['electronegativity_variance'] = rng.uniform(0, 1.5, n)
    df['group_diversity'] = rng.integers(2, 6, n).astype(float)
    df['packing_efficiency_proxy'] = rng.uniform(0.1, 1.0, n)
    df['li_to_anion_ratio'] = rng.uniform(0, 2.0, n)
    df['heaviest_element_mass'] = rng.uniform(30, 200, n)
    df['lightest_element_mass'] = rng.uniform(1, 40, n)
    df['is_mixture'] = 0
    df['formula_complexity'] = rng.uniform(5, 100, n)
    df['total_atoms'] = rng.uniform(3, 30, n)
    return df


@pytest.fixture
def tmp_csv(sample_ddse_with_features, tmp_path):
    """Write sample DDSE data to a temp CSV and return its path."""
    p = tmp_path / "sample_ddse.csv"
    sample_ddse_with_features.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def simple_formulas():
    """Collection of formulas with expected parse results."""
    return {
        'Li3InCl6': {'Li': 3, 'In': 1, 'Cl': 6},
        'Li2S': {'Li': 2, 'S': 1},
        'NaCl': {'Na': 1, 'Cl': 1},
        'Li7La3Zr2O12': {'Li': 7, 'La': 3, 'Zr': 2, 'O': 12},
        'LiPF6': {'Li': 1, 'P': 1, 'F': 6},
    }


@pytest.fixture
def complex_formulas():
    """Complex / nested / mixture formulas."""
    return {
        '((Li2S)0.75(P2S5)0.25)96(P2O5)4': {
            'Li': 0.75 * 96 * 2,  # 144
            'S': 0.75 * 96 * 1 + 0.25 * 96 * 5,  # 72 + 120 = 192
            'P': 0.25 * 96 * 2 + 4 * 2,  # 48 + 8 = 56
            'O': 4 * 5,  # 20
        },
        'Li3OCl': {'Li': 3, 'O': 1, 'Cl': 1},
    }


@pytest.fixture
def zero_conductivity_df():
    """DataFrame with zero and negative ionic conductivity values."""
    return pd.DataFrame({
        'electrolyte': ['Li2S', 'LiCl', 'NaCl'],
        'Temp_K': [300, 300, 300],
        'Ea_eV': [0.3, 0.4, 0.5],
        'Ionic_Conductivity': [0.0, -1e-5, 1e-3],
        'doi': ['a', 'b', 'c'],
        'Material_Type': ['Sulfide', 'Halide', 'Halide'],
    })
