"""Tests for src/features/get_composition.py — formula parsing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from collections import defaultdict

from src.features.get_composition import (
    preprocess_formula,
    parse_complex_formula,
    parse_mixture_formula,
    PERIODIC_TABLE,
)


# ===================================================================
# preprocess_formula
# ===================================================================

class TestPreprocessFormula:
    def test_strips_whitespace(self):
        assert preprocess_formula("  Li3InCl6  ") == "Li3InCl6"

    def test_removes_ball_milled_suffix(self):
        assert preprocess_formula("Li3InCl6 (ball milled)") == "Li3InCl6"

    def test_removes_dried_suffix(self):
        assert preprocess_formula("Li2S(dried)") == "Li2S"

    def test_removes_annealed_suffix(self):
        assert preprocess_formula("LiCl [annealed]") == "LiCl"

    def test_keeps_chemical_parentheses(self):
        result = preprocess_formula("Ca(BH4)2")
        assert "Ca" in result
        assert "BH4" in result

    def test_non_string_returns_empty(self):
        assert preprocess_formula(None) == ""
        assert preprocess_formula(123) == ""
        assert preprocess_formula(np.nan) == ""

    def test_empty_string(self):
        assert preprocess_formula("") == ""
        assert preprocess_formula("   ") == ""


# ===================================================================
# parse_complex_formula — simple compounds
# ===================================================================

class TestParseComplexFormulaSimple:
    def test_simple_binary(self):
        comp = parse_complex_formula("NaCl")
        assert comp == {'Na': 1.0, 'Cl': 1.0}

    def test_with_subscripts(self):
        comp = parse_complex_formula("Li2S")
        assert comp['Li'] == pytest.approx(2.0)
        assert comp['S'] == pytest.approx(1.0)

    def test_multidigit_subscripts(self):
        comp = parse_complex_formula("Li7La3Zr2O12")
        assert comp['Li'] == pytest.approx(7.0)
        assert comp['La'] == pytest.approx(3.0)
        assert comp['Zr'] == pytest.approx(2.0)
        assert comp['O'] == pytest.approx(12.0)

    def test_single_element(self):
        comp = parse_complex_formula("Li")
        assert comp == {'Li': 1.0}

    def test_decimal_subscripts(self):
        comp = parse_complex_formula("Li0.5Na0.5Cl")
        assert comp['Li'] == pytest.approx(0.5)
        assert comp['Na'] == pytest.approx(0.5)
        assert comp['Cl'] == pytest.approx(1.0)


# ===================================================================
# parse_complex_formula — nested / parenthesized
# ===================================================================

class TestParseComplexFormulaNested:
    def test_simple_parentheses(self):
        comp = parse_complex_formula("Ca(OH)2")
        assert comp['Ca'] == pytest.approx(1.0)
        assert comp['O'] == pytest.approx(2.0)
        assert comp['H'] == pytest.approx(2.0)

    def test_nested_parentheses(self):
        comp = parse_complex_formula("((Li2S)0.75(P2S5)0.25)")
        assert comp['Li'] == pytest.approx(2 * 0.75)
        assert comp['P'] == pytest.approx(2 * 0.25)
        expected_S = 1 * 0.75 + 5 * 0.25
        assert comp['S'] == pytest.approx(expected_S)

    def test_deeply_nested(self):
        """((Li2S)0.75(P2S5)0.25)96(P2O5)4"""
        comp = parse_complex_formula("((Li2S)0.75(P2S5)0.25)96(P2O5)4")
        assert comp['Li'] == pytest.approx(0.75 * 96 * 2)
        assert comp['P'] == pytest.approx(0.25 * 96 * 2 + 4 * 2)
        assert comp['S'] == pytest.approx(0.75 * 96 * 1 + 0.25 * 96 * 5)
        assert comp['O'] == pytest.approx(4 * 5)

    def test_adjacent_groups(self):
        comp = parse_complex_formula("(Li2)(S3)")
        assert comp['Li'] == pytest.approx(2.0)
        assert comp['S'] == pytest.approx(3.0)

    def test_unmatched_parenthesis_raises(self):
        with pytest.raises(ValueError, match="Unmatched parenthesis"):
            parse_complex_formula("(Li2S")


# ===================================================================
# parse_mixture_formula
# ===================================================================

class TestParseMixtureFormula:
    def test_single_compound_passthrough(self):
        comp = parse_mixture_formula("Li3InCl6")
        assert comp['Li'] == pytest.approx(3.0)
        assert comp['In'] == pytest.approx(1.0)
        assert comp['Cl'] == pytest.approx(6.0)

    def test_mixture_with_dash(self):
        """0.75LiBH4-0.25Ca(BH4)2"""
        comp = parse_mixture_formula("0.75LiBH4-0.25Ca(BH4)2")
        assert comp['Li'] == pytest.approx(0.75)
        assert comp['Ca'] == pytest.approx(0.25)
        # B: 0.75*1 + 0.25*2 = 1.25
        assert comp['B'] == pytest.approx(1.25)

    def test_empty_formula(self):
        assert parse_mixture_formula("") == {}

    def test_none_formula(self):
        assert parse_mixture_formula(None) == {}

    def test_descriptor_stripped_before_parse(self):
        comp = parse_mixture_formula("Li3InCl6 (ball milled)")
        assert comp['Li'] == pytest.approx(3.0)
        assert comp['In'] == pytest.approx(1.0)

    def test_all_elements_in_periodic_table(self):
        """Every element returned by parsing standard formulas should be known."""
        formulas = ['Li3InCl6', 'Li6PS5Cl', 'Li7La3Zr2O12', 'NaTaOCl4',
                     'Li2S', 'LiPF6', 'Li10GeP2S12']
        for f in formulas:
            comp = parse_mixture_formula(f)
            for elem in comp:
                assert elem in PERIODIC_TABLE, (
                    f"Element {elem} from '{f}' not in PERIODIC_TABLE"
                )

    def test_total_atoms_positive(self):
        """Parsed composition should always have positive atom counts."""
        formulas = ['Li3InCl6', 'NaCl', '((Li2S)0.75(P2S5)0.25)96(P2O5)4']
        for f in formulas:
            comp = parse_mixture_formula(f)
            assert all(v > 0 for v in comp.values()), f"Non-positive count in {f}: {comp}"


# ===================================================================
# PERIODIC_TABLE completeness
# ===================================================================

class TestPeriodicTable:
    def test_common_battery_elements_present(self):
        """All elements commonly found in solid-state electrolytes must be present."""
        required = ['Li', 'Na', 'K', 'O', 'S', 'Cl', 'F', 'Br', 'I',
                     'P', 'Si', 'Ge', 'Sn', 'La', 'Zr', 'Ta', 'In',
                     'Al', 'Ti', 'Ga', 'Nb', 'Ba', 'Ca', 'Mg', 'B', 'N']
        for elem in required:
            assert elem in PERIODIC_TABLE, f"{elem} missing from PERIODIC_TABLE"

    def test_properties_complete(self):
        """Each element must have all four properties."""
        for elem, props in PERIODIC_TABLE.items():
            for prop in ['atomic_mass', 'electronegativity', 'ionic_radius', 'group']:
                assert prop in props, f"{elem} missing property '{prop}'"

    def test_atomic_mass_positive(self):
        for elem, props in PERIODIC_TABLE.items():
            assert props['atomic_mass'] > 0, f"{elem} has non-positive atomic mass"

    def test_electronegativity_positive(self):
        for elem, props in PERIODIC_TABLE.items():
            assert props['electronegativity'] > 0, f"{elem} has non-positive EN"
