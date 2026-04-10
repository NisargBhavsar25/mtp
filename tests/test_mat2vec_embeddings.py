"""Tests for mat2vec embedding generation.

Special focus on detecting zero-array embeddings — a common silent failure
where mat2vec returns np.zeros(dim) because an element is missing from the
pretrained vocabulary or the formula cannot be parsed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock
import os

try:
    from pymatgen.core import Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


# ===================================================================
# Helper: build a fake Word2Vec-like model
# ===================================================================

class FakeWordVectors:
    """Mimics gensim's KeyedVectors for testing."""

    def __init__(self, known_elements, dim=200):
        self.dim = dim
        self._vectors = {}
        rng = np.random.default_rng(42)
        for elem in known_elements:
            self._vectors[elem] = rng.standard_normal(dim).astype(np.float32)
        self.key_to_index = {k: i for i, k in enumerate(known_elements)}

    def __contains__(self, key):
        return key in self._vectors

    def __getitem__(self, key):
        return self._vectors[key]


class FakeWord2VecModel:
    """Mimics a gensim Word2Vec model."""

    def __init__(self, known_elements, dim=200):
        self.wv = FakeWordVectors(known_elements, dim)


# ===================================================================
# Tests for zero-array detection in training path
# ===================================================================

@pytest.mark.skipif(not HAS_PYMATGEN, reason="pymatgen not installed")
class TestMat2VecZeroArrayDetection:
    """Ensure the pipeline properly detects / handles zero embeddings."""

    @pytest.fixture
    def known_elements(self):
        return ['Li', 'Na', 'Cl', 'O', 'S', 'P', 'In', 'La', 'Zr',
                'Ge', 'F', 'Ta', 'B', 'H', 'Ca', 'N']

    @pytest.fixture
    def fake_model(self, known_elements):
        return FakeWord2VecModel(known_elements, dim=200)

    def _generate_embedding(self, formula, model, dim=200):
        """Replicate the embedding logic from train_best_save.py."""
        from pymatgen.core import Composition
        try:
            comp = Composition(str(formula))
            elements = comp.get_el_amt_dict()

            token_embeddings = []
            weights = []
            for element, amount in elements.items():
                if element in model.wv:
                    token_embeddings.append(model.wv[element])
                    weights.append(amount)

            if token_embeddings:
                emb = np.average(token_embeddings, axis=0, weights=weights)
                if len(emb) != dim:
                    resized = np.zeros(dim)
                    m = min(len(emb), dim)
                    resized[:m] = emb[:m]
                    return resized
                return emb
            else:
                return np.zeros(dim)
        except Exception:
            return np.zeros(dim)

    def test_known_formula_nonzero(self, fake_model):
        """Standard formula with all known elements → non-zero embedding."""
        emb = self._generate_embedding('Li3InCl6', fake_model)
        assert emb.shape == (200,)
        assert not np.allclose(emb, 0), "Embedding should NOT be all zeros for Li3InCl6"

    def test_unknown_element_returns_zero(self, fake_model):
        """Formula with ALL unknown elements → zero embedding."""
        # 'Xe' and 'Kr' are not in our fake model's vocabulary
        emb = self._generate_embedding('XeKr', fake_model)
        assert np.allclose(emb, 0), "Should be zero array when no elements are in vocab"

    def test_partial_unknown_still_nonzero(self, fake_model):
        """Formula with some known + some unknown elements → non-zero (uses known only)."""
        # 'Li' is known, 'Xe' is not
        emb = self._generate_embedding('LiXe', fake_model)
        # Should still get Li embedding (weighted by 1 atom)
        assert not np.allclose(emb, 0), "Should use known elements even if some are unknown"

    def test_empty_formula_returns_zero(self, fake_model):
        """Empty or unparseable formula → zero embedding."""
        emb = self._generate_embedding('', fake_model)
        assert np.allclose(emb, 0)

    def test_invalid_formula_returns_zero(self, fake_model):
        """Gibberish formula → zero embedding."""
        emb = self._generate_embedding('!!!???', fake_model)
        assert np.allclose(emb, 0)

    def test_embedding_dimension_correct(self, fake_model):
        """All embeddings must have exactly 200 dimensions."""
        formulas = ['Li2S', 'NaCl', 'Li7La3Zr2O12', 'Li10GeP2S12']
        for f in formulas:
            emb = self._generate_embedding(f, fake_model)
            assert emb.shape == (200,), f"Wrong dim for {f}: {emb.shape}"

    def test_weighted_average_correct(self, fake_model):
        """Verify that stoichiometric weighting is applied correctly."""
        # Li2S: 2 atoms Li, 1 atom S
        emb = self._generate_embedding('Li2S', fake_model)
        expected = np.average(
            [fake_model.wv['Li'], fake_model.wv['S']],
            axis=0,
            weights=[2, 1]
        )
        np.testing.assert_allclose(emb, expected, rtol=1e-5)

    def test_batch_zero_array_fraction(self, fake_model):
        """In a realistic batch, count what fraction returns zero arrays."""
        formulas = [
            'Li3InCl6', 'Li2S', 'NaCl', 'Li7La3Zr2O12',
            'XeKr2',  # all unknown → zero
            'Li10GeP2S12', 'LiPF6',
            'RaRn',  # all unknown → zero
        ]
        embeddings = [self._generate_embedding(f, fake_model) for f in formulas]
        zero_count = sum(1 for e in embeddings if np.allclose(e, 0))

        # We expect exactly 2 zero arrays (XeKr2 and RaRn)
        assert zero_count == 2, f"Expected 2 zero arrays, got {zero_count}"

        # Verify non-zero ones are actually non-zero
        for f, e in zip(formulas, embeddings):
            if f not in ('XeKr2', 'RaRn'):
                assert not np.allclose(e, 0), f"{f} should have non-zero embedding"


# ===================================================================
# Tests for zero-array detection in inference path
# ===================================================================

class TestInferenceMat2VecZeroArray:
    """Test the Mat2VecGenerator from predict_properties.py."""

    @pytest.fixture
    def fake_model(self):
        known = ['Li', 'Na', 'Cl', 'O', 'S', 'P', 'In', 'La', 'Zr',
                 'Ge', 'F', 'Ta', 'B', 'H', 'Ca', 'N']
        return FakeWord2VecModel(known, dim=200)

    def _make_generator(self, fake_model):
        """Build a Mat2VecGenerator with the fake model injected."""
        from src.inference.predict_properties import Mat2VecGenerator
        gen = Mat2VecGenerator.__new__(Mat2VecGenerator)
        gen.model = fake_model
        gen.dim = 200
        return gen

    def test_known_formula_nonzero(self, fake_model):
        gen = self._make_generator(fake_model)
        emb = gen.get_embedding('Li3InCl6')
        assert emb.shape == (200,)
        assert not np.allclose(emb, 0)

    def test_unknown_formula_returns_zero(self, fake_model):
        gen = self._make_generator(fake_model)
        emb = gen.get_embedding('XeKr')
        assert np.allclose(emb, 0), "Unknown elements should produce zero array"

    def test_no_model_loaded_returns_zero(self):
        """When Mat2Vec model fails to load → zero array."""
        from src.inference.predict_properties import Mat2VecGenerator
        gen = Mat2VecGenerator.__new__(Mat2VecGenerator)
        gen.model = None
        gen.dim = 200
        emb = gen.get_embedding('Li3InCl6')
        assert np.allclose(emb, 0)

    def test_embedding_shape_always_consistent(self, fake_model):
        gen = self._make_generator(fake_model)
        formulas = ['Li2S', 'NaCl', '', 'XeKr', 'Li10GeP2S12']
        for f in formulas:
            emb = gen.get_embedding(f)
            assert emb.shape == (200,), f"Shape mismatch for '{f}': {emb.shape}"


# ===================================================================
# Tests for dimension mismatch handling
# ===================================================================

class TestDimensionMismatch:
    def test_model_dim_smaller_than_target(self):
        """If pretrained model has dim=100 but we target dim=200, pad with zeros."""
        model_100d = FakeWord2VecModel(['Li', 'Cl'], dim=100)
        from src.inference.predict_properties import Mat2VecGenerator
        gen = Mat2VecGenerator.__new__(Mat2VecGenerator)
        gen.model = model_100d
        gen.dim = 200

        emb = gen.get_embedding('LiCl')
        assert emb.shape == (200,)
        # First 100 dims should be non-zero (from actual embedding)
        assert not np.allclose(emb[:100], 0)
        # Last 100 dims should be zero (padding)
        assert np.allclose(emb[100:], 0)

    def test_model_dim_larger_than_target(self):
        """If pretrained model has dim=300 but we target dim=200, truncate."""
        model_300d = FakeWord2VecModel(['Li', 'Cl'], dim=300)
        from src.inference.predict_properties import Mat2VecGenerator
        gen = Mat2VecGenerator.__new__(Mat2VecGenerator)
        gen.model = model_300d
        gen.dim = 200

        emb = gen.get_embedding('LiCl')
        assert emb.shape == (200,)
        assert not np.allclose(emb, 0)
