"""Tests for src/evaluation/calculate_metrics.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from src.evaluation.calculate_metrics import calculate_metrics


class TestCalculateMetrics:
    def test_perfect_prediction(self):
        """When predictions match exactly, R²=1, MAE=0, RMSE=0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = calculate_metrics(y, y)
        assert metrics['R_adj^2'] == pytest.approx(1.0)
        assert metrics['MAE'] == pytest.approx(0.0)
        assert metrics['RMSE'] == pytest.approx(0.0)
        assert metrics['MBE'] == pytest.approx(0.0)

    def test_constant_offset(self):
        """Predictions have a constant bias."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true + 0.5
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['MBE'] == pytest.approx(0.5)
        assert metrics['MAE'] == pytest.approx(0.5)
        assert metrics['RMSE'] == pytest.approx(0.5)

    def test_n_correct(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = calculate_metrics(y, y)
        assert metrics['N'] == 3

    def test_mtv_mpv(self):
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([3.0, 5.0])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['MTV'] == pytest.approx(3.0)  # mean(2, 4)
        assert metrics['MPV'] == pytest.approx(4.0)  # mean(3, 5)

    def test_std_of_errors(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = calculate_metrics(y_true, y_pred)
        errors = np.array([1.0, 2.0, 3.0])
        assert metrics['STD'] == pytest.approx(np.std(errors, ddof=1))

    def test_negative_r2_possible(self):
        """Model worse than predicting the mean → R² < 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # way off
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['R_adj^2'] < 0

    def test_adjusted_r2_formula(self):
        """Check adj R² matches the formula: 1 - (1-R²)(n-1)/(n-p-1)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        metrics = calculate_metrics(y_true, y_pred)

        n = 5
        p = 1
        errors = y_pred - y_true
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot
        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        assert metrics['R_adj^2'] == pytest.approx(r2_adj)

    def test_rmse_always_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['RMSE'] >= 0

    def test_mae_always_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['MAE'] >= 0

    def test_mae_leq_rmse(self):
        """MAE <= RMSE is always true (Cauchy-Schwarz)."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = y_true + rng.standard_normal(100) * 0.5
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['MAE'] <= metrics['RMSE'] + 1e-10
