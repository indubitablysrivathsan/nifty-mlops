import numpy as np
import pandas as pd
import pytest

from src.utils import ic_metrics, summarize_fold_results


def test_ic_metrics_perfect_correlation():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = ic_metrics(preds, y)
    assert m["ic_pearson"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["dir_acc"] == pytest.approx(1.0)


def test_ic_metrics_constant_predictions_returns_nan():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    preds = np.array([1.0, 1.0, 1.0, 1.0])
    m = ic_metrics(preds, y)
    assert np.isnan(m["ic_pearson"])


def test_ic_metrics_too_few_points():
    y = pd.Series([1.0, 2.0])
    preds = np.array([1.0, 2.0])
    m = ic_metrics(preds, y)
    assert np.isnan(m["ic_pearson"])


def test_summarize_fold_results():
    rows = [
        {"ic_pearson": 0.1, "ic_spearman": 0.1, "rmse": 1.0, "dir_acc": 0.5},
        {"ic_pearson": 0.2, "ic_spearman": 0.2, "rmse": 1.2, "dir_acc": 0.6},
        {"ic_pearson": -0.05, "ic_spearman": -0.02, "rmse": 0.9, "dir_acc": 0.4},
    ]
    summary = summarize_fold_results(rows)
    assert summary["n_folds"] == 3
    assert summary["positive_fold_pct"] == pytest.approx(2 / 3)
    assert summary["mean_ic_pearson"] == pytest.approx(np.mean([0.1, 0.2, -0.05]))
