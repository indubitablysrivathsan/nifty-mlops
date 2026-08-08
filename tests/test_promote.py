from unittest.mock import patch

from src import config
from src.promote import evaluate_gate


def _base_challenger(**overrides):
    challenger = {
        "n_folds": 80,
        "mean_ic_pearson": config.PROMOTION_MIN_MEAN_IC + 0.02,
        "positive_fold_pct": config.PROMOTION_MIN_POSITIVE_FOLD_PCT + 0.05,
    }
    challenger.update(overrides)
    return challenger


@patch("src.promote.get_current_production_ic", return_value=None)
def test_gate_passes_with_no_prior_production_model(_mock):
    passed, reasons = evaluate_gate(_base_challenger())
    assert passed
    assert reasons == []


@patch("src.promote.get_current_production_ic", return_value=None)
def test_gate_fails_below_ic_threshold(_mock):
    challenger = _base_challenger(mean_ic_pearson=config.PROMOTION_MIN_MEAN_IC - 0.01)
    passed, reasons = evaluate_gate(challenger)
    assert not passed
    assert any("mean_ic_pearson" in r for r in reasons)


@patch("src.promote.get_current_production_ic", return_value=None)
def test_gate_fails_below_positive_fold_pct(_mock):
    challenger = _base_challenger(positive_fold_pct=config.PROMOTION_MIN_POSITIVE_FOLD_PCT - 0.1)
    passed, reasons = evaluate_gate(challenger)
    assert not passed
    assert any("positive_fold_pct" in r for r in reasons)


@patch("src.promote.get_current_production_ic", return_value=0.20)
def test_gate_fails_if_worse_than_current_production(_mock):
    challenger = _base_challenger(mean_ic_pearson=0.10)
    passed, reasons = evaluate_gate(challenger)
    assert not passed
    assert any("worse than current Production" in r for r in reasons)


@patch("src.promote.get_current_production_ic", return_value=0.05)
def test_gate_passes_if_better_than_current_production(_mock):
    challenger = _base_challenger(mean_ic_pearson=0.10)
    passed, reasons = evaluate_gate(challenger)
    assert passed


def test_gate_fails_with_zero_folds():
    challenger = _base_challenger(n_folds=0)
    with patch("src.promote.get_current_production_ic", return_value=None):
        passed, reasons = evaluate_gate(challenger)
    assert not passed
    assert any("0 folds" in r for r in reasons)
