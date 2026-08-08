import pandas as pd
import pytest

from src import config
from src.folds import build_folds


def _fake_model_df(n_days=1800, start="2016-01-01"):
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({config.TARGET: range(n_days)}, index=idx)


def _fake_expiries(n_months=90, start="2016-01-01"):
    # last business day of each month, standing in for monthly expiry
    months = pd.date_range(start, periods=n_months, freq="ME")
    return pd.Series(pd.to_datetime(months))


def test_build_folds_no_lookahead():
    model_df = _fake_model_df()
    expiries = _fake_expiries()
    folds = build_folds(model_df, expiries)

    assert len(folds) > 0
    for f in folds:
        # every training date must be at least EMBARGO_DAYS before train_end
        assert f.train_idx.max() <= f.train_end - pd.Timedelta(days=config.EMBARGO_DAYS)
        # test dates must fall strictly after test_start and up to test_end
        assert f.test_idx.min() > f.test_start
        assert f.test_idx.max() <= f.test_end
        # no overlap between this fold's train and test sets
        assert len(f.train_idx.intersection(f.test_idx)) == 0


def test_build_folds_respects_min_train_expiries():
    model_df = _fake_model_df()
    expiries = _fake_expiries()
    folds = build_folds(model_df, expiries)

    first_fold = folds[0]
    # first fold's train_end should be the (MIN_TRAIN_EXPIRIES)-th expiry
    assert first_fold.train_end == expiries.iloc[config.MIN_TRAIN_EXPIRIES]


def test_build_folds_drops_small_folds():
    # very short history should produce zero usable folds
    model_df = _fake_model_df(n_days=50)
    expiries = _fake_expiries(n_months=90)
    folds = build_folds(model_df, expiries)
    assert folds == []
