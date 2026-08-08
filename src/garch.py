"""
Per-fold GARCH(1,1) volatility feature — identical logic to simulation.ipynb.
Fit strictly on that fold's training returns only; the test-period sigma
path is a genuine out-of-sample forecast (res.forecast), not re-fit on test
data, so this introduces no look-ahead.
"""
import numpy as np
import pandas as pd
from arch import arch_model

from src import config


def compute_garch_features(daily_ret: pd.Series, train_idx: pd.DatetimeIndex,
                             test_idx: pd.DatetimeIndex):
    """
    daily_ret: log return series (nifty_ret_1d), NOT yet scaled to percent.
    Returns (train_feat, test_feat) DataFrames with columns
    ['garch_sigma', 'garch_sigma_chg'], indexed like model_df.
    """
    train_ret = daily_ret.loc[train_idx].dropna() * 100

    try:
        am = arch_model(train_ret, vol="Garch", p=config.GARCH_P, q=config.GARCH_Q,
                         dist=config.GARCH_DIST)
        res = am.fit(disp="off")
    except Exception:
        nan_train = pd.DataFrame(np.nan, index=train_idx, columns=["garch_sigma", "garch_sigma_chg"])
        nan_test = nan_train.reindex(test_idx)
        return nan_train, nan_test

    sigma_train = res.conditional_volatility / 100
    sigma_change_train = sigma_train.diff()

    train_feat = pd.DataFrame({
        "garch_sigma": sigma_train.values,
        "garch_sigma_chg": sigma_change_train.values,
    }, index=train_ret.index)

    fcast = res.forecast(horizon=len(test_idx), reindex=False)
    sigma_path = np.sqrt(fcast.variance.values[-1]) / 100

    test_feat = pd.DataFrame({
        "garch_sigma": sigma_path,
        "garch_sigma_chg": np.diff(sigma_path, prepend=sigma_train.iloc[-1]),
    }, index=test_idx[:len(sigma_path)])

    return train_feat, test_feat


def fit_garch_for_all_folds(model_df: pd.DataFrame, folds, price_col="nifty_close", n_jobs=-1):
    """Precomputes GARCH features for every fold, in parallel (each fold's
    GARCH fit is independent of the others). Returns dict[fold_id] -> (train_feat, test_feat)."""
    from joblib import Parallel, delayed

    daily_ret = np.log(model_df[price_col]).diff()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_garch_features)(daily_ret, f.train_idx, f.test_idx) for f in folds
    )
    return {f.fold_id: result for f, result in zip(folds, results)}


def fit_garch_final(model_df: pd.DataFrame, price_col="nifty_close") -> pd.DataFrame:
    """
    For fit_final.py: fit GARCH once on ALL historical returns (this is the
    single final production fit, not a walk-forward fold — there is no
    "future" data left to leak from at this point, since it trains on
    everything up to "now").
    """
    daily_ret = np.log(model_df[price_col]).diff()
    ret_pct = daily_ret.dropna() * 100
    am = arch_model(ret_pct, vol="Garch", p=config.GARCH_P, q=config.GARCH_Q, dist=config.GARCH_DIST)
    res = am.fit(disp="off")

    sigma = res.conditional_volatility / 100
    sigma_chg = sigma.diff()
    feat = pd.DataFrame({"garch_sigma": sigma.values, "garch_sigma_chg": sigma_chg.values},
                         index=ret_pct.index)
    return feat, res
