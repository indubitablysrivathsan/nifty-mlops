"""
Expiry-aligned walk-forward folds — identical logic to modelling_20d.ipynb
(cell 5): train up to 20 trading observations before expiry[i], test between expiry[i] and
expiry[i+1]. Reads monthly expiry boundaries from the parquet exported by
extract_features.py rather than querying nse.db directly, so this module has
no raw-database dependency.
"""
from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass
class Fold:
    fold_id: int
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_monthly_expiries() -> pd.Series:
    path = config.PROCESSED_DIR / "monthly_expiries.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_features` first."
        )
    return pd.to_datetime(pd.read_parquet(path)["expiry"])


def build_folds(model_df: pd.DataFrame, monthly_expiries: pd.Series = None) -> list[Fold]:
    """
    model_df must be indexed by trade_date (DatetimeIndex), already filtered
    to rows with a non-null target and feature set.
    """
    if monthly_expiries is None:
        monthly_expiries = load_monthly_expiries()

    valid_expiries = monthly_expiries[monthly_expiries <= model_df.index.max()].reset_index(drop=True)

    folds = []
    for i in range(config.MIN_TRAIN_EXPIRIES, len(valid_expiries) - 1):
        train_end = valid_expiries.iloc[i]
        test_start = valid_expiries.iloc[i]
        test_end = valid_expiries.iloc[i + 1]

        expiry_pos = model_df.index.searchsorted(train_end, side="right") - 1
        embargo_pos = expiry_pos - config.EMBARGO_ENTRIES
        if embargo_pos < 0:
            continue

        train_end_embargoed = model_df.index[embargo_pos]
        train_mask = model_df.index <= train_end_embargoed
        test_mask = (model_df.index > test_start) & (model_df.index <= test_end)

        train_idx = model_df.index[train_mask]
        test_idx = model_df.index[test_mask]

        if len(train_idx) < config.MIN_TRAIN_ROWS or len(test_idx) < config.MIN_TEST_ROWS:
            continue

        folds.append(Fold(
            fold_id=len(folds),
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))

    return folds
