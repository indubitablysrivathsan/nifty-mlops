"""Shared helpers: metrics, data loading/prep, MLflow client wrappers."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config


def load_model_df() -> pd.DataFrame:
    """
    Loads the DVC-tracked feature snapshot and prepares the modelling frame:
    index by trade_date, drop rows with a missing target or any missing
    candidate feature (same as model_df construction in modelling_20d.ipynb).
    """
    df = pd.read_parquet(config.FEATURES_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()

    required_cols = [config.TARGET] + config.BASE_FEATURE_COLS
    df = df.dropna(subset=required_cols)
    return df


def ic_metrics(preds: np.ndarray, y_true: pd.Series) -> dict:
    if len(y_true) <= 2:
        return {"ic_pearson": np.nan, "ic_spearman": np.nan, "rmse": np.nan, "dir_acc": np.nan}
    if np.allclose(preds, preds[0]) or np.allclose(y_true, y_true.iloc[0]):
        pearson, spear = np.nan, np.nan
    else:
        pearson = np.corrcoef(preds, y_true)[0, 1]
        spear = spearmanr(preds, y_true)[0]
    return {
        "ic_pearson": pearson,
        "ic_spearman": spear,
        "rmse": float(np.sqrt(np.mean((preds - y_true) ** 2))),
        "dir_acc": float((np.sign(preds) == np.sign(y_true)).mean()),
    }


def summarize_fold_results(fold_rows: list[dict]) -> dict:
    df = pd.DataFrame(fold_rows)
    n = df["ic_pearson"].count()
    mean_ic = df["ic_pearson"].mean()
    std_ic = df["ic_pearson"].std()
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if n > 1 and std_ic and std_ic > 0 else np.nan
    positive_pct = (df["ic_pearson"] > 0).mean()
    return {
        "n_folds": int(n),
        "mean_ic_pearson": float(mean_ic),
        "mean_ic_spearman": float(df["ic_spearman"].mean()),
        "ic_t_stat": float(t_stat) if pd.notna(t_stat) else None,
        "positive_fold_pct": float(positive_pct),
        "mean_rmse": float(df["rmse"].mean()),
        "mean_dir_acc": float(df["dir_acc"].mean()),
    }
