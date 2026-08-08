"""
Central configuration for the pipeline.

Everything that used to be a hardcoded constant scattered across notebook
cells (feature groups, fold parameters, model hyperparameters, promotion
thresholds) lives here so train.py / fit_final.py / extract_features.py /
app.py all agree on the same values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw scraped DuckDB (candL project output). Only needed for extract_features.py.
# Never required by train.py / fit_final.py / app.py — those only touch the
# already-extracted feature snapshot below.
NSE_DB_PATH = Path(os.environ.get("NSE_DB_PATH", PROJECT_ROOT / "data" / "raw" / "nse.db"))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PATH = PROCESSED_DIR / "daily_features.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "model.pkl"
FINAL_MODEL_META_PATH = MODELS_DIR / "model_meta.json"

# sqlite backend (not the plain-file store) so the MLflow Model Registry
# is actually usable — the file store is registry-incapable in recent
# MLflow versions without extra migration steps.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "nifty_fwd_ret_20d")
MLFLOW_REGISTRY_MODEL_NAME = os.environ.get("MLFLOW_REGISTRY_MODEL_NAME", "nifty_fwd_ret_20d_model")

# ── Target ─────────────────────────────────────────────────────────────────
TARGET = "fwd_ret_20d"
HORIZON_DAYS = 20

# ── Feature groups (verbatim from feature_analysis.ipynb / modelling_20d.ipynb) ──
FEATURE_GROUPS = {
    "price_vol": ["nifty_close", "vix_close", "underlying_daily_vol",
                  "futures_daily_vol", "applicable_daily_vol"],
    "options_derived": ["pcr", "max_pain_dist_pct", "basis", "cost_of_carry", "fut_chng_oi_pct"],
    "breadth": ["advances", "declines", "adv_decl_ratio", "price_band_hits",
                "traded_value_cr", "num_trades", "market_cap_cr"],
    "fii_dii_positioning": ["fii_fut_net_pct", "client_fut_net_pct", "fii_pe_net_pct",
                             "fii_net_flow_cr", "dii_fut_net_pct", "pro_fut_net_pct",
                             "fii_stk_fut_net_pct", "client_stk_fut_net_pct"],
    "options_skew": ["fii_opt_skew_pct", "client_opt_skew_pct", "dii_opt_skew_pct",
                      "pro_opt_skew_pct", "fii_vol_opt_skew_pct", "client_vol_opt_skew_pct"],
    "vol_futures_flow": ["fii_vol_fut_net_pct", "client_vol_fut_net_pct", "pro_vol_fut_net_pct"],
    "divergence_stats": ["fii_client_divergence", "fii_stats_fut_net_pct", "fii_stats_oi_net_cr"],
    "engineered_deltas": ["vix_chg_5d", "pcr_chg_5d", "basis_chg_5d", "max_pain_dist_chg_5d",
                           "vix_realized_spread", "fii_fut_net_chg_5d"],
}

# Emission-raw features previously reserved for the (now-abandoned) HMM. HMM
# is not used in this pipeline at all, so these are simply folded back in as
# ordinary features — see working paper §3.3 for why the HMM state itself
# (not these raw inputs) was the leaky part.
EMISSION_RAW = ["vix_close", "underlying_daily_vol", "price_band_hits",
                "fii_stats_fut_net_pct", "client_opt_skew_pct",
                "max_pain_dist_pct", "basis_chg_5d"]

# Confounds identified via detrended-correlation check in feature analysis —
# excluded from the model feature set (see working paper §4.2).
CONFOUNDS = ["nifty_close", "market_cap_cr", "num_trades"]

ALL_RAW_FEATURES = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})
BASE_FEATURE_COLS = sorted(c for c in ALL_RAW_FEATURES if c not in CONFOUNDS)

GARCH_FEATURE_COLS = ["garch_sigma", "garch_sigma_chg"]

# ── Feature-set variants ──────────────────────────────────────────────────
FEATURE_VARIANTS = {
    "baseline_full": BASE_FEATURE_COLS,
    "baseline_full_garch": BASE_FEATURE_COLS + GARCH_FEATURE_COLS,
}

# ── Walk-forward fold parameters (identical to modelling_20d.ipynb) ────────
MIN_TRAIN_EXPIRIES = 24
EMBARGO_ENTRIES = 20
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 5

# ── GARCH(1,1) fit parameters (identical to simulation.ipynb) ──────────────
GARCH_P, GARCH_Q = 1, 1
GARCH_DIST = "t"

# ── Model hyperparameters (fixed, not tuned per fold — same as notebooks) ──
MODEL_PARAMS = {
    "random_forest": dict(
        n_estimators=300, max_depth=4, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    ),
    "xgboost": dict(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    ),
    "lightgbm": dict(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    ),
    "catboost": dict(
        iterations=300, depth=4, learning_rate=0.03,
        l2_leaf_reg=1.0, subsample=0.8, random_seed=42,
        verbose=False,
    ),
}

# ── Promotion gate (registry + deployment) ──────────────────────────────────
PROMOTION_MIN_MEAN_IC = float(os.environ.get("PROMOTION_MIN_MEAN_IC", 0.05))
PROMOTION_MIN_POSITIVE_FOLD_PCT = float(os.environ.get("PROMOTION_MIN_POSITIVE_FOLD_PCT", 0.55))
# A challenger must not be worse than the current production model's mean IC
# by more than this tolerance (0.0 = must be >=, small positive value gives
# a little slack against noise).
PROMOTION_REGRESSION_TOLERANCE = float(os.environ.get("PROMOTION_REGRESSION_TOLERANCE", 0.0))

MLFLOW_MODEL_STAGE_PRODUCTION = "Production"
