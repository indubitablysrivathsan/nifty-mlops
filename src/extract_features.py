"""
Recreates `daily_features` exactly as built in exploration.ipynb, using
DuckDB directly against the raw scraped nse.db (candL project output).

Usage:
    python -m src.extract_features
    (reads NSE_DB_PATH from .env, writes data/processed/daily_features.parquet)

This script does not select, engineer, or drop anything beyond what the
original notebook did — it is a straight port of the SQL cells so the
feature table used in this repo matches the one used in the research.
"""
import logging

import duckdb

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── SQL: forward returns for NIFTY 50 ───────────────────────────────────────
SQL_FORWARD_RETURNS = """
CREATE OR REPLACE TABLE forward_returns AS
WITH daily AS (
    SELECT trade_date, index_name, close
    FROM nse.market_activity_index
    WHERE index_name = 'NIFTY 50'
    ORDER BY trade_date
)
SELECT
    d.trade_date,
    d.index_name,
    d.close,
    ROUND((f1.close - d.close) / d.close * 100, 4) AS fwd_ret_1d,
    ROUND((f5.close - d.close) / d.close * 100, 4) AS fwd_ret_5d,
    ROUND((f20.close - d.close) / d.close * 100, 4) AS fwd_ret_20d,
    CASE WHEN f1.close > d.close THEN 1 ELSE 0 END AS up_1d
FROM daily d
LEFT JOIN daily f1
    ON f1.trade_date = (SELECT MIN(trade_date) FROM daily WHERE trade_date > d.trade_date)
LEFT JOIN daily f5
    ON f5.trade_date = (SELECT trade_date FROM daily WHERE trade_date > d.trade_date
                         ORDER BY trade_date LIMIT 1 OFFSET 4)
LEFT JOIN daily f20
    ON f20.trade_date = (SELECT trade_date FROM daily WHERE trade_date > d.trade_date
                          ORDER BY trade_date LIMIT 1 OFFSET 19)
;
"""

# ── SQL: participant_activity pivot (FII/Client/DII/Pro, OI + VOL) ─────────
SQL_PARTICIPANT_PIVOT = """
CREATE OR REPLACE TABLE participant_pivot AS
WITH base AS (
    SELECT
        trade_date, participant_type, metric_type,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='NA' AND direction='long'  THEN contracts ELSE 0 END) AS idx_fut_long,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='NA' AND direction='short' THEN contracts ELSE 0 END) AS idx_fut_short,
        SUM(CASE WHEN asset_class='STOCK' AND option_side='NA' AND direction='long'  THEN contracts ELSE 0 END) AS stk_fut_long,
        SUM(CASE WHEN asset_class='STOCK' AND option_side='NA' AND direction='short' THEN contracts ELSE 0 END) AS stk_fut_short,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='CE' AND direction='long'  THEN contracts ELSE 0 END) AS idx_ce_long,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='CE' AND direction='short' THEN contracts ELSE 0 END) AS idx_ce_short,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='PE' AND direction='long'  THEN contracts ELSE 0 END) AS idx_pe_long,
        SUM(CASE WHEN asset_class='INDEX' AND option_side='PE' AND direction='short' THEN contracts ELSE 0 END) AS idx_pe_short
    FROM nse.participant_activity
    GROUP BY trade_date, participant_type, metric_type
),
derived AS (
    SELECT
        trade_date, participant_type, metric_type,
        ROUND(100.0 * (idx_fut_long - idx_fut_short) / NULLIF(idx_fut_long + idx_fut_short, 0), 2) AS idx_fut_net_pct,
        ROUND(100.0 * (stk_fut_long - stk_fut_short) / NULLIF(stk_fut_long + stk_fut_short, 0), 2) AS stk_fut_net_pct,
        ROUND(100.0 * ((idx_ce_long - idx_ce_short) - (idx_pe_long - idx_pe_short))
              / NULLIF((idx_ce_long + idx_ce_short + idx_pe_long + idx_pe_short), 0), 2) AS idx_opt_skew_pct
    FROM base
)
SELECT
    trade_date,
    MAX(CASE WHEN participant_type='FII'    AND metric_type='OI'  THEN idx_fut_net_pct END) AS fii_oi_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='Client' AND metric_type='OI'  THEN idx_fut_net_pct END) AS client_oi_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='DII'    AND metric_type='OI'  THEN idx_fut_net_pct END) AS dii_oi_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='Pro'    AND metric_type='OI'  THEN idx_fut_net_pct END) AS pro_oi_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='FII'    AND metric_type='OI'  THEN idx_opt_skew_pct END) AS fii_oi_idx_opt_skew_pct,
    MAX(CASE WHEN participant_type='Client' AND metric_type='OI'  THEN idx_opt_skew_pct END) AS client_oi_idx_opt_skew_pct,
    MAX(CASE WHEN participant_type='DII'    AND metric_type='OI'  THEN idx_opt_skew_pct END) AS dii_oi_idx_opt_skew_pct,
    MAX(CASE WHEN participant_type='Pro'    AND metric_type='OI'  THEN idx_opt_skew_pct END) AS pro_oi_idx_opt_skew_pct,
    MAX(CASE WHEN participant_type='FII'    AND metric_type='OI'  THEN stk_fut_net_pct END) AS fii_oi_stk_fut_net_pct,
    MAX(CASE WHEN participant_type='Client' AND metric_type='OI'  THEN stk_fut_net_pct END) AS client_oi_stk_fut_net_pct,
    MAX(CASE WHEN participant_type='FII'    AND metric_type='VOL' THEN idx_fut_net_pct END) AS fii_vol_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='Client' AND metric_type='VOL' THEN idx_fut_net_pct END) AS client_vol_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='Pro'    AND metric_type='VOL' THEN idx_fut_net_pct END) AS pro_vol_idx_fut_net_pct,
    MAX(CASE WHEN participant_type='FII'    AND metric_type='VOL' THEN idx_opt_skew_pct END) AS fii_vol_idx_opt_skew_pct,
    MAX(CASE WHEN participant_type='Client' AND metric_type='VOL' THEN idx_opt_skew_pct END) AS client_vol_idx_opt_skew_pct
FROM derived
GROUP BY trade_date
;
"""

# ── SQL: main daily_features table ──────────────────────────────────────────
SQL_DAILY_FEATURES = """
CREATE OR REPLACE TABLE daily_features AS
SELECT
    fr.trade_date,
    fr.close             AS nifty_close,
    fr.fwd_ret_1d,
    fr.fwd_ret_5d,
    fr.fwd_ret_20d,
    fr.up_1d,

    vix.close            AS vix_close,
    pcr_agg.pcr,
    mp_agg.max_pain_dist_pct,

    fut.basis,
    fut.cost_of_carry,
    fut.chng_oi_per      AS fut_chng_oi_pct,

    fv.underlying_daily_vol,
    fv.futures_daily_vol,
    fv.applicable_daily_vol,

    mab.advances,
    mab.declines,
    ROUND(mab.advances::DOUBLE / NULLIF(mab.declines, 0), 4) AS adv_decl_ratio,
    mab.price_band_hits,

    mas.traded_value_cr,
    mas.num_trades,
    mas.market_cap_cr,

    ROUND((fii_long_fut.contracts - fii_short_fut.contracts) /
          NULLIF(fii_long_fut.contracts + fii_short_fut.contracts, 0) * 100, 2) AS fii_fut_net_pct,
    ROUND((cli_long_fut.contracts - cli_short_fut.contracts) /
          NULLIF(cli_long_fut.contracts + cli_short_fut.contracts, 0) * 100, 2) AS client_fut_net_pct,
    ROUND((fii_long_pe.contracts - fii_short_pe.contracts) /
          NULLIF(fii_long_pe.contracts + fii_short_pe.contracts, 0) * 100, 2) AS fii_pe_net_pct,

    fii_flow.fii_net_flow_cr

FROM forward_returns fr

LEFT JOIN nse.market_activity_index vix
    ON vix.trade_date = fr.trade_date AND vix.index_name = 'INDIA VIX'

LEFT JOIN nse.fo_volatility fv
    ON fv.trade_date = fr.trade_date AND fv.ticker = 'NIFTY'

LEFT JOIN nse.market_activity_breadth mab
    ON mab.trade_date = fr.trade_date

LEFT JOIN nse.market_activity_summary mas
    ON mas.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, ROUND(SUM(pe_oi) / NULLIF(SUM(ce_oi), 0), 4) AS pcr
    FROM nse.options_analytics
    WHERE ticker = 'NIFTY' AND instrument_type = 'IDO'
    GROUP BY trade_date
) pcr_agg ON pcr_agg.trade_date = fr.trade_date

LEFT JOIN (
    SELECT ranked.trade_date,
        ROUND(
            SUM(((ranked.underlying_price - ranked.max_pain) / NULLIF(ranked.max_pain, 0) * 100) * (ranked.pe_oi + ranked.ce_oi))
            / NULLIF(SUM(ranked.pe_oi + ranked.ce_oi), 0)
        , 4) AS max_pain_dist_pct
    FROM (
        SELECT oa.trade_date, oa.expiry, oa.max_pain, oa.pe_oi, oa.ce_oi, oa.underlying_price,
            ROW_NUMBER() OVER (PARTITION BY oa.trade_date ORDER BY (oa.pe_oi + oa.ce_oi) DESC) AS liquidity_rank
        FROM nse.options_analytics oa
        WHERE oa.ticker = 'NIFTY' AND oa.instrument_type = 'IDO'
    ) ranked
    WHERE ranked.liquidity_rank <= 3
    GROUP BY ranked.trade_date
) mp_agg ON mp_agg.trade_date = fr.trade_date

LEFT JOIN (
    SELECT fa.trade_date, fa.basis, fa.cost_of_carry, fa.chng_oi_per
    FROM nse.futures_analytics fa
    WHERE fa.ticker = 'NIFTY' AND fa.instrument_type = 'IDF'
    AND (fa.trade_date, fa.expiry) IN (
        SELECT trade_date, MIN(expiry)
        FROM nse.futures_analytics
        WHERE ticker = 'NIFTY' AND instrument_type = 'IDF'
        GROUP BY trade_date
    )
) fut ON fut.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'FII' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'long' AND option_side = 'NA'
) fii_long_fut ON fii_long_fut.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'FII' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'short' AND option_side = 'NA'
) fii_short_fut ON fii_short_fut.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'Client' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'long' AND option_side = 'NA'
) cli_long_fut ON cli_long_fut.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'Client' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'short' AND option_side = 'NA'
) cli_short_fut ON cli_short_fut.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'FII' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'long' AND option_side = 'PE'
) fii_long_pe ON fii_long_pe.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, contracts FROM nse.participant_activity
    WHERE participant_type = 'FII' AND metric_type = 'OI'
      AND asset_class = 'INDEX' AND direction = 'short' AND option_side = 'PE'
) fii_short_pe ON fii_short_pe.trade_date = fr.trade_date

LEFT JOIN (
    SELECT trade_date, buy_amount_cr, sell_amount_cr,
           ROUND(buy_amount_cr - sell_amount_cr, 2) AS fii_net_flow_cr
    FROM nse.fii_stats
    WHERE instrument = 'INDEX FUTURES'
) fii_flow ON fii_flow.trade_date = fr.trade_date

ORDER BY fr.trade_date
;
"""

SQL_ADD_PARTICIPANT_COLUMNS = """
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS dii_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS pro_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS client_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS dii_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS pro_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_stk_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS client_stk_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_vol_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS client_vol_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS pro_vol_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_vol_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS client_vol_opt_skew_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_client_divergence DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_stats_fut_net_pct DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_stats_oi_net_cr DOUBLE;
"""

SQL_FILL_PARTICIPANT_COLUMNS = """
UPDATE daily_features df
SET dii_fut_net_pct         = pp.dii_oi_idx_fut_net_pct,
    pro_fut_net_pct         = pp.pro_oi_idx_fut_net_pct,
    fii_opt_skew_pct        = pp.fii_oi_idx_opt_skew_pct,
    client_opt_skew_pct     = pp.client_oi_idx_opt_skew_pct,
    dii_opt_skew_pct        = pp.dii_oi_idx_opt_skew_pct,
    pro_opt_skew_pct        = pp.pro_oi_idx_opt_skew_pct,
    fii_stk_fut_net_pct     = pp.fii_oi_stk_fut_net_pct,
    client_stk_fut_net_pct  = pp.client_oi_stk_fut_net_pct,
    fii_vol_fut_net_pct     = pp.fii_vol_idx_fut_net_pct,
    client_vol_fut_net_pct  = pp.client_vol_idx_fut_net_pct,
    pro_vol_fut_net_pct     = pp.pro_vol_idx_fut_net_pct,
    fii_vol_opt_skew_pct    = pp.fii_vol_idx_opt_skew_pct,
    client_vol_opt_skew_pct = pp.client_vol_idx_opt_skew_pct,
    fii_client_divergence   = ROUND(pp.fii_oi_idx_fut_net_pct - pp.client_oi_idx_fut_net_pct, 2)
FROM participant_pivot pp
WHERE df.trade_date = pp.trade_date;
"""

SQL_FILL_FII_STATS = """
UPDATE daily_features df
SET fii_stats_fut_net_pct = ROUND(
        (fs.buy_contracts - fs.sell_contracts)::DOUBLE
        / NULLIF(fs.buy_contracts + fs.sell_contracts, 0) * 100
    , 2),
    fii_stats_oi_net_cr = fs.oi_amount_cr
FROM (
    SELECT trade_date, buy_contracts, sell_contracts, oi_amount_cr
    FROM nse.fii_stats
    WHERE instrument = 'INDEX FUTURES'
) fs
WHERE df.trade_date = fs.trade_date;
"""

# One-off patches for two known single-day data gaps around market holidays
# (2021-03-31 vol fields, 2016-02-10 FII/client positioning) — same dates and
# same "average of the neighbouring trading days" patch as the notebook.
SQL_PATCH_GAPS = """
UPDATE daily_features
SET underlying_daily_vol = (SELECT AVG(underlying_daily_vol) FROM daily_features
                             WHERE trade_date IN ('2021-03-30', '2021-04-01')),
    futures_daily_vol    = (SELECT AVG(futures_daily_vol) FROM daily_features
                             WHERE trade_date IN ('2021-03-30', '2021-04-01')),
    applicable_daily_vol = (SELECT AVG(applicable_daily_vol) FROM daily_features
                             WHERE trade_date IN ('2021-03-30', '2021-04-01'))
WHERE trade_date = '2021-03-31' AND underlying_daily_vol IS NULL;

UPDATE daily_features
SET fii_fut_net_pct    = (SELECT AVG(fii_fut_net_pct) FROM daily_features
                           WHERE trade_date IN ('2016-02-09', '2016-02-11')),
    client_fut_net_pct = (SELECT AVG(client_fut_net_pct) FROM daily_features
                           WHERE trade_date IN ('2016-02-09', '2016-02-11')),
    fii_pe_net_pct      = (SELECT AVG(fii_pe_net_pct) FROM daily_features
                           WHERE trade_date IN ('2016-02-09', '2016-02-11'))
WHERE trade_date = '2016-02-10' AND fii_fut_net_pct IS NULL;
"""

SQL_ADD_DELTA_COLUMNS = """
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS vix_chg_5d DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS pcr_chg_5d DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS basis_chg_5d DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS max_pain_dist_chg_5d DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS vix_realized_spread DOUBLE;
ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS fii_fut_net_chg_5d DOUBLE;
"""

SQL_FILL_DELTA_COLUMNS = """
UPDATE daily_features df
SET vix_chg_5d          = sub.vix_chg_5d,
    pcr_chg_5d           = sub.pcr_chg_5d,
    basis_chg_5d         = sub.basis_chg_5d,
    max_pain_dist_chg_5d = sub.max_pain_dist_chg_5d,
    vix_realized_spread  = sub.vix_realized_spread,
    fii_fut_net_chg_5d   = sub.fii_fut_net_chg_5d
FROM (
    SELECT
        trade_date,
        vix_close - LAG(vix_close, 5) OVER (ORDER BY trade_date) AS vix_chg_5d,
        pcr - LAG(pcr, 5) OVER (ORDER BY trade_date) AS pcr_chg_5d,
        basis - LAG(basis, 5) OVER (ORDER BY trade_date) AS basis_chg_5d,
        max_pain_dist_pct - LAG(max_pain_dist_pct, 5) OVER (ORDER BY trade_date) AS max_pain_dist_chg_5d,
        vix_close - (applicable_daily_vol * SQRT(252) * 100) AS vix_realized_spread,
        fii_fut_net_pct - LAG(fii_fut_net_pct, 5) OVER (ORDER BY trade_date) AS fii_fut_net_chg_5d
    FROM daily_features
) sub
WHERE df.trade_date = sub.trade_date;
"""


def build_daily_features(nse_db_path) -> "duckdb.DuckDBPyRelation":
    """Runs the full extraction pipeline and returns the daily_features table."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{nse_db_path}' AS nse (READ_ONLY)")

    log.info("Building forward_returns ...")
    con.execute(SQL_FORWARD_RETURNS)

    log.info("Building participant_pivot ...")
    con.execute(SQL_PARTICIPANT_PIVOT)

    log.info("Building daily_features (core joins) ...")
    con.execute(SQL_DAILY_FEATURES)

    log.info("Adding + filling participant-derived columns ...")
    con.execute(SQL_ADD_PARTICIPANT_COLUMNS)
    con.execute(SQL_FILL_PARTICIPANT_COLUMNS)
    con.execute(SQL_FILL_FII_STATS)

    log.info("Patching known single-day gaps ...")
    con.execute(SQL_PATCH_GAPS)

    log.info("Adding 5-day engineered deltas ...")
    con.execute(SQL_ADD_DELTA_COLUMNS)
    con.execute(SQL_FILL_DELTA_COLUMNS)

    df = con.execute("SELECT * FROM daily_features ORDER BY trade_date").fetchdf()
    con.close()
    return df


def build_monthly_expiries(nse_db_path, feature_dates) -> "pd.DataFrame":
    """
    Same logic as modelling_20d.ipynb cell 4: distinct NIFTY expiries in
    range, collapsed to the last expiry of each calendar month ("the
    monthly", still true after weekly expiries started). Exported separately
    so the fold builder (src/folds.py) never needs nse.db access downstream
    of this script — train.py / fit_final.py only read this parquet.
    """
    import pandas as pd

    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{nse_db_path}' AS nse (READ_ONLY)")
    expiries = con.execute("""
        SELECT DISTINCT expiry FROM nse.instruments
        WHERE ticker = 'NIFTY' AND expiry IS NOT NULL
        ORDER BY expiry
    """).fetchdf()
    con.close()

    expiries["expiry"] = pd.to_datetime(expiries["expiry"])
    lo, hi = feature_dates.min(), feature_dates.max()
    expiries_in_range = expiries[(expiries["expiry"] >= lo) & (expiries["expiry"] <= hi)].copy()
    expiries_in_range["year_month"] = expiries_in_range["expiry"].dt.to_period("M")
    monthly = expiries_in_range.groupby("year_month")["expiry"].max().reset_index(drop=True)
    return monthly.to_frame(name="expiry")


def main():
    if not config.NSE_DB_PATH.exists():
        raise FileNotFoundError(
            f"nse.db not found at {config.NSE_DB_PATH}. Set NSE_DB_PATH in .env, "
            "or build/download it from the candL project first."
        )

    df = build_daily_features(config.NSE_DB_PATH)
    log.info(f"daily_features shape: {df.shape}")

    monthly_expiries = build_monthly_expiries(config.NSE_DB_PATH, df["trade_date"])
    log.info(f"monthly_expiries: {len(monthly_expiries)} boundaries")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FEATURES_PATH, index=False)
    monthly_expiries.to_parquet(config.PROCESSED_DIR / "monthly_expiries.parquet", index=False)
    log.info(f"Wrote {config.FEATURES_PATH} and monthly_expiries.parquet")


if __name__ == "__main__":
    main()
