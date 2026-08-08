import sys
from pathlib import Path
import json
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

df = pd.read_parquet("data/processed/daily_features.parquet")

features = [
    "adv_decl_ratio",
    "advances",
    "applicable_daily_vol",
    "basis",
    "basis_chg_5d",
    "client_fut_net_pct",
    "client_opt_skew_pct",
    "client_stk_fut_net_pct",
    "client_vol_fut_net_pct",
    "client_vol_opt_skew_pct",
    "cost_of_carry",
    "declines",
    "dii_fut_net_pct",
    "dii_opt_skew_pct",
    "fii_client_divergence",
    "fii_fut_net_chg_5d",
    "fii_fut_net_pct",
    "fii_net_flow_cr",
    "fii_opt_skew_pct",
    "fii_pe_net_pct",
    "fii_stats_fut_net_pct",
    "fii_stats_oi_net_cr",
    "fii_stk_fut_net_pct",
    "fii_vol_fut_net_pct",
    "fii_vol_opt_skew_pct",
    "fut_chng_oi_pct",
    "futures_daily_vol",
    "max_pain_dist_chg_5d",
    "max_pain_dist_pct",
    "pcr",
    "pcr_chg_5d",
    "price_band_hits",
    "pro_fut_net_pct",
    "pro_opt_skew_pct",
    "pro_vol_fut_net_pct",
    "traded_value_cr",
    "underlying_daily_vol",
    "vix_chg_5d",
    "vix_close",
    "vix_realized_spread",
]

# API smoke test: select a different valid historical observation
# on each run and send it to the prediction endpoint.
valid_df = df.dropna(subset=features)
row = valid_df.sample(n=1).iloc[0]

trade_date = row["trade_date"]
payload = {col: float(row[col]) for col in features}

print("=" * 60)
print("API SMOKE TEST")
print("=" * 60)
print(f"Selected historical trade date: {trade_date}")
print(f"Endpoint: http://localhost:8000/predict")
print("Sending prediction request...")

r = requests.post(
    "http://localhost:8000/predict",
    json=payload,
)

print(f"HTTP status: {r.status_code}")
print("Response:")
print(json.dumps(r.json(), indent=2))
print("=" * 60)