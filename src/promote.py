"""
Promotion gate. A challenger config (from train.py's best_config.json) is
eligible for the MLflow Model Registry / Production stage only if ALL hold:

  1. It completed the full walk-forward evaluation (best_config.json exists
     and has a fold count > 0 — enforced upstream by train.py).
  2. mean_ic_pearson >= config.PROMOTION_MIN_MEAN_IC
  3. positive_fold_pct >= config.PROMOTION_MIN_POSITIVE_FOLD_PCT
  4. mean_ic_pearson is not worse than the CURRENT Production model's
     mean_ic_pearson (as recorded on that model version's tags) by more
     than config.PROMOTION_REGRESSION_TOLERANCE. If there is no current
     Production model yet, this condition is automatically satisfied.

If any condition fails, the run is left as an MLflow experiment only —
fit_final.py will not export a new models/model.pkl and nothing is
registered or transitioned to Production.
"""
import logging

import mlflow
from mlflow.tracking import MlflowClient

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_current_production_ic() -> float | None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(
            config.MLFLOW_REGISTRY_MODEL_NAME, stages=[config.MLFLOW_MODEL_STAGE_PRODUCTION]
        )
    except Exception:
        return None

    if not versions:
        return None

    tags = versions[0].tags or {}
    ic = tags.get("mean_ic_pearson")
    return float(ic) if ic is not None else None


def evaluate_gate(challenger: dict) -> tuple[bool, list[str]]:
    """challenger: the dict loaded from models/best_config.json"""
    reasons = []

    if challenger.get("n_folds", 0) <= 0:
        reasons.append("walk-forward evaluation did not complete (0 folds)")

    mean_ic = challenger.get("mean_ic_pearson")
    if mean_ic is None or mean_ic < config.PROMOTION_MIN_MEAN_IC:
        reasons.append(
            f"mean_ic_pearson={mean_ic} below threshold {config.PROMOTION_MIN_MEAN_IC}"
        )

    pos_pct = challenger.get("positive_fold_pct")
    if pos_pct is None or pos_pct < config.PROMOTION_MIN_POSITIVE_FOLD_PCT:
        reasons.append(
            f"positive_fold_pct={pos_pct} below threshold {config.PROMOTION_MIN_POSITIVE_FOLD_PCT}"
        )

    current_ic = get_current_production_ic()
    if current_ic is not None and mean_ic is not None:
        if mean_ic < current_ic - config.PROMOTION_REGRESSION_TOLERANCE:
            reasons.append(
                f"mean_ic_pearson={mean_ic:.4f} is worse than current Production "
                f"model's {current_ic:.4f} (tolerance={config.PROMOTION_REGRESSION_TOLERANCE})"
            )

    passed = len(reasons) == 0
    return passed, reasons


if __name__ == "__main__":
    import json

    with open(config.MODELS_DIR / "best_config.json") as fh:
        challenger = json.load(fh)

    passed, reasons = evaluate_gate(challenger)
    if passed:
        log.info("Promotion gate: PASSED")
    else:
        log.warning("Promotion gate: FAILED")
        for r in reasons:
            log.warning(f"  - {r}")
    raise SystemExit(0 if passed else 1)
