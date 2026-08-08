"""
fit_final.py — the one place a "final" model is actually produced.

    models/best_config.json (from train.py)
        -> promotion gate (src/promote.py)
        -> refit winning model_name/feature_variant on ALL historical data
        -> models/model.pkl + models/model_meta.json
        -> MLflow Model Registry, transitioned to Production (only if gate passed)

If the gate fails, this script logs why and exits non-zero WITHOUT writing
model.pkl or touching the registry — the previous production artifact (if
any) is left untouched, satisfying "no new production model artifact is
exported" from the promotion policy.

No predictive model training occurs downstream of this script.
app.py / CI only load the promoted XGBoost model; the API may fit the
GARCH feature generator once from the bundled historical snapshot.
"""
import json
import logging
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from src import config, garch, promote, utils
from src.models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    best_config_path = config.MODELS_DIR / "best_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(f"{best_config_path} not found — run `python -m src.train` first.")

    with open(best_config_path) as fh:
        challenger = json.load(fh)

    passed, reasons = promote.evaluate_gate(challenger)
    if not passed:
        log.warning("Promotion gate FAILED — no final model will be exported or registered.")
        for r in reasons:
            log.warning(f"  - {r}")
        raise SystemExit(1)

    log.info(f"Promotion gate PASSED for {challenger['model_name']} / {challenger['feature_variant']} "
              f"(mean_ic={challenger['mean_ic_pearson']:.4f})")

    model_df = utils.load_model_df()
    feature_cols = challenger["feature_cols"]
    use_garch = challenger["feature_variant"].endswith("garch")

    garch_result = None
    if use_garch:
        log.info("Fitting final GARCH(1,1) on full history ...")
        garch_feat, _ = garch.fit_garch_final(model_df)
        model_df = model_df.join(garch_feat)
        model_df = model_df.dropna(subset=config.GARCH_FEATURE_COLS)
        feature_cols = feature_cols  # already includes garch cols from train.py's variant

    X = model_df[feature_cols]
    y = model_df[config.TARGET]

    log.info(f"Refitting {challenger['model_name']} on {len(X)} rows, {len(feature_cols)} features ...")
    model = get_model(challenger["model_name"])
    model.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_name": challenger["model_name"],
        "feature_variant": challenger["feature_variant"],
        "feature_cols": feature_cols,
        "target": config.TARGET,
        "uses_garch": use_garch,
    }
    with open(config.FINAL_MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)
    log.info(f"Wrote {config.FINAL_MODEL_PATH}")

    meta = {
        "model_name": challenger["model_name"],
        "feature_variant": challenger["feature_variant"],
        "feature_cols": feature_cols,
        "target": config.TARGET,
        "uses_garch": use_garch,
        "train_rows": int(len(X)),
        "walk_forward_mean_ic_pearson": challenger["mean_ic_pearson"],
        "walk_forward_positive_fold_pct": challenger["positive_fold_pct"],
        "walk_forward_n_folds": challenger["n_folds"],
        "source_mlflow_run_id": challenger["run_id"],
    }
    with open(config.FINAL_MODEL_META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info(f"Wrote {config.FINAL_MODEL_META_PATH}")

    _register_and_promote(challenger)


def _register_and_promote(challenger: dict):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    with mlflow.start_run(run_name="fit_final") as run:
        mlflow.log_params({
            "model_name": challenger["model_name"],
            "feature_variant": challenger["feature_variant"],
        })
        mlflow.log_metric("mean_ic_pearson", challenger["mean_ic_pearson"])
        mlflow.log_metric("positive_fold_pct", challenger["positive_fold_pct"])
        mlflow.log_artifact(str(config.FINAL_MODEL_PATH))
        mlflow.log_artifact(str(config.FINAL_MODEL_META_PATH))

        model_uri = f"runs:/{run.info.run_id}/{config.FINAL_MODEL_PATH.name}"
        try:
            mv = mlflow.register_model(model_uri, config.MLFLOW_REGISTRY_MODEL_NAME)
        except Exception as e:
            log.error(f"Model registration failed: {e}")
            raise

        client.set_model_version_tag(mv.name, mv.version, "mean_ic_pearson",
                                      str(challenger["mean_ic_pearson"]))
        client.set_model_version_tag(mv.name, mv.version, "positive_fold_pct",
                                      str(challenger["positive_fold_pct"]))
        client.set_model_version_tag(mv.name, mv.version, "source_run_id", challenger["run_id"])

        client.transition_model_version_stage(
            name=mv.name, version=mv.version,
            stage=config.MLFLOW_MODEL_STAGE_PRODUCTION,
            archive_existing_versions=True,
        )
        log.info(f"Registered {mv.name} v{mv.version} and promoted to Production.")


if __name__ == "__main__":
    main()
