"""
train.py — walk-forward cross-validation + MLflow experiment tracking.

    Dataset (DVC) -> walk-forward CV (4 models x 2 feature variants)
                  -> MLflow (per-fold + summary metrics for every run)
                  -> models/best_config.json  (winning config for fit_final.py)

No model is fit on the full dataset here, and nothing is registered to the
MLflow Model Registry here — that only happens in fit_final.py + promote.py,
after this script has picked a winner by walk-forward IC. This script never
touches nse.db; it only reads the DVC-tracked feature snapshot.
"""
import os
import json
import logging

import mlflow
import pandas as pd
from joblib import Parallel, delayed

from src import config, garch, utils
from src.folds import build_folds
from src.models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_fold(model_name, feat_cols, train_df, test_df, garch_train_feat=None, garch_test_feat=None):
    train, test = train_df.copy(), test_df.copy()
    cols = feat_cols.copy()

    if garch_train_feat is not None:
        train = train.join(garch_train_feat)
        test = test.join(garch_test_feat)
        train = train.dropna(subset=config.GARCH_FEATURE_COLS)
        cols = cols + [c for c in config.GARCH_FEATURE_COLS if c not in cols]

    X_train, y_train = train[cols], train[config.TARGET]
    X_test, y_test = test[cols], test[config.TARGET]

    if len(X_train) < 30 or len(X_test) < 2:
        return None

    model = get_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {"fold": None, **utils.ic_metrics(preds, y_test), "n_test": len(y_test)}


def _run_one_fold_job(f, model_name, variant_name, feat_cols, model_df, garch_by_fold):
    """Top-level function (not a closure) so joblib can pickle it for
    process-based parallelism, matching the Parallel(delayed(...)) pattern
    used across all fold loops in the research notebooks."""
    use_garch = variant_name.endswith("garch")
    train_df = model_df.loc[f.train_idx]
    test_df = model_df.loc[f.test_idx]
    garch_train, garch_test = (garch_by_fold[f.fold_id] if use_garch else (None, None))

    result = run_fold(model_name, feat_cols, train_df, test_df, garch_train, garch_test)
    if result is None:
        return None
    result["fold"] = f.fold_id
    result["test_start"] = str(f.test_start.date())
    return result


def run_experiment(model_name, variant_name, feat_cols, model_df, folds, garch_by_fold, n_jobs=-1):
    raw_rows = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold_job)(f, model_name, variant_name, feat_cols, model_df, garch_by_fold)
        for f in folds
    )
    rows = [r for r in raw_rows if r is not None]

    if not rows:
        return None, []

    summary = utils.summarize_fold_results(rows)
    return summary, rows


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    log.info("Loading feature snapshot ...")
    model_df = utils.load_model_df()
    log.info(f"model_df shape: {model_df.shape}, range {model_df.index.min()} -> {model_df.index.max()}")

    folds = build_folds(model_df)
    log.info(f"Built {len(folds)} walk-forward folds")

    log.info("Precomputing per-fold GARCH features ...")
    garch_by_fold = garch.fit_garch_for_all_folds(model_df, folds)

    all_results = []
    for model_name in config.MODEL_PARAMS:
        for variant_name, feat_cols in config.FEATURE_VARIANTS.items():
            run_name = f"{model_name}__{variant_name}"
            log.info(f"Running {run_name} ({len(folds)} folds) ...")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("feature_variant", variant_name)
                mlflow.log_param("n_features", len(feat_cols))
                mlflow.log_param("target", config.TARGET)
                mlflow.log_param("n_folds_requested", len(folds))
                mlflow.log_params({f"model__{k}": v for k, v in config.MODEL_PARAMS[model_name].items()})

                summary, fold_rows = run_experiment(model_name, variant_name, feat_cols, model_df, folds, garch_by_fold)

                if summary is None:
                    mlflow.set_tag("status", "no_valid_folds")
                    log.warning(f"{run_name}: no valid folds produced, skipping")
                    continue

                mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})

                fold_df = pd.DataFrame(fold_rows)
                fold_csv = f"/tmp/{run_name}_fold_results.csv"
                os.makedirs(os.path.dirname(fold_csv), exist_ok=True)
                fold_df.to_csv(fold_csv, index=False)
                mlflow.log_artifact(fold_csv)

                meets_threshold = (
                    summary["mean_ic_pearson"] >= config.PROMOTION_MIN_MEAN_IC
                    and summary["positive_fold_pct"] >= config.PROMOTION_MIN_POSITIVE_FOLD_PCT
                )
                mlflow.set_tag("meets_promotion_threshold", str(meets_threshold))

                run_id = mlflow.active_run().info.run_id
                all_results.append({
                    "run_id": run_id,
                    "model_name": model_name,
                    "feature_variant": variant_name,
                    "feature_cols": feat_cols,
                    "meets_threshold": meets_threshold,
                    **summary,
                })

                log.info(f"{run_name}: mean_ic={summary['mean_ic_pearson']:.4f} "
                         f"(t={summary['ic_t_stat']}), positive_fold_pct={summary['positive_fold_pct']:.3f}")

    if not all_results:
        raise RuntimeError("No experiments produced valid results.")

    results_df = pd.DataFrame(all_results).sort_values("mean_ic_pearson", ascending=False)
    cv_metrics = {}

    for _, row in results_df.iterrows():
        key = f"{row['model_name']}__{row['feature_variant']}"
        cv_metrics[key] = {"mean_ic_pearson": float(row["mean_ic_pearson"]), "positive_fold_pct": float(row["positive_fold_pct"]), "n_folds": int(row["n_folds"]), "ic_t_stat": float(row["ic_t_stat"]), "meets_threshold": bool(row["meets_threshold"])}

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.MODELS_DIR / "cv_metrics.json", "w") as fh:
        json.dump(cv_metrics, fh, indent=2)
    log.info(f"Wrote {config.MODELS_DIR / 'cv_metrics.json'}")

    cv_metrics_df = results_df[["model_name", "feature_variant", "mean_ic_pearson", "positive_fold_pct", "ic_t_stat", "n_folds", "meets_threshold"]].copy()
    cv_metrics_df.columns = ["model", "feature_variant", "mean_ic_pearson", "positive_fold_pct", "ic_t_stat", "n_folds", "meets_threshold"]
    cv_metrics_df.to_csv(config.MODELS_DIR / "cv_metrics.csv", index=False)
    log.info(f"Wrote {config.MODELS_DIR / 'cv_metrics.csv'}")

    log.info("\n" + results_df[["model_name", "feature_variant", "mean_ic_pearson",
                                  "positive_fold_pct", "meets_threshold"]].to_string(index=False))

    eligible = results_df[results_df["meets_threshold"]]
    if eligible.empty:
        log.warning("No experiment met the promotion threshold this run. "
                     "Writing best-by-IC anyway for inspection, but fit_final.py "
                     "will refuse to promote it.")
        best = results_df.iloc[0]
    else:
        best = eligible.iloc[0]

    best_config = {
        "run_id": best["run_id"],
        "model_name": best["model_name"],
        "feature_variant": best["feature_variant"],
        "feature_cols": best["feature_cols"],
        "mean_ic_pearson": best["mean_ic_pearson"],
        "positive_fold_pct": best["positive_fold_pct"],
        "meets_threshold": bool(best["meets_threshold"]),
        "n_folds": int(best["n_folds"]),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "best_config.json", "w") as fh:
        json.dump(best_config, fh, indent=2)
    log.info(f"Wrote {config.MODELS_DIR / 'best_config.json'}: {best_config['model_name']}"
              f" / {best_config['feature_variant']} (mean_ic={best_config['mean_ic_pearson']:.4f})")


if __name__ == "__main__":
    main()
