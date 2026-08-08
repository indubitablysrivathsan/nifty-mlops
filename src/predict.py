"""
predict.py — loads models/model.pkl and exposes predict() for src/app.py.

If the production model uses GARCH features, they are computed once from
the bundled historical feature snapshot when the predictor is initialized.
No training occurs during prediction requests.
"""

import pickle

import pandas as pd

from src import config, garch


class ModelPredictor:
    def __init__(self, model_path=None):
        model_path = model_path or config.FINAL_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python -m src.train` then "
                "`python -m src.fit_final` first."
            )

        with open(model_path, "rb") as fh:
            self.artifact = pickle.load(fh)

        self.model = self.artifact["model"]
        self.feature_cols = self.artifact["feature_cols"]
        self.uses_garch = self.artifact["uses_garch"]

        self.garch_features = None

        if self.uses_garch:
            df = pd.read_parquet(config.FEATURES_PATH)
            df = df.sort_index()

            garch_features, _ = garch.fit_garch_final(df)
            self.garch_features = garch_features.iloc[-1]

    def _fill_garch(self, features: dict) -> dict:
        features = dict(features)

        if not self.uses_garch:
            return features

        if self.garch_features is None:
            raise RuntimeError("GARCH features were not initialized.")

        features["garch_sigma"] = float(self.garch_features["garch_sigma"])
        features["garch_sigma_chg"] = float(self.garch_features["garch_sigma_chg"])

        return features

    def predict_one(self, features: dict) -> float:
        if self.uses_garch:
            features = self._fill_garch(features)

        missing = [c for c in self.feature_cols if c not in features]

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = pd.DataFrame([{c: features[c] for c in self.feature_cols}])
        pred = self.model.predict(X)[0]

        return float(pred)


_predictor_singleton = None


def get_predictor() -> ModelPredictor:
    global _predictor_singleton

    if _predictor_singleton is None:
        _predictor_singleton = ModelPredictor()

    return _predictor_singleton