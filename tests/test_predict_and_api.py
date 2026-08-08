import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.predict import ModelPredictor


@pytest.fixture
def fake_model_path(tmp_path):
    """Builds a tiny, valid model.pkl artifact (no GARCH) for fast tests —
    this does NOT exercise train.py/fit_final.py, just the predict/app
    contract that consumes their output."""
    feature_cols = config.BASE_FEATURE_COLS[:5]
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(50, len(feature_cols)), columns=feature_cols)
    y = rng.randn(50)

    model = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X, y)

    artifact = {
        "model": model,
        "model_name": "random_forest",
        "feature_variant": "baseline_full",
        "feature_cols": feature_cols,
        "target": config.TARGET,
        "uses_garch": False,
        "garch_result": None,
    }
    path = tmp_path / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    return path


def test_predictor_predict_one(fake_model_path):
    predictor = ModelPredictor(model_path=fake_model_path)
    features = {c: 0.1 for c in predictor.feature_cols}
    pred = predictor.predict_one(features)
    assert isinstance(pred, float)


def test_predictor_missing_feature_raises(fake_model_path):
    predictor = ModelPredictor(model_path=fake_model_path)
    features = {c: 0.1 for c in predictor.feature_cols[:-1]}  # drop one
    with pytest.raises(ValueError):
        predictor.predict_one(features)


def test_api_predict_endpoint(fake_model_path, monkeypatch):
    import src.predict as predict_module

    predict_module._predictor_singleton = None
    monkeypatch.setattr(config, "FINAL_MODEL_PATH", fake_model_path)

    from src.app import app

    client = TestClient(app)

    feature_cols = predict_module.ModelPredictor(model_path=fake_model_path).feature_cols
    payload = {c: 0.1 for c in feature_cols}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["model_name"] == "random_forest"


def test_api_health_endpoint(fake_model_path, monkeypatch):
    import src.predict as predict_module

    predict_module._predictor_singleton = None
    monkeypatch.setattr(config, "FINAL_MODEL_PATH", fake_model_path)

    from src.app import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
