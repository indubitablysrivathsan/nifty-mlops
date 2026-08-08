"""
FastAPI prediction service. Loads models/model.pkl at startup (produced
offline by train.py + fit_final.py — no training happens in this process
or in the Docker container it runs in).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src import config
from src.predict import get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_predictor()
        log.info("Model loaded successfully at startup.")
    except FileNotFoundError as e:
        # Don't crash the process — /health will report unhealthy and
        # /predict will 503 until a model is available (e.g. mounted later).
        log.error(f"Model not available at startup: {e}")
    yield


app = FastAPI(
    title="NIFTY 20D Forward Return Model API",
    description="Serves the promoted MLflow-registered model for fwd_ret_20d.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """
    Feature values keyed by name (see /features for the exact required set
    for the currently loaded model). garch_sigma / garch_sigma_chg may be
    omitted if the model uses the GARCH variant — they will be approximated;
    see src/predict.py for the caveat on that approximation.
    """
    model_config = ConfigDict(extra="allow")


class PredictResponse(BaseModel):
    prediction: float
    target: str
    model_name: str
    feature_variant: str


@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        return {
            "status": "ok",
            "model_name": predictor.artifact["model_name"],
            "feature_variant": predictor.artifact["feature_variant"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/features")
def features():
    predictor = get_predictor()
    return {
        "target": config.TARGET,
        "required_features": predictor.feature_cols,
        "uses_garch": predictor.uses_garch,
        "note": "garch_sigma / garch_sigma_chg are optional if uses_garch is true; "
                "they will be approximated from the stored GARCH fit if omitted.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_dict = request.model_dump(exclude_none=True)
    try:
        pred = predictor.predict_one(features_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        prediction=pred,
        target=predictor.artifact["target"],
        model_name=predictor.artifact["model_name"],
        feature_variant=predictor.artifact["feature_variant"],
    )
