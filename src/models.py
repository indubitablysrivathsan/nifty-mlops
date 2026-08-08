"""Model factory — same 4 families and fixed hyperparameters as modelling_20d.ipynb."""
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src import config

MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestRegressor(**config.MODEL_PARAMS["random_forest"]),
    "xgboost": lambda: XGBRegressor(**config.MODEL_PARAMS["xgboost"]),
    "lightgbm": lambda: LGBMRegressor(**config.MODEL_PARAMS["lightgbm"]),
    "catboost": lambda: CatBoostRegressor(**config.MODEL_PARAMS["catboost"]),
}


def get_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]()
