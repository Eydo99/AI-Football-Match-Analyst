# preserve the public names after the xg classes were renamed.
from src.models.xg_model.xg_logistic_regression import LogisticRegressionXGModel as LogisticRegressionModel
from src.models.xg_model.xg_random_forest import RandomForestXGModel as RandomForestModel
from src.models.xg_model.xgboost_model import XGBoostXGModel as XGBoostModel

__all__ = [
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
]
