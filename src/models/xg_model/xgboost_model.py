"""xgboost pipeline and search space for goal probabilities."""
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.models.xg_model.xg_base_model import XGBaseModel


class XGBoostXGModel(XGBaseModel):
    """Uses the shared training and prediction methods with boosted trees."""

    def build_pipeline(self):
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            scale_pos_weight=1,
        )
        return Pipeline([("clf", classifier)])

    def get_param_grid(self):
        return {
            "clf__n_estimators": [200, 300,400,500],
            "clf__max_depth": [3, 4, 5 ,6],
            "clf__learning_rate": [0.01, 0.05, 0.08 , 0.1],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8,1.0],
            "clf__min_child_weight": [1, 5,8,10],
        }