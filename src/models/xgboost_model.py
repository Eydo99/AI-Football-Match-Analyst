"""xgboost pipeline and search space for goal probabilities."""
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseXGModel


class XGBoostModel(BaseXGModel):
    """use the shared training and prediction methods with boosted trees."""

    def build_pipeline(self):
        """return a fresh, unfitted pipeline for numeric shot features."""
        # trees do not need scaling, and xgboost handles missing numeric values.
        # keep one worker here because the shared grid search runs fits in parallel.
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            scale_pos_weight=1,
        )
        # the step name must match the clf__ prefix in the parameter grid.
        return Pipeline([("clf", classifier)])

    def get_param_grid(self):
        """return the six-parameter search grid specified in the team guide."""
        # keep the grid focused on probability quality and the guide's ranges.
        # do not rebalance goals; xg needs probabilities at the observed base rate.
        return {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__min_child_weight": [1, 5, 10],
        }
