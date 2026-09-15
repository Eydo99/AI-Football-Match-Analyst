"""
Gradient-Boosted Trees (XGBoost) baseline for xG modeling -- closest to
production xG models.

No feature scaling needed. Fill in the TODOs below.
"""
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseXGModel


class XGBoostModel(BaseXGModel): 
    """Gradient-boosted trees baseline. See BaseXGModel for train()/predict_proba()."""

    def build_pipeline(self): # this method returns a pipeline that includes the XGBoost classifier
        """Build the XGBoost pipeline.

        TODO: confirm whether any preprocessing steps belong here.
        """
        

    def get_param_grid(self):
        """Return the hyperparameter grid for GridSearchCV.

        TODO: fill in the values to search over, e.g.
            'clf__n_estimators': [...],
            'clf__max_depth': [...],
            'clf__learning_rate': [...],
            'clf__subsample': [...],
            'clf__colsample_bytree': [...],
            'clf__min_child_weight': [...],
        """
        param_grid = {
            # TODO
        }
        return param_grid
