"""
Random Forest baseline for xG modeling.

No feature scaling needed, but the estimator is still wrapped in a
Pipeline for a consistent interface with the other models. Fill in
the TODOs below.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseXGModel


class RandomForestModel(BaseXGModel):
    """Bagged trees baseline. See BaseXGModel for train()/predict_proba()."""

    def build_pipeline(self): # this method returns a pipeline that includes the Random Forest classifier   
        """Build the Random Forest pipeline.

        TODO: confirm whether any preprocessing steps belong here.
        """
        

    def get_param_grid(self):
        """Return the hyperparameter grid for GridSearchCV.

        TODO: fill in the values to search over, e.g.
            'clf__n_estimators': [...],
            'clf__max_depth': [...],
            'clf__min_samples_leaf': [...],
            'clf__max_features': [...],
        """
        param_grid = {
            # TODO
        }
        return param_grid
