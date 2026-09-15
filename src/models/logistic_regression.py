"""
Logistic Regression baseline for xG modeling.

Needs scaled features, so the scaler stays inside the pipeline with the
classifier. Fill in the TODOs below.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseXGModel


class LogisticRegressionModel(BaseXGModel):
    """Interpretable baseline. See BaseXGModel for train()/predict_proba()."""

    def build_pipeline(self): # this method returns a pipeline that includes the scaler and the Logistic Regression classifier
        """Build the scaler + Logistic Regression pipeline.

        TODO: confirm whether any additional preprocessing steps belong here.
        """
        

    def get_param_grid(self):
        """Return the hyperparameter grid for GridSearchCV.

        TODO: fill in the values to search over, e.g.
            'clf__C': [...],
            'clf__penalty': [...],
            'clf__solver': [...],
        """
        param_grid = {
            # TODO
        }
        return param_grid
