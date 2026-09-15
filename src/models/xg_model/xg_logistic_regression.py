
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.xg_model.xg_base_model import XGBaseModel


class LogisticRegressionXGModel(XGBaseModel):
    """Interpretable baseline. See XGBaseModel for train()/predict_proba()/evaluate()."""

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight=None, random_state=42))
        ])

    def get_param_grid(self):
        return {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
        }