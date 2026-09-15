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
        pipeline_lr=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000 , class_weight=None, random_state=42))
        ])
        return pipeline_lr

        

    def get_param_grid(self):
        param_grid_lr={
            'clf__C': [0.001, 0.01 , 0.1 , 1 , 10 , 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs']
         }
        return param_grid_lr
