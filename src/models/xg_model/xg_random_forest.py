"""
Random Forest baseline for xG modeling.

No feature scaling needed, but the estimator is still wrapped in a
Pipeline for a consistent interface with the other models.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.xg_model.xg_base_model import XGBaseModel


class RandomForestXGModel(XGBaseModel):
    """Bagged trees baseline. See XGBaseModel for train()/predict_proba()/evaluate()."""

    def build_pipeline(self):
        return Pipeline([
            ('model', RandomForestClassifier(
                class_weight=None,
                n_jobs=-1,
                random_state=42,
                bootstrap=True,
                oob_score=True,
            ))
        ])

    def get_param_grid(self):
        return {
            'model__n_estimators': [200, 400, 600],
            'model__max_depth': [4, 6, 8, None],
            'model__min_samples_leaf': [1, 5, 10],
            'model__max_features': ['sqrt', 'log2'],
        }