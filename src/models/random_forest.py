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
        model_pipeline = Pipeline([
            ('model',RandomForestClassifier(
                class_weight=None, 
                n_jobs=-1,
                  random_state=42, 
                  bootstrap = True,
                  oob_score = True
                  ))              
        ])
        return model_pipeline 
        
        

    def get_param_grid(self):
        param_grid = {
            'model__n_estimators':[200,400,600],
            'model__max_depth':[4,6,8,None],
            'model__min_samples_leaf':[1,5,10],
            'model__max_features':['sqrt','log2']
            
        }
        return param_grid
